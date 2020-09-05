from skmultiflow.data import FileStream
from skmultiflow.data import SEAGenerator
from skmultiflow.data import HyperplaneGenerator
from skmultiflow.trees import HoeffdingTree
from skmultiflow.anomaly_detection import HalfSpaceTrees
from skmultiflow.lazy.sam_knn import SAMKNN
from skmultiflow.lazy.knn import KNN
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.evaluation import EvaluateHoldout
from sklearn.cluster import KMeans
import numpy as np
import psycopg2
import os
import sys
import os.path
import traceback
import random
import time
import csv
import json
import re
import subprocess
from subprocess import PIPE
import pandas as pd
from datetime import datetime
 

def main():
    while True:

        # connect to the database
        conn = psycopg2.connect(os.environ['DATABASE_URL'])
        cur = conn.cursor()

        # send a SQL query to see if there's a task in queue
        cur.execute("SELECT id, dataset_name, algorithm_name, evaluation, state, created_at, started_at, updated_at, finished_at, dataset_params, algorithm_params, evaluation_params FROM api_job WHERE state='queued' ORDER BY created_at LIMIT 1")
        result = cur.fetchone()

        # if so, update the task to in progres...
        if result :
            id = result[0]
            dataset_name  = result [1]
            dataset_params = result[9]
            algorithm_name = result[2]
            algorithm_params = result[10]          
            evaluation = result[3]
            eval_params = result[11]

            cur.execute("UPDATE api_job SET state='in_progress', started_at=(%s) WHERE id=%s",[datetime.now(), id])
            conn.commit()

            try:
                # run the evaluation with the given dataset, params, algo, etc.
                results, data_summary = prepareForRun(id, dataset_name, algorithm_name, dataset_params, algorithm_params, evaluation, eval_params)

                # update  json results
                cur.execute("UPDATE api_job SET results=(%s), data_summary=(%s) WHERE id=%s",([results,data_summary,id]))
                
                # update the results on the way, mark the task finished after done
                cur.execute("UPDATE api_job SET state='finished', finished_at=(%s) WHERE id=%s",[datetime.now(),id])
            
            except Exception as e:
                print("Failed process: id-" + str(id))
                traceback.print_exc()
                cur.execute("UPDATE api_job SET state='failed', finished_at=(%s) WHERE id=%s",[datetime.now(),id])

            finally:
                conn.commit()

            try:
                if(os.path.isfile(getFile(id))):
                    os.remove(getFile(id))
            except Exception as e:
                print("Failed removing file: " + str(id))
                print(str(e))
        else:
            print("Wait for new job...")

        cur.close()
        conn.close()

        # sleep 
        time.sleep(10)

    exit(0)


def prepareForRun(id, dataset, algo_name, dataset_params, algo_params, evaluation, eval_params):
    headers = ""
    data_summary = {}
    if('noise_percentage' in dataset_params): # generated 
        if('n_drift_features' in dataset_params): #hyperplane
            stream = HyperplaneGenerator(random_state = None, 
                                n_features = int(dataset_params['n_features']), 
                                n_drift_features = int(dataset_params['n_drift_features']), 
                                mag_change = float(dataset_params['mag_change']), 
                                noise_percentage = float(dataset_params['noise_percentage']), 
                                sigma_percentage = float(dataset_params['sigma_percentage']))
        else: #sea
            stream = SEAGenerator(classification_function = 0, 
                                    random_state = None, 
                                    balance_classes = False, 
                                    noise_percentage = float(dataset_params['noise_percentage']))
    else: 
        used_dataset = prepareDataset(dataset, dataset_params)
        frame = pd.read_csv(used_dataset)
        headers = (frame).columns.tolist()
        # headers = (pd.read_csv(used_dataset, index_col=0)).columns.tolist()
        data_summary = pd.read_csv(used_dataset, index_col=0).describe().to_json()
       
        stream = FileStream(used_dataset)
    stream.prepare_for_use()    

    if algo_name == "hoeffding_tree":
        run_hoeffdingtree(getFile(id),stream, algo_params, evaluation, eval_params)
        with open(getFile(id)) as file: 
            out = readAndParseResults(file)
            
    elif algo_name =="samknn":
        samknn_result = run_samknn(getFile(id),stream, algo_params, evaluation, eval_params)
        with open(getFile(id)) as file: 
            out = readAndParseResults(file) 
    
    elif algo_name =="knn":
        if('sample_size' in dataset_params):
            sample_size = int(dataset_params['sample_size'])
        else:
            sample_size = 300
        if('start_value' in dataset_params): 
            if('stop_value' in dataset_params): 
                sample_size = int(dataset_params['stop_value']) - int(dataset_params['start_value'])
            
        knn_result = run_knn(getFile(id),stream, headers, sample_size, algo_params, evaluation, eval_params) 
        out = knn_result.to_json(orient='records')       

    elif algo_name == "k_means":
        kmeans_result = run_kmeans(used_dataset, algo_params)
        data = pd.read_csv(dataset+".csv")
        sub_data = data[int(dataset_params['start_value']):int(dataset_params['stop_value'])]
        res = sub_data.merge(pd.Series(kmeans_result).rename('cluster'), left_index=True, right_index=True)
        out = res.to_json(orient='records')
    
    elif algo_name == "d3":
        d3_result = run_d3(used_dataset, algo_params)
        out = json.dumps( d3_result) 


    elif algo_name == "half_space_tree":
        run_halfspacetree(getFile(id), stream, algo_params, evaluation, eval_params)
        with open(getFile(id)) as file: 
            out = readAndParseResults(file) 
        
    else:
        print("algorithm not found")

    return out, data_summary


def  prepareDataset(dataset, dataset_params):
    if('start_value' in dataset_params and 'stop_value'in dataset_params):
        data = pd.read_csv(dataset+".csv")
        sub_data = data[int(dataset_params['start_value']):int(dataset_params['stop_value'])]
        sub_data.to_csv("subdata.csv")
        used_dataset="subdata.csv"
    else: 
        used_dataset=dataset+".csv"
    return used_dataset


def run_d3(dataset_name,algo_params):
    try:
        process = subprocess.run(['python', "ai_core/D3.py", dataset_name, str(algo_params['w']), str(algo_params['rho']), str(algo_params['auc'])], check=True, stdout=PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    
    pattern = re.compile("<RESULTS_START>(.*)<RESULTS_END>",re.MULTILINE)
    search_results = pattern.search(process.stdout.decode("utf-8"))
    d3_results_json= search_results.group(1) 
    d3_results = json.loads(d3_results_json)

    x_array = d3_results[0]
    acc_array = d3_results[1]

    results = []

    for i in range(0,len(x_array)):
        item = {}
        item["data_percentage"] = float("{:.2f}".format(x_array[i]))
        item["acc"] = float("{:.2f}".format(acc_array[i]))
        results.insert(i,item)
    
    return results  


def run_hoeffdingtree(resultFile,stream,algo_params, evaluation, eval_params):
    ht = HoeffdingTree(grace_period = int(algo_params['grace_period']),
                      tie_threshold = float(algo_params['tie_threshold']),   
                      binary_split = bool(algo_params['binary_split']),
                      remove_poor_atts = bool(algo_params['remove_poor_atts']),
                      no_preprune = bool(algo_params['no_preprune']),
                      leaf_prediction = str(algo_params['leaf_prediction']),
                      nb_threshold = int(algo_params['nb_threshold']))
   
    if evaluation=="holdout": 
        evaluator = EvaluateHoldout(show_plot = False,
                                    max_samples = int(eval_params['max_sample']),
                                    n_wait = int(eval_params['n_wait']),
                                    batch_size = int(eval_params['batch_size']),
                                    metrics = ['accuracy', 'kappa','kappa_t'],
                                    output_file = resultFile)
    elif evaluation=="basic":
        print("Basic evaluation")
        # https://scikit-multiflow.readthedocs.io/en/stable/api/generated/skmultiflow.trees.HoeffdingTreeClassifier.html?highlight=hoeffding
    else:
        evaluator = EvaluatePrequential(show_plot = False,
                                        pretrain_size = int(eval_params['pretrain_size']),
                                        max_samples = int(eval_params['max_sample']),
                                        batch_size = int(eval_params['batch_size']),
                                        n_wait = int(eval_params['n_wait']),
                                        metrics = ['accuracy', 'kappa','kappa_t','kappa_m','true_vs_predicted'],
                                        output_file = resultFile)
    
    # print("evaluate with pretrain size:" + str(eval_params['pretrain_size']) + 
    #         " max sample:" + str(eval_params['max_sample']) + 
    #         " batch size:" + str(eval_params['batch_size']) +
    #         "n_wait:" + str(eval_params['n_wait']))

    evaluator.evaluate(stream=stream, model=ht)

#TODO: bunun sonuclarini alamadik henuz.
def run_halfspacetree(resultFile,stream, algo_params, evaluation, eval_params):
    hst = HalfSpaceTrees(n_features = int(algo_params['n_features']), 
                        window_size = int(algo_params['window_size']),
                        depth = int(algo_params['depth']),
                        n_estimators = int(algo_params['n_estimators']),
                        size_limit = int(algo_params['size_limit']),
                        anomaly_threshold = float(algo_params['anomaly_threshold']),
                        random_state = None)
    if evaluation=="holdout": 
        evaluator = EvaluateHoldout(show_plot = False,
                                    max_samples = int(eval_params['max_sample']),
                                    n_wait = int(eval_params['n_wait']),
                                    batch_size = int(eval_params['batch_size']),
                                    metrics = ['accuracy', 'kappa','kappa_t'],
                                    output_file = resultFile)
    else:
        evaluator = EvaluatePrequential(show_plot=False,
                                        pretrain_size = int(eval_params['pretrain_size']),
                                        max_samples = int(eval_params['max_sample']),
                                        batch_size = int(eval_params['batch_size']),
                                        n_wait = int(eval_params['n_wait']),
                                        metrics = ['accuracy', 'kappa','kappa_t','kappa_m','true_vs_predicted'],
                                        output_file = resultFile)
    
    # print("evaluate with pretrain size:" + str(eval_params['pretrain_size']) + 
    #         " max sample:" + str(eval_params['max_sample']) + 
    #         " batch size:" + str(eval_params['batch_size']) +
    #         "n_wait:" + str(eval_params['n_wait']))



# NOT: samknn sadece prequentialla calisir
def run_samknn(resultFile, stream,algo_params, evaluation, eval_params):
    classifier = SAMKNN(n_neighbors=int(algo_params['neighbors']), 
                        weighting='distance', 
                        max_window_size=int(algo_params['max_window_size']), 
                        stm_size_option='maxACCApprox',
                        use_ltm=False)

       
    evaluator = EvaluatePrequential(pretrain_size = int(eval_params['pretrain_size']), 
                                        max_samples=int(eval_params['max_sample']), 
                                        batch_size= int(eval_params['batch_size']),
                                        n_wait = int(eval_params['n_wait']), 
                                        max_time=1000,
                                        output_file=resultFile, 
                                        metrics=['accuracy', 'kappa'])
   
    evaluator.evaluate(stream=stream, model=classifier)


def run_knn(resultFile, stream,headers, sample_size, algo_params, evaluation, eval_params):
    pretrain_size = int(algo_params['pretrain_size'])
    neighbors = int(algo_params['neighbors'])
    max_window_size = int(algo_params['max_window_size'])
    leaf_size = int(algo_params['leaf_size'])
    
    print("Running knn with parameters sample_size: %d" % sample_size)

    X, y = stream.next_sample(pretrain_size)

    print("Received %d samples by using pretrain size %d" % (len(X), pretrain_size))

    knn = KNN(n_neighbors = neighbors,
              max_window_size = max_window_size, 
              leaf_size = leaf_size,
              nominal_attributes = None)

    print("Created knn model with %d neighbors. max_windows_size %d and lead_size %d." % (neighbors, max_window_size, leaf_size))

    knn.partial_fit(X, y) 

    n_samples = 0
    corrects = 0
    
    clusters=[]
    correctness=[]

    print("Fetching the next %d samples after the fırst %d pretrain samples" % (sample_size, pretrain_size))

    X, y = stream.next_sample(sample_size)

    print("Received %d samples after requesting %d samples" % (len(X), sample_size))

    while n_samples < len(X):
        tX = [X[n_samples]]
        tY = [y[n_samples]]
        my_pred = knn.predict(tX)

        clusters.insert(n_samples,my_pred[0])

        if tY[0] == my_pred[0]:
            corrects += 1
            correctness.insert(n_samples, 1)
        else:
            correctness.insert(n_samples, 0)
       
        knn = knn.partial_fit(tX, tY)
        n_samples += 1


    if headers is "":
        headers=list()
        for i in range(1,len(X[0])+1):
            headers.append("attr"+str(i))
        
    headers.append("found_label")

    print("Headers %d: "%(len(headers)) +" " +str(headers) )
    result = np.concatenate((X, np.array(y)[:,None]), axis=1)
    result = np.concatenate((result, np.array(clusters)[:,None]), axis=1)
    
    return pd.DataFrame(data=result, columns=headers)
 


def run_kmeans(dataset_name, algo_params):
    data = pd.read_csv(dataset_name)
    kmeans = KMeans(
            n_clusters = int(algo_params['n_cluster']), 
            init='k-means++', 
            max_iter = int(algo_params['max_iter']), 
            n_init = int(algo_params['n_init']))
    return kmeans.fit_predict(data.values)


def getFile(id):
    return "result-"+str(id)+".csv"

def readAndParseResults(file):
    # read and filter the comment lines
    reader = csv.DictReader(filter(lambda row: row[0]!='#',file))
    
    # skip header row
    next(reader)
    
    # Parse the CSV into JSON  
    return json.dumps( [ row for row in reader ] )  
            


if __name__ == "__main__":
    main()