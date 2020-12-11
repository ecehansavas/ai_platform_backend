from skmultiflow.data import FileStream
from skmultiflow.data import SEAGenerator
from skmultiflow.data import HyperplaneGenerator
from skmultiflow.trees import HoeffdingTree
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
from subprocess import PIPE, Popen
import pandas as pd
 

class AIJob:
    def __init__(self,id, dataset_name, algorithm_name, dataset_params, algorithm_params, evaluation, eval_params):
        self.id=id
        self.dataset_name = dataset_name
        self.algorithm_name = algorithm_name
        self.dataset_params = dataset_params
        self.algorithm_params = algorithm_params
        self.evaluation = evaluation
        self.eval_params = eval_params
 
    def runTheJob(self):
        headers = ""
        data_summary = {}

        if('noise_percentage' in self.dataset_params): # generated 
            if(isHyperPlaneGenerator(self.dataset_params)): 
                stream = hyperPlaneGeneratorStream(self.dataset_params) 
            else: #sea
                stream = seaGenaratorStream(self.dataset_params)
        else: 
            used_dataset = prepareDataset(self.dataset_name, self.dataset_params)
            frame = pd.read_csv(used_dataset)
            headers = (frame).columns.tolist()
            data_summary = pd.read_csv(used_dataset).describe().to_json()
        
            stream = FileStream(used_dataset)
        stream.prepare_for_use()    

        if self.algorithm_name == "hoeffding_tree":
            basic_result = run_hoeffdingtree(getFile(self.id),stream, self.algorithm_params, self.evaluation, self.eval_params, self.id, self.dataset_params)
            if(os.path.isfile(getFile(self.id))):
                with open(getFile(self.id)) as file: 
                    out = readAndParseResults(file)
            else:
                out = json.dumps(basic_result)
        
        elif self.algorithm_name =="knn":
            if('sample_size' in self.dataset_params):
                sample_size = int(self.dataset_params['sample_size'])
            else:
                sample_size = 300
            if('start_value' in self.dataset_params): 
                if('stop_value' in self.dataset_params): 
                    sample_size = int(self.dataset_params['stop_value']) - int(self.dataset_params['start_value'])
                
            knn_result = run_knn(getFile(self.id),stream, headers, sample_size, self.algorithm_params, self.evaluation, self.eval_params, self.id) 
            out = knn_result.to_json(orient='records')       

        elif self.algorithm_name == "k_means":
            kmeans_result = run_kmeans(used_dataset, self.algorithm_params)
            data = pd.read_csv(self.dataset_name+".csv")
            sub_data = data[int(self.dataset_params['start_value']):int(self.dataset_params['stop_value'])]
            res = sub_data.merge(pd.Series(kmeans_result).rename('cluster'), left_index=True, right_index=True)
            out = res.to_json(orient='records')
        
        elif self.algorithm_name == "d3":
            d3_result = run_d3(used_dataset, self.algorithm_params, self.id)
            out = json.dumps( d3_result) 
        
        elif self.algorithm_name == "clustream":
            clustream_result = run_clustream(used_dataset, self.algorithm_params, self.id)
            out = json.dumps(clustream_result)
        
        elif self.algorithm_name == "denstream":
            denstream_result = run_denstream(used_dataset, self.algorithm_params, self.id)
            out = json.dumps(denstream_result) 

        elif self.algorithm_name == "streamkm":
            streamkm_result = run_streamkm(used_dataset, self.algorithm_params, self.id)
            out = json.dumps(streamkm_result) 
            
        else:
            print("algorithm not found")

        return out, data_summary



# ------------------- DATA PREPARATIONS
def isHyperPlaneGenerator(dataset_params):
    return ('n_drift_features' in dataset_params) #hyperplane

def hyperPlaneGeneratorStream(dataset_params):
     return HyperplaneGenerator(random_state = None, 
                                n_features = int(dataset_params['n_features']), 
                                n_drift_features = int(dataset_params['n_drift_features']), 
                                mag_change = float(dataset_params['mag_change']), 
                                noise_percentage = float(dataset_params['noise_percentage']), 
                                sigma_percentage = float(dataset_params['sigma_percentage']))

def seaGenaratorStream(dataset_params):
    return SEAGenerator(classification_function = 0, 
                        random_state = None, 
                        balance_classes = False, 
                        noise_percentage = float(dataset_params['noise_percentage']))


def prepareDataset(dataset, dataset_params):
    if('start_value' in dataset_params and 'stop_value'in dataset_params):
        data = pd.read_csv(dataset+".csv")
        sub_data = data[int(dataset_params['start_value']):int(dataset_params['stop_value'])]
        sub_data.to_csv("subdata.csv")
        used_dataset="subdata.csv" # eren: create a sub directory for the job and place this subdata.csv there
    else: 
        used_dataset=dataset+".csv"
    return used_dataset

# ------------------- END OF DATA PREPARATIONS

# ------------------- ALGORITHMS 
def run_kmeans(dataset_name, algo_params):
    data = pd.read_csv(dataset_name)
    kmeans = KMeans(
            n_clusters = int(algo_params['n_cluster']), 
            init='k-means++', 
            max_iter = int(algo_params['max_iter']), 
            n_init = int(algo_params['n_init']))
    return kmeans.fit_predict(data.values)


def run_knn(resultFile, stream,headers, sample_size, algo_params, evaluation, eval_params, jobid):
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
        time.sleep(0.1)
        tX = [X[n_samples]]
        tY = [y[n_samples]]
        my_pred = knn.predict(tX)

        clusters.insert(n_samples,my_pred[0])

        if tY[0] == my_pred[0]:
            corrects += 1
            correctness.insert(n_samples, 1)
        else:
            correctness.insert(n_samples, 0)
       
        try:
            accuracy = round((corrects / (n_samples+1)),2)
            print('{} KNN samples analyzed '.format(n_samples) + ' accuracy: {}' .format(accuracy))
        except ZeroDivisionError:
            accuracy = 0
            print('{} KNN samples analyzed '.format(n_samples) + ' accuracy: 0' )

        progress = {}
        progress["n_samples"] = n_samples
        progress["correct_cnt"] = corrects
        progress["accuracy"] = accuracy
        append_progress(jobid, progress)
        
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


def run_hoeffdingtree(resultFile,stream,algo_params, evaluation, eval_params, jobid, dataset_params):
    ht = HoeffdingTree(grace_period = int(algo_params['grace_period']),
                      tie_threshold = float(algo_params['tie_threshold']),   
                      binary_split = bool(algo_params['binary_split']),
                      remove_poor_atts = bool(algo_params['remove_poor_atts']),
                      no_preprune = bool(algo_params['no_preprune']),
                      leaf_prediction = str(algo_params['leaf_prediction']),
                      nb_threshold = int(algo_params['nb_threshold']))

    print("Algo params: " + json.dumps(algo_params))
   
    if evaluation=="holdout": 
        evaluator = EvaluateHoldout(show_plot = False,
                                    max_samples = int(eval_params['max_sample']),
                                    n_wait = int(eval_params['n_wait']),
                                    batch_size = int(eval_params['batch_size']),
                                    metrics = ['accuracy', 'kappa','kappa_t'],
                                    output_file = resultFile)
        evaluator.evaluate(stream=stream, model=ht)
    
   

    elif evaluation=="prequential":
        evaluator = EvaluatePrequential(show_plot = False,
                                        pretrain_size = int(eval_params['pretrain_size']),
                                        max_samples = int(eval_params['max_sample']),
                                        batch_size = int(eval_params['batch_size']),
                                        n_wait = int(eval_params['n_wait']),
                                        metrics = ['accuracy', 'kappa','kappa_t','kappa_m','true_vs_predicted'],
                                        output_file = resultFile)
    
        print("evaluate with pretrain size:" + str(eval_params['pretrain_size']) + 
                " max sample:" + str(eval_params['max_sample']) + 
                " batch size:" + str(eval_params['batch_size']) +
                "n_wait:" + str(eval_params['n_wait']))

        evaluator.evaluate(stream=stream, model=ht)

    else:
        print("Basic evaluation")
        # https://scikit-multiflow.readthedocs.io/en/stable/api/generated/skmultiflow.trees.HoeffdingTreeClassifier.html?highlight=hoeffding
        n_samples = 0
        correct_cnt = 0
       
        max_samples =  300
        if('start_value' in dataset_params): 
            if('stop_value' in dataset_params): 
                max_samples =  int(dataset_params['stop_value']) - int(dataset_params['start_value'])
        
        # Train the estimator with the samples provided by the data stream
        while n_samples < max_samples and stream.has_more_samples():
            time.sleep(0.1)
            X, y = stream.next_sample()
            y_pred = ht.predict(X)
            if y[0] == y_pred[0]:
                correct_cnt += 1
            ht = ht.partial_fit(X, y)
            try:
                print('{} samples analyzed.'.format(n_samples))
                accuracy =  round((correct_cnt / n_samples),2)
                print('Hoeffding Tree accuracy: {}'.format(correct_cnt / n_samples))
                progress = {}
                progress["n_samples"] = n_samples
                progress["correct_cnt"] = correct_cnt
                progress["accuracy"] = accuracy
                append_progress(jobid, progress)
            except ZeroDivisionError:
                print("0")
           
            n_samples += 1
       
        return progress

    
# eren: explain how do we handle the D3 algorithm
def run_d3(dataset_name,algo_params, jobid):
    print("Running D3 algorithm with dataset " + dataset_name)
    print("Algorithm parameters: " + str(algo_params))
    results = []
    drifts = []
    drifted_items = {}

    try:
        process = subprocess.Popen(['python', "ai_core/D3.py", dataset_name, str(algo_params['w']), str(algo_params['rho']), str(algo_params['auc'])], stdout=PIPE)
        driftPattern = re.compile("<DRIFT_START>(.*)<DRIFT_END>")
        accuracyPattern = re.compile("<ACCURACY_START>(.*),(.*),(.*)<ACCURACY_END>")

        condition = True
        while condition:
            print("D3 process is polled")
          
            for line in process.stdout:
                drift_search_results = driftPattern.search(line.decode('utf-8'))
                accuracy_search_results = accuracyPattern.search(line.decode('utf-8'))

                if drift_search_results:
                    d3_drifts_json = drift_search_results.group(1) 
                    drifts.append(d3_drifts_json)
                    
                elif accuracy_search_results:
                    accuracy = json.loads(accuracy_search_results.group(1))
                    data_length = json.loads(accuracy_search_results.group(2))
                    index = json.loads(accuracy_search_results.group(3))
                          
                    if len(accuracy)>0:
                        item={}
                        acc = accuracy[-1]
                        item["accuracy"] = round(float(acc),4)
                        item["percentage"] = round(int(index)/int(data_length),1)
                        append_progress(jobid, item)
                        
                        results.clear()
                        for acc in accuracy:
                            item={}
                            item["accuracy"] = round(float(acc),4)
                            item["percentage"] = round(int(index)/int(data_length),1)
                            results.append(item)
            condition = process.poll() is None
            if condition is False:
                break

        
        drifted_items["drifted_items"] = drifts
        results.append(drifted_items)
        print("Finished running D3")
                
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

    #print("D3 algorithm results obtained: " + str(results))
    
    return results  


# CluStream Algorithm 
def run_clustream(dataset_name,algo_params,jobid):
    print("Running CluStream algorithm with dataset " + dataset_name)
    print("Algorithm parameters: " + str(algo_params))

    xfname="xfname.txt"
    lfname="lfname.txt"
    all_data = pd.read_csv(dataset_name)

    x = all_data.iloc[:,:-1]
    x.to_csv(xfname, index=False, header=None)

    labels = all_data[all_data.columns[-1]]
    labels.to_csv(lfname, index=False, header=None)

    results = []
    
    try:
        process = subprocess.Popen(['Rscript', "ai_core/run-CluStream.r", xfname, lfname, str(algo_params['class']), str(algo_params['horizon']), str(algo_params['m']), str(algo_params['part_size']),], stdout=PIPE)
        # "<ACCURACY_START>",si, ":", si+part_size-1, "datalength:", data_length, "acc", ari, "meanacc",mean(na.omit(aris)), "time", total_time,"<ACCURACY_END>\n"
        accuracy_pattern = re.compile("<ACCURACY_START> (.*) : (.*) datalength: (.*) acc (.*) meanacc (.*) time (.*) <ACCURACY_END>")
            
        condition = True
        while condition:
            print("Clustream process is polled")

            for line in process.stdout: 
                search_results = accuracy_pattern.search(line.decode("utf-8"))
                if search_results:
                    item={}
                    item["start_index"] = search_results.group(1)
                    item["stop_index"] = search_results.group(2)
                    datalength = int(search_results.group(3))
                    item["percentage"] = round(int(search_results.group(2))/int(datalength),2)
                    item["accuracy"] = round(float(search_results.group(4)),4)
                    item["mean_accuracy"] = round(float(search_results.group(5)),4)
                    item["time"] = search_results.group(6)
                    # print("received and parsed item: ", item)                 
                    append_progress(jobid, item)
                    results.append(item)
            
            condition = process.poll() is None
            if condition is False:
                break
    
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    
    print("Finished running CluStream")
    # print("CluStream algorithm results obtained: " + str(results))
    
    return results  



# DenStream Algorithm
def run_denstream(dataset_name,algo_params, jobid):
    print("Running DenStream algorithm with dataset " + dataset_name)
    print("Algorithm parameters: " + str(algo_params))

    xfname="xfname.txt"
    lfname="lfname.txt"
    all_data = pd.read_csv(dataset_name)

    x = all_data.iloc[:,:-1]
    x.to_csv(xfname, index=False, header=None)

    labels = all_data[all_data.columns[-1]]
    labels.to_csv(lfname, index=False, header=None)
    results = []
    
    try:
        process = subprocess.Popen(['Rscript', "ai_core/run-DenStream.r", xfname, lfname, str(algo_params['class']), str(algo_params['epsilon']), str(algo_params['part_size'])], stdout=PIPE)
         # "<ACCURACY_START>",si, ":", si+part_size-1, "datalength:", data_length, "acc", ari, "meanacc",mean(na.omit(aris)), "time", total_time,"<ACCURACY_END>\n"
        accuracy_pattern = re.compile("<ACCURACY_START> (.*) : (.*) datalength: (.*) acc (.*) meanacc (.*) time (.*) <ACCURACY_END>")
            
        condition = True
        while condition:
            print("Denstream process is polled")

            for line in process.stdout: 
                search_results = accuracy_pattern.search(line.decode("utf-8"))
                if search_results:
                    item={}
                    item["start_index"] = search_results.group(1)
                    item["stop_index"] = search_results.group(2)
                    datalength = int(search_results.group(3))
                    item["percentage"] = round(int(search_results.group(2))/int(datalength),2)
                    item["accuracy"] = round(float(search_results.group(4)),4)
                    item["mean_accuracy"] = round(float(search_results.group(5)),4)
                    item["time"] = search_results.group(6)
                    # print("received and parsed item: ", item)                 
                    append_progress(jobid, item)
                    results.append(item)

            condition = process.poll() is None
            if condition is False:
                break
    
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    
    print("Finished running DenStream")
    #print("DenStream algorithm results obtained: " + str(results))
    return results  

# StreamKM++ Algorithm
def run_streamkm(dataset_name,algo_params, jobid):
    print("Running Stream KM++ algorithm with dataset " + dataset_name)
    print("Algorithm parameters: " + str(algo_params))

    xfname="xfname.txt"
    lfname="lfname.txt"
    all_data = pd.read_csv(dataset_name)

    x = all_data.iloc[:,:-1]
    x.to_csv(xfname, index=False, header=None)

    labels = all_data[all_data.columns[-1]]
    labels.to_csv(lfname, index=False, header=None)
    results = []
    
    try:
        process = subprocess.Popen(['Rscript', "ai_core/run-StreamKm.r", xfname, lfname, str(algo_params['part_size']), str(algo_params['n_cluster']), str(algo_params['size_coreset'])], stdout=PIPE)
         # "<ACCURACY_START>",si, ":", si+part_size-1, "datalength:", data_length, "acc", ari, "meanacc",mean(na.omit(aris)), "time", total_time,"<ACCURACY_END>\n"
        accuracy_pattern = re.compile("<ACCURACY_START> (.*) : (.*) datalength: (.*) acc (.*) meanacc (.*) time (.*) <ACCURACY_END>")
            
        condition = True
        while condition:
            print("StreamKM++ process is polled")

            for line in process.stdout: 
                search_results = accuracy_pattern.search(line.decode("utf-8"))
                if search_results:
                    item={}
                    item["start_index"] = search_results.group(1)
                    item["stop_index"] = search_results.group(2)
                    datalength = int(search_results.group(3))
                    item["percentage"] = round(int(search_results.group(2))/int(datalength),2)
                    item["accuracy"] = round(float(search_results.group(4)),4)
                    item["mean_accuracy"] = round(float(search_results.group(5)),4)
                    item["time"] = search_results.group(6)             
                    append_progress(jobid, item)
                    results.append(item)

            condition = process.poll() is None
            if condition is False:
                break
    
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    
    print("Finished running StreamKM++")
    #print("StreamKM++ algorithm results obtained: " + str(results))
    return results  

# ------------------- END OF ALGORITHMS



# ------------------- UTILS

def readAndParseResults(file):
    # read and filter the comment lines
    reader = csv.DictReader(filter(lambda row: row[0]!='#',file))
    
    # skip header row
    next(reader)
    
    # Parse the CSV into JSON  
    return json.dumps( [ row for row in reader ] )  

def getFile(id):
    return "result-"+str(id)+".csv"

# to append results on run time
def append_progress(jobid, progress):
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    cur = conn.cursor()

    cur.execute("SELECT progress FROM api_job WHERE id=%s",[jobid])
    result = cur.fetchone()[0]
    if not result.get("progress"):
        result["progress"] = []

    result["progress"].append(progress)

    cur.execute("UPDATE api_job SET progress=%s WHERE id=%s",[json.dumps(result), jobid])
    conn.commit()

    cur.close()
    conn.close()