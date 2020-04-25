from skmultiflow.data import FileStream
from skmultiflow.data import SEAGenerator
from skmultiflow.data import HyperplaneGenerator
from skmultiflow.trees import HoeffdingTree
from skmultiflow.anomaly_detection import HalfSpaceTrees
from skmultiflow.lazy.sam_knn import SAMKNN
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.evaluation import EvaluateHoldout
from sklearn.cluster import KMeans
import numpy as np
import psycopg2
import os
import time
import csv
import json
import subprocess
from subprocess import PIPE
import pandas as pd
from DenStream import DenStream
from CluStream import CluStream
from datetime import datetime

# https://github.com/narjes23/Clustream-algorithm
# https://github.com/ogozuacik/d3-discriminative-drift-detector-concept-drift/ 
# https://github.com/issamemari/DenStream


def main():
    # while true
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

            # run the evaluation with the given dataset, params, algo, etc.
            results = prepareForRun(dataset_name, algorithm_name, dataset_params, algorithm_params,evaluation, eval_params)

            # update  json results
            cur.execute("UPDATE api_job SET results=(%s) WHERE id=%s",([results,id]))

            # update the results on the way, mark the task finished after done
            cur.execute("UPDATE api_job SET state='finished', finished_at=(%s) WHERE id=%s",[datetime.now(),id])
            conn.commit()
        else:
            print("Nothing to do...")

        cur.close()
        conn.close()

        # sleep 
        time.sleep(10)

    exit(0)


def prepareForRun(dataset, algo_name, dataset_params, algo_params, evaluation, eval_params):
    if('noise_percentage' in dataset_params): # generated 
        print('Generatordeyim')
        if('n_drift_features' in dataset_params): #hyperplane
            print("hyperplane generator")
            stream = HyperplaneGenerator(random_state=None, n_features=pd.to_numeric(eval_params['n_features']), 
                                n_drift_features=pd.to_numeric(eval_params['n_drift_features']), 
                                mag_change=pd.to_numeric(eval_params['mag_change']), 
                                noise_percentage=pd.to_numeric(eval_params['noise_percentage']), 
                                sigma_percentage=pd.to_numeric(eval_params['sigma_percentage']))
        else: #sea
            print("sea generator")
            stream = SEAGenerator(classification_function=0, random_state=None, balance_classes=False, 
                    noise_percentage=pd.to_numeric(eval_params['noise_percentage']))
    else: 
        used_dataset = prepareDataset(dataset, dataset_params)
        stream = FileStream(used_dataset)
    stream.prepare_for_use()

    if algo_name == "hoeffding_tree":
        run_hoeffdingtree(stream, algo_params, evaluation)
        with open("result.csv") as file: 
            # read and filter the comment lines
            reader = csv.DictReader(filter(lambda row: row[0]!='#',file))
            # skip header row
            next(reader)
            # Parse the CSV into JSON  
            out = json.dumps( [ row for row in reader ] )  
            
    elif algo_name =="knn":
        knn_result = run_knn(stream, algo_params, evaluation)
        with open("knn.csv") as file: 
            # read and filter the comment lines
            reader = csv.DictReader(filter(lambda row: row[0]!='#',file))
            # skip header row
            next(reader)
            # Parse the CSV into JSON  
            out = json.dumps( [ row for row in reader ] )  

    elif algo_name == "k_means":
        kmeans_result = run_kmeans(used_dataset,algo_params)
        out = pd.Series(kmeans_result).to_json(orient='values')
    
    elif algo_name == "d3":
        d3_result = run_d3(used_dataset,algo_params)
        out = json.dumps( d3_result)
        # if evaluation=="holdout": 
        # else: 

    elif algo_name == "denstream":
        print("run denstream run")
        denstream = run_denstream(used_dataset,algo_params)
        out = json.dumps(denstream) # TODO: duzelt

    elif algo_name == "clustream":
        print("insert clustream ")
        clustream = run_clustream(used_dataset, algo_params)
        out = json.dumps(clustream) # TODO: Duzelt

    #https://scikit-multiflow.github.io/scikit-multiflow/api/generated/skmultiflow.anomaly_detection.HalfSpaceTrees.html#skmultiflow.anomaly_detection.HalfSpaceTrees
    elif algo_name == "half_space_tree":
        run_halfspacetree(stream, algo_params, evaluation)
        with open("result.csv") as file: 
            # read and filter the comment lines
            reader = csv.DictReader(filter(lambda row: row[0]!='#',file))
            # skip header row
            next(reader)
            # Parse the CSV into JSON  
            out = json.dumps( [ row for row in reader ] )  
        
    else:
        print("algorithm not found")

    return out

#TODO: median mean falan ekle
#TODO: data histogrami ekle
def  prepareDataset(dataset, dataset_params):
    if('start_value' in dataset_params and 'stop_value'in dataset_params):
        data = pd.read_csv(dataset+".csv")

        sub_data = data[pd.to_numeric(dataset_params['start_value']):pd.to_numeric(dataset_params['stop_value'])]
        print(sub_data)
        sub_data.to_csv("subdata.csv")
        used_dataset="subdata.csv"
    else: 
        used_dataset=dataset+".csv"
    return used_dataset


def run_d3(dataset_name,algo_params):
    process = subprocess.run(['python', "ai_core/D3.py", dataset_name,algo_params['w'], algo_params['rho'], algo_params['auc']], check=True, stdout=PIPE)
    results = {}
    results["output"] = process.stdout.decode("utf-8")
    return results

#TODO: calismiyor
def run_denstream(dataset_name,algo_params):
    data = pd.read_csv(dataset_name)
    print( data.values)
    denstream = DenStream(eps=pd.to_numeric(algo_params['epsilon']), lambd=pd.to_numeric(algo_params['lambda']), beta=pd.to_numeric(algo_params['beta']), mu=pd.to_numeric(algo_params['mu'])).fit_predict(data.values)
    return denstream

#TODO: calismiyor
def run_clustream(dataset, algorithm_params):
    data = pd.read_csv(dataset)
    print(data.values)
    clustream = CluStream(nb_initial_points=1000, time_window=1000, timestamp=0, clocktime=0, nb_micro_cluster=100,
                nb_macro_cluster=5, micro_clusters=[], alpha=2, l=2, h=1000)
    return clustream

def run_hoeffdingtree(stream,algo_params, evaluation):
    ht = HoeffdingTree()
    if evaluation=="holdout": 
        evaluator = EvaluateHoldout(show_plot=False,
                                    max_samples=pd.to_numeric(algo_params['max_sample']),
                                    metrics=['accuracy', 'kappa','kappa_t'],
                                    batch_size = pd.to_numeric(algo_params['batch_size']),
                                    output_file='result.csv')
    else:
        evaluator = EvaluatePrequential(show_plot=False,
                                        metrics=['accuracy', 'kappa','kappa_t','kappa_m','true_vs_predicted'],
                                        output_file='result.csv')

    evaluator.evaluate(stream=stream, model=ht)

#TODO: bunun sonuclarini alamadik henuz.
def run_halfspacetree(stream, algo_params, evaluation):
    print("halfspace")
    hst = HalfSpaceTrees(n_features=pd.to_numeric(algo_params['n_features']), 
                        window_size=pd.to_numeric(algo_params['window_size']),
                        depth=pd.to_numeric(algo_params['depth']),
                        n_estimators=pd.to_numeric(algo_params['n_estimators']),
                        size_limit=pd.to_numeric(algo_params['size_limit']),
                        anomaly_threshold=pd.to_numeric(algo_params['anomaly_threshold']),
                        random_state=None)
    if evaluation=="holdout": 
        evaluator = EvaluateHoldout(show_plot=False,
                                    max_samples=pd.to_numeric(algo_params['max_sample']),
                                    metrics=['accuracy', 'kappa','kappa_t'],
                                    batch_size = pd.to_numeric(algo_params['batch_size']),
                                    output_file='result.csv')
    else:
        evaluator = EvaluatePrequential(show_plot=False,
                                        metrics=['accuracy', 'kappa','kappa_t','kappa_m','true_vs_predicted'],
                                        output_file='result.csv')
    print("bitti half space")



def run_knn(stream,algo_params, evaluation):
    knn = SAMKNN(n_neighbors=pd.to_numeric(algo_params['neighbors']), weighting='distance', 
          max_window_size=pd.to_numeric(algo_params['max_window_size']), stm_size_option='maxACCApprox',use_ltm=False)
     
    if evaluation=="holdout":
        evaluator = EvaluateHoldout( max_samples=pd.to_numeric(algo_params['max_sample']), 
                                        batch_size=pd.to_numeric(algo_params['batch_size']), 
                                        n_wait=pd.to_numeric(algo_params['n_wait']), 
                                        output_file='knn.csv', 
                                        metrics=['accuracy', 'kappa_t'])
    else:      
        evaluator = EvaluatePrequential(pretrain_size=pd.to_numeric(algo_params['pretrain_size']), 
                                        max_samples=pd.to_numeric(algo_params['max_sample']), 
                                        batch_size=pd.to_numeric(algo_params['batch_size']), 
                                        n_wait=pd.to_numeric(algo_params['n_wait']),  
                                        output_file='knn.csv',  
                                        metrics=['accuracy', 'kappa_t'])

    evaluator.evaluate(stream=stream, model=knn)

def run_kmeans(dataset_name, algo_params):
    data = pd.read_csv(dataset_name)
    kmeans = KMeans(n_clusters=pd.to_numeric(algo_params['n_cluster']), init='k-means++', 
    max_iter=pd.to_numeric(algo_params['max_iter']), n_init=pd.to_numeric(algo_params['n_init']),
             random_state=pd.to_numeric(algo_params['random_state']))
    return kmeans.fit_predict(data.values)





if __name__ == "__main__":
    main()