from skmultiflow.data import FileStream
from skmultiflow.trees import HoeffdingTree
from skmultiflow.lazy.sam_knn import SAMKNN
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.evaluation import EvaluateHoldout
from sklearn.cluster import KMeans
import psycopg2
import os
import time
import csv
import json
import subprocess
from subprocess import PIPE
import pandas as pd
from DenStream import DenStream

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
        cur.execute("SELECT id, dataset_name, algorithm_name, evaluation, state, created_at, started_at, updated_at, finished_at, dataset_params, algorithm_params  FROM api_job WHERE state='queued' ORDER BY created_at LIMIT 1")
        result = cur.fetchone()

        # if so, update the task to in progres...
        if result :
            id = result[0]
            dataset_name  = result [1]
            dataset_params = result[9]
            algorithm_name = result[2]
            algorithm_params= result[10]          
            evaluation= result[3]
        
            cur.execute("UPDATE api_job SET state='in_progress' WHERE id=%s",[id])
            conn.commit()

            # run the evaluation with the given dataset, params, algo, etc.
            results = runAlgorithm(dataset_name, algorithm_name, dataset_params, algorithm_params,evaluation)

            # update  json results
            cur.execute("UPDATE api_job SET results=(%s) WHERE id=%s",([results,id]))

            # update the results on the way, mark the task finished after done
            cur.execute("UPDATE api_job SET state='finished' WHERE id=%s",[id])
            conn.commit()
        else:
            print("Nothing to do...")

        cur.close()
        conn.close()

        # sleep 
        time.sleep(10)

    exit(0)


def runAlgorithm(dataset, algo_name, dataset_params, algo_params, evaluation):
    if(dataset_params['start_value'] and dataset_params['stop_value']):
        data = pd.read_csv(dataset+".csv")

        sub_data = data[pd.to_numeric(dataset_params['start_value']):pd.to_numeric(dataset_params['stop_value'])]
        print(sub_data)
        sub_data.to_csv("subdata.csv")
        used_dataset="subdata.csv"

    else: 
        used_dataset=dataset+".csv"

    stream = FileStream(used_dataset)
    stream.prepare_for_use()

    if algo_name=="hoeffding_tree":
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

    elif algo_name =="k_means":
        kmeans_result = run_kmeans(used_dataset,algo_params)
        out= pd.Series(kmeans_result).to_json(orient='values')
    
    elif algo_name == "d3":
        d3_result = run_d3(used_dataset,algo_params)
        out = json.dumps( d3_result)
        # if evaluation=="holdout": 
        # else: 

    elif algo_name=="denstream":
        print("run denstream run")
        denstream = run_denstream(used_dataset,algo_params)
        out = json.dumps(denstream) # TODO: duzelt

    elif algo_name=="clustream":
        print("insert clustream ")

    else:
        print("algorithm not found")

    return out


def run_d3(dataset_name,algo_params):
    process = subprocess.run(['python', "ai_core/D3.py", dataset_name,algo_params['w'], algo_params['rho'], algo_params['auc']], check=True, stdout=PIPE)
    results = {}
    results["output"] = process.stdout.decode("utf-8")
    return results

def run_denstream(dataset_name,algo_params):
    data = pd.read_csv(dataset_name)
    print( data.values)
    denstream = DenStream(eps=pd.to_numeric(algo_params['epsilon']), lambd=pd.to_numeric(algo_params['lambda']), beta=pd.to_numeric(algo_params['beta']), mu=pd.to_numeric(algo_params['mu'])).fit_predict(data.values)
    return denstream


def run_hoeffdingtree(stream,algo_params, evaluation):
    ht = HoeffdingTree()
    if evaluation=="holdout": 
        evaluator = EvaluateHoldout(show_plot=False,
                                    pretrain_size=pd.to_numeric(algo_params['pretrain_size']),
                                    max_samples=pd.to_numeric(algo_params['max_sample']),
                                    metrics=['accuracy', 'kappa','kappa_t'],
                                    batch_size = pd.to_numeric(algo_params['batch_size']),
                                    restart_stream=algo_params['restart_stream'],
                                    output_file='result.csv')
    else:
        evaluator = EvaluatePrequential(show_plot=False,
                                        max_samples=pd.to_numeric(algo_params['max_sample']),
                                        metrics=['accuracy', 'kappa','kappa_t','kappa_m','true_vs_predicted'],
                                        output_file='result.csv')

    evaluator.evaluate(stream=stream, model=ht)

def run_knn(stream,algo_params, evaluation):
    knn = SAMKNN(n_neighbors=pd.to_numeric(algo_params['neighbors']), weighting='distance', 
          max_window_size=pd.to_numeric(algo_params['max_window_size']), stm_size_option='maxACCApprox',use_ltm=False)
     
    if evaluation=="holdout":
        evaluator = EvaluateHoldout(pretrain_size=pd.to_numeric(algo_params['pretrain_size']), 
                                        max_samples=pd.to_numeric(algo_params['max_sample']), 
                                        batch_size=pd.to_numeric(algo_params['batch_size']), 
                                        n_wait=pd.to_numeric(algo_params['n_wait']), 
                                        max_time=pd.to_numeric(algo_params['max_time']), 
                                        output_file='knn.csv', 
                                        metrics=['accuracy', 'kappa_t'])
    else:      
        evaluator = EvaluatePrequential(pretrain_size=pd.to_numeric(algo_params['pretrain_size']), 
                                        max_samples=pd.to_numeric(algo_params['max_sample']), 
                                        batch_size=pd.to_numeric(algo_params['batch_size']), 
                                        n_wait=pd.to_numeric(algo_params['n_wait']), 
                                        max_time=pd.to_numeric(algo_params['max_time']), 
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