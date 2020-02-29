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
        run_hoeffdingtree(stream, evaluation)
        with open("result.csv") as file: 
            # read and filter the comment lines
            reader = csv.DictReader(filter(lambda row: row[0]!='#',file))
            # skip header row
            next(reader)
            # Parse the CSV into JSON  
            out = json.dumps( [ row for row in reader ] )  
            
    elif algo_name =="knn":
        knn_result = run_knn(stream, evaluation)
        with open("knn.csv") as file: 
            # read and filter the comment lines
            reader = csv.DictReader(filter(lambda row: row[0]!='#',file))
            # skip header row
            next(reader)
            # Parse the CSV into JSON  
            out = json.dumps( [ row for row in reader ] )  

    elif algo_name =="k_means":
        kmeans_result = run_kmeans(used_dataset)

    
    elif algo_name == "d3":
        d3_result = run_d3(used_dataset)
        out = json.dumps( d3_result)
        # if evaluation=="holdout": 
        # else: 

    elif algo_name=="denstream":
        print("run denstream run")
        denstream = run_denstream(used_dataset)
        out = json.dumps(denstream)

    elif algo_name=="clustream":
        print("insert clustream ")

    else:
        print("algorithm not found")

    return out


def run_d3(dataset_name):
    process = subprocess.run(['python', "ai_core/D3.py", dataset_name, "100", "0.1", "0.7"], check=True, stdout=PIPE)
    results = {}
    results["output"] = process.stdout.decode("utf-8")
    return results

def run_denstream(dataset_name):
    data = pd.read_csv(dataset_name)
    print( data.values)
    print("end of data")
    #  denstream = DenStream(eps=0.3, lambd=0.1, beta=0.5, mu=11)
    denstream = DenStream(eps=0.3, lambd=0.1, beta=0.5, mu=11).fit_predict(data.values)
    return result


def run_hoeffdingtree(stream, evaluation):
    ht = HoeffdingTree()
    if evaluation=="holdout": 
        evaluator = EvaluateHoldout(show_plot=False,
                                    max_samples=400,
                                    metrics=['accuracy', 'kappa','kappa_t'],
                                    restart_stream=False,
                                    output_file='result.csv')
    else:
        evaluator = EvaluatePrequential(show_plot=False,
                                        pretrain_size=200,
                                        max_samples=300,
                                        metrics=['accuracy', 'kappa','kappa_t','kappa_m','true_vs_predicted'],
                                        output_file='result.csv')

    evaluator.evaluate(stream=stream, model=ht)

def run_knn(stream, evaluation):
    knn = SAMKNN(n_neighbors=5, weighting='distance', max_window_size=1000, stm_size_option='maxACCApprox',use_ltm=False)
     
    if evaluation=="holdout":
        evaluator = EvaluateHoldout(pretrain_size=0, 
                                        max_samples=200, 
                                        batch_size=1, 
                                        n_wait=100, 
                                        max_time=100, 
                                        output_file='knn.csv', 
                                        metrics=['accuracy', 'kappa_t'])
    else:      
        evaluator = EvaluatePrequential(pretrain_size=0, 
                                        max_samples=200, 
                                        batch_size=1, 
                                        n_wait=100, 
                                        max_time=100, 
                                        output_file='knn.csv',  
                                        metrics=['accuracy', 'kappa_t'])

    evaluator.evaluate(stream=stream, model=knn)

if __name__ == "__main__":
    main()