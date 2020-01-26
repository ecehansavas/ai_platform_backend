from skmultiflow.data import FileStream
from skmultiflow.trees import HoeffdingTree
from skmultiflow.evaluation import EvaluatePrequential
import psycopg2
import os
import time
import csv
import json


def main():
    # while true
    while True:

        # connect to the database
        conn = psycopg2.connect(os.environ['DATABASE_URL'])
        cur = conn.cursor()

        # send a SQL query to see if there's a task in queue
        cur.execute("SELECT * FROM api_job WHERE state='queued' ORDER BY created_at LIMIT 1")
        result = cur.fetchone()
        # if so, update the task to in progres...
        if result :
            id = result[0]
        
            dataset_name  = result [1]
            dataset_params = result[8]
            algorithm_name = result[2]
            algorithm_params= result[9]

            cur.execute("UPDATE api_job SET state='in_progress' WHERE id=%s",[id])
            conn.commit()

            # run the evaluation with the given dataset, params, algo, etc.
            results = runAlgorithm(dataset_name, algorithm_name, dataset_params, algorithm_params)
            print(results)
            # update  json results
            cur.execute("UPDATE api_job SET results=(%s) WHERE id=%s",([results,id]))

            # update the results on the way, mark the task finished after done
            cur.execute("UPDATE api_job SET state='finished' WHERE id=%s",[id])
            conn.commit()

        cur.close()
        conn.close()

        # sleep 
        time.sleep(10)

    exit(0)


def runAlgorithm(dataset, algo_name, dataset_params, algo_params):
    stream = FileStream(dataset+".csv")
    stream.prepare_for_use()

    if algo_name=="hoeffding_tree":
        ht = HoeffdingTree()

        evaluator = EvaluatePrequential(show_plot=False,
                                        pretrain_size=200,
                                        max_samples=300,
                                        metrics=['accuracy', 'kappa','kappa_t','kappa_m','true_vs_predicted'],
                                        output_file='result.csv')
        evaluator.evaluate(stream=stream, model=ht)

        with open("result.csv") as file: 
            # read and filter the comment lines
            reader = csv.DictReader(filter(lambda row: row[0]!='#',file))
            # skip header row
            next(reader)
            # Parse the CSV into JSON  
            out = json.dumps( [ row for row in reader ] )  
            
    elif algo_name =="knn":
        print("insert here knn")

    elif algo_name =="k-means":
        print("insert here kmeans")
    else:
        print("algorithm not found")

    return out


if __name__ == "__main__":
    main()