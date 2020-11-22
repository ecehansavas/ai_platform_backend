import psycopg2
import os
import sys
import os.path
import traceback
import time
import csv
import json
import re
from AIJob import AIJob
from datetime import datetime
 

def main():
    # connect to the database
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
  
    while True:
        cur = conn.cursor()

        # eren: you can create a seperate file for all DB access operations and move all DB actions there. That would clean up the code a lot
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

            print("Starting processing job with id: " + str(id))

            cur.execute("UPDATE api_job SET state='in_progress', started_at=(%s) WHERE id=%s",[datetime.now(), id])
            conn.commit()

            try:
                # run the evaluation with the given dataset, params, algo, etc.
                ai_job = AIJob(id, dataset_name, algorithm_name, dataset_params, algorithm_params, evaluation, eval_params)
                results, data_summary = ai_job.runTheJob()
        
                # update json results
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
        
        # sleep 
        time.sleep(10)

    conn.close()
    exit(0)


def getFile(id):
    return "result-"+str(id)+".csv"


if __name__ == "__main__":
    main()