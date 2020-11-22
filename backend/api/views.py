from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from .models import Job
import json

# eren: delete this comment :)
# Create your views here.
def index(request):
    return HttpResponse("Hello, hello. Tell me what you want right now")

@csrf_exempt
def jobs_list(request):
    MAX_OBJECTS = 50
    jobs = Job.objects.all()[:MAX_OBJECTS]
    data = {"jobs": list(jobs.values("id","dataset_name","algorithm_name","evaluation"
    ,"state","created_at","started_at","updated_at","finished_at"
    ,"dataset_params", "algorithm_params","evaluation_params","results", "data_summary", "progress"))}
    return JsonResponse(data)

@csrf_exempt
def get_job(request, id):
    job = get_object_or_404(Job, pk=id)
    data = {"job": {
        "id": job.id,
        "dataset_name": job.dataset_name,
        "algorithm_name": job.algorithm_name,
        "evaluation": job.evaluation,
        "state":job.state,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "updated_at": job.updated_at,
        "finished_at": job.finished_at,
        "dataset_params": job.dataset_params, 
        "algorithm_params": job.algorithm_params,
        "evaluation_params": job.evaluation_params,
        "results": job.results,
        "data_summary": job.data_summary,
        "progress": job.progress
    }}
    return JsonResponse(data)

@csrf_exempt
def new_job(request):
    print("I AM IN NEW JOB", flush=True)
    
    body_unicode = request.body.decode('utf-8')
    body = json.loads(body_unicode)
    print(body)
    job = Job()
    job.dataset_name = body['dataset_name']
    job.algorithm_name= body['algorithm_name']
    job.evaluation = body['selected_evaluation']
    job.dataset_params = body['dataset_parameters']
    job.algorithm_params = body['algorithm_parameters']
    job.evaluation_params = body['evaluation_parameters']
    job.results ={}
    job.data_summary = {}
    job.progress = {}
    job.save()
    
    return HttpResponse("new job") # eren: return a better response

@csrf_exempt
def delete_job(request,id):
    job = get_object_or_404(Job, pk=id)
    job.delete()
    return HttpResponse(status=202) 

# eren: move your dataset files to a proper directory