from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from .models import Job


# Create your views here.
def index(request):
    return HttpResponse("Hello, hello. Tell me what you want right now")

@csrf_exempt
def jobs_list(request):
    MAX_OBJECTS = 20
    jobs = Job.objects.all()[:MAX_OBJECTS]
    data = {"jobs": list(jobs.values("id","dataset_name","algorithm_name"
    ,"state","created_at","started_at","updated_at","finished_at"
    ,"dataset_params", "algorithm_params","results"))}
    return JsonResponse(data)

@csrf_exempt
def get_job(request, id):
    job = get_object_or_404(Job, pk=id)
    data = {"job": {
        "id": job.id,
        "dataset_name": job.dataset_name,
        "algorithm_name": job.algorithm_name,
        "state":job.state,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "updated_at": job.updated_at,
        "finished_at": job.finished_at,
        "dataset_params": job.dataset_params, 
        "algorithm_params": job.algorithm_params,
        "results": job.results
    }}
    return JsonResponse(data)

@csrf_exempt
def new_job(request):
    job = Job()
    job.dataset_name = 'kdd99'
    job.algorithm_name= 'hoeffding-tree'
    job.dataset_params = {}
    job.algorithm_params = {}
    job.results ={}
    job.save()
    
    return HttpResponse("new job")