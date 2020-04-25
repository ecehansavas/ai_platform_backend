from django.db import models
from django.contrib.postgres.fields import JSONField

class Job(models.Model):
    STATES = [
        ('queued', 'In Queue'),
        ('in_progress', 'In Progress'),
        ('finished', 'Finished'),
        ('failed', 'Failed')
    ]
    dataset_name = models.CharField(max_length=50)
    algorithm_name = models.CharField(max_length=50)
    evaluation = models.CharField(max_length=50, default='prequential')
    state = models.CharField(max_length=50, default="queued", choices=STATES)
    created_at = models.DateTimeField('date created', auto_now_add=True)
    started_at = models.DateTimeField('date started', null=True)
    updated_at = models.DateTimeField('last update date', auto_now=True)
    finished_at = models.DateTimeField('date finished', null=True)
    dataset_params = JSONField()
    algorithm_params = JSONField()
    evaluation_params = JSONField(default=dict)
    results = JSONField()