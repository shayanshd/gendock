from django.contrib import admin
from . import models
# Register your models here.
admin.site.register(models.UploadedCSV)
admin.site.register(models.CleanedSmile)
admin.site.register(models.TrainLog)