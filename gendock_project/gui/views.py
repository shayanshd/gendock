# csv_processor/views.py
from django.shortcuts import render, redirect
from django.views import View
from .models import UploadedCSV
import os
from .tasks import process_csv_task
from celery.result import AsyncResult
from celery_progress.backend import Progress
from django.http import HttpResponse
from celery.app import default_app

def train_view(request):
    csvs = UploadedCSV.objects.filter(cleaned_smiles_file__isnull=False).values()
    print('************************',csvs)

    
    return render(request, 'train.html', {'csvs': csvs})

class ProcessCSVView(View):
    def post(self, request):      
        pk_list = request.POST.getlist('selected_csvs')
        if len(pk_list) != 0:
            task = process_csv_task.delay(pk_list)
            return render(request, 'process_csv.html', context={'task_id': task.task_id, 'value' : 0})

        return HttpResponse('Please select a file to proceed')

class GetProgress(View):
    def get(self, request, task_id):
        progress = Progress(AsyncResult(task_id)) 
        percent_complete = int(progress.get_info()['progress']['percent'])
            
        if percent_complete == 100:
            uploaded_csv_list = UploadedCSV.objects.all()
            context = {'uploaded_csv_list': uploaded_csv_list}
            return render(request, 'csv_list.html', context=context)
        
        context = {'task_id':task_id, 'value': percent_complete}
        return render(request, 'process_csv.html',context=context)
    
    def post(self, request, task_id):
        print(request.POST)
        default_app.control.revoke(task_id, terminate=True, signal='SIGKILL')
        return HttpResponse('Stopped')
            
class UploadCSVView(View):
    def get(self, request):
        uploaded_csv_list = UploadedCSV.objects.all()
        context = {'uploaded_csv_list': uploaded_csv_list}
        return render(request, 'upload_csv.html', context=context)
    
    def post(self, request):
        csv_id_to_delete = request.POST.getlist('selected_csvs')
        csv_file = request.FILES.get('csv_file')
        if csv_file:
            if csv_file.name.endswith('.csv'):
                UploadedCSV.objects.create(csv_file=csv_file)
                return redirect('upload')
            else:
                return render(request, 'upload_csv.html', {'error_message': 'Please upload a valid CSV file.'})

        if csv_id_to_delete:
            for pk in csv_id_to_delete:

                uploaded_csv = UploadedCSV.objects.get(pk=pk)

                # Delete the associated cleaned SMILES file
                cleaned_smiles_file = uploaded_csv.cleaned_smiles_file
                if cleaned_smiles_file:
                    if os.path.exists(cleaned_smiles_file):
                        os.remove(cleaned_smiles_file)

                # Delete the UploadedCSV object
                uploaded_csv.delete()
            uploaded_csv_list = UploadedCSV.objects.all()
            context = {'uploaded_csv_list': uploaded_csv_list}
            return render(request,'csv_list.html', context=context)
        return redirect('upload')
    
