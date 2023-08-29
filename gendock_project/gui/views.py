# csv_processor/views.py
from django.shortcuts import render, redirect
from django.views import View
from .models import UploadedCSV, CleanedSmile
import os
from .tasks import process_csv_task, start_training
from celery.result import AsyncResult
from celery_progress.backend import Progress
from django.http import HttpResponse, JsonResponse
from celery.app import default_app
import json


class TrainProgressView(View):
    def get(self, request):
        # Start the Celery task
        task = start_training.delay()
        task_id = task.id

        response_data = {
            'message': 'Training task started successfully.',
            'task_id': task_id
        }
        return JsonResponse(response_data)


class TrainView(View):
    def get(self, request):  
        cleaned = CleanedSmile.objects.filter(task_status__in=['C'])
        return render(request, 'train.html', {'cleaned': cleaned})
    
    def post(self, request):
        cleaned_file = request.POST.get('cleaned_file')
        epochs = request.POST.get('epochs')
        print(cleaned_file, epochs)
        if not cleaned_file or not epochs:
            return JsonResponse({'error': 'Both cleaned file and epochs are required.'}, status=400)

        # Update config.json file with epochs
        try:
            with open('rest/experiments/LSTM_Chem/config.json', 'r') as config_file:
                config_data = json.load(config_file)

            config_data['num_epochs'] = int(epochs)
            config_data['data_filename'] = cleaned_file

            with open('rest/experiments/LSTM_Chem/config.json', 'w') as config_file:
                json.dump(config_data, config_file)

            return JsonResponse({'message': 'Config updated successfully.'})
        except Exception as e:
            return JsonResponse({'error': 'An error occurred while updating config.'}, status=500)


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
            cleaned_smi = CleanedSmile.objects.get(task_id=task_id)
            context = {'cleaned_smi': cleaned_smi}
            return render(request, 'process_csv_done.html', context=context)
            # return redirect('train')
            # return render(request, 'csv_list.html', context=context)
        
        context = {'task_id':task_id, 'value': percent_complete}
        return render(request, 'process_csv.html',context=context)
    
    def post(self, request, task_id):
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
                # uploaded_csv.cleanedsmile_set.all().delete()
                # Delete the associated cleaned SMILES file
                # cleaned_smiles_file = uploaded_csv.cleaned_smiles_file
                # if cleaned_smiles_file:
                #     if os.path.exists(cleaned_smiles_file):
                #         os.remove(cleaned_smiles_file)
                # Delete the UploadedCSV object
                uploaded_csv.delete()

            uploaded_csv_list = UploadedCSV.objects.all()
            context = {'uploaded_csv_list': uploaded_csv_list}
            return render(request,'csv_list.html', context=context)
        return redirect('upload')
    
