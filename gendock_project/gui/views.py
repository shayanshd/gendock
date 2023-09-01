# csv_processor/views.py
from django.shortcuts import render, redirect
from django.views import View
from .models import UploadedCSV, CleanedSmile
from .tasks import process_csv_task, start_training
from celery.result import AsyncResult
from celery_progress.backend import Progress
from django.http import HttpResponse
from celery.app import default_app
import json


class TrainProgressView(View):
    def read_new_lines(self, filename, last_position):
        new_lines = []      
        with open(filename, 'r') as file:
            file.seek(last_position) 
            new_lines = file.readlines()             
            last_position = file.tell()       
        return new_lines, last_position

    
    def get(self, request):
        filename = './celery.logs'
        new_lines, last_position = self.read_new_lines(filename, request.session['last_position'])
        request.session['last_position'] = last_position
        result = default_app.AsyncResult(request.session['task_id'])
        if result.state == 'SUCCESS':
            return HttpResponse(status=286)
        return render(request, 'train_progress.html',context={'new_lines':new_lines})

    def post(self, request, *args, **kwargs):
        pass


class TrainView(View):
    def get(self, request):  
        
        cleaned = CleanedSmile.objects.filter(task_status__in=['C'])
        return render(request, 'train.html', {'cleaned': cleaned})
    
    def post(self, request):
        cleaned_file = request.POST.get('cleaned_file')
        epochs = request.POST.get('epochs')
        print(cleaned_file, epochs)
        if not cleaned_file or not epochs:
            return HttpResponse('<p class="text-red-600">Please enter a valid number of epochs.</p>')  # Redirect back to the train page

        # Update config.json file with epochs
        active_tasks = default_app.control.inspect().active()
        if not any(active_tasks.values()):
            with open('rest/experiments/LSTM_Chem/config.json', 'r') as config_file:
                config_data = json.load(config_file)

            config_data['num_epochs'] = int(epochs)
            config_data['data_filename'] = cleaned_file

            with open('rest/experiments/LSTM_Chem/config.json', 'w') as config_file:
                json.dump(config_data, config_file)
            result = start_training.delay()
            task_id = result.id
            request.session['task_id'] = task_id
            request.session['last_position'] = 0
            print('suc')
            return render(request, 'train_progress.html') 
        print('nisuc')    
        return HttpResponse('Another task is already in progress.')



class ProcessCSVView(View):
    def post(self, request):      
        pk_list = request.POST.getlist('selected_csvs')
        if len(pk_list) != 0:
            active_tasks = default_app.control.inspect().active()
            if not any(active_tasks.values()):
            # There are no active tasks, so we can start a new one
                print(active_tasks.values())
                task = process_csv_task.delay(pk_list)
                return render(request, 'process_csv.html', context={'task_id': task.task_id, 'value' : 0})    
            return HttpResponse('Another task is already in progress.')
        return HttpResponse('Please select a file to proceed')

class GetProgress(View):
    def get(self, request, task_id):
        progress = Progress(AsyncResult(task_id)) 
        percent_complete = int(progress.get_info()['progress']['percent'])
            
        if percent_complete == 100:
            cleaned_smi = CleanedSmile.objects.get(task_id=task_id)
            context = {'cleaned_smi': cleaned_smi}
            return render(request, 'process_csv_done.html', context=context)
        
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
                uploaded_csv.delete()

            uploaded_csv_list = UploadedCSV.objects.all()
            context = {'uploaded_csv_list': uploaded_csv_list}
            return render(request,'csv_list.html', context=context)
        return redirect('upload')
    
