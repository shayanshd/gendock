# csv_processor/views.py
from django.shortcuts import render, redirect
from django.views import View
from .models import UploadedCSV, CleanedSmile, TrainLog, ReceptorConfiguration
from .tasks import process_csv_task, start_training, generate_smiles, process_nd_worker, generate_more_smiles
from celery.result import AsyncResult
from celery_progress.backend import Progress
from django.http import HttpResponse, HttpResponseBadRequest, JsonResponse
from celery.app import default_app
from .forms import GenerateSmilesForm, ReceptorConfModelForm
import json
import ast
import csv

class DockingMasterTableView(View):
    def get(self, request, generation_number):
        with open('./rest/generations/master_results_table_gen' + str(generation_number) + '.csv', 'r') as csvfile:
            csvreader = csv.DictReader(csvfile)
            csv_data = list(csvreader)
        return render(request, 'docking_master_table.html', context={'csv_data': csv_data} )

class DockingProgressView(View):
    def get(self, request, dock_task_id, generation_number):
        result = AsyncResult(dock_task_id)
        progress = Progress(result) 
        percent_complete = int(progress.get_info()['progress']['percent'])      
        current_smile = int(progress.get_info()['progress']['current'])
        total_smile = int(progress.get_info()['progress']['total'])
        print(percent_complete, current_smile, total_smile)
        context={'dock_task_id':dock_task_id, 'dock_progress':percent_complete, 'dock_current':current_smile,
                 'dock_total':total_smile, 'generation_number':generation_number}
        if result.successful():  
            with open('./rest/generations/results/results_gen' + str(generation_number) + '.csv', 'r') as csvfile:
                csvreader = csv.DictReader(csvfile)
                csv_data = list(csvreader)
            context = {'dock_task_id': dock_task_id, 'dock_progress':percent_complete,
                       'docking_results': csv_data, 'generation_number':generation_number}
        
        return render(request, 'docking_progress.html', context=context)

class DockingView(View):
    def post(self, request):
        # Retrieve the value of hidden_generation_number from POST data
        generation_number = int(request.POST.get('hidden_generation_number'))      
        task = process_nd_worker.delay(generation_number)
        return render(request, 'docking_progress.html', context={'dock_task_id':task.id, 'dock_progress':0, 'generation_number':generation_number})
    
    def get(self, request):
        # Handle GET requests if needed
        return HttpResponseBadRequest("Invalid request method")

class GenerateProgressView(View):
    def get(self, request, task_id):
        
        result = AsyncResult(task_id)
        progress = Progress(result) 
        percent_complete = int(progress.get_info()['progress']['percent'])
        current_gen = int(progress.get_info()['progress']['current'])
        total_gen = int(progress.get_info()['progress']['total'])
        context={'task_id':task_id, 'progress':percent_complete, 'current':current_gen, 'total':total_gen}
        if result.successful():  
            form = ReceptorConfModelForm()         
            [validity, uniqueness, originality] = result.result
            context = {'task_id': task_id, 'validity': validity,
                        'uniqueness': uniqueness, 'originality': originality, 'form':form}
        
        return render(request, 'generate_progress.html', context=context)
    def post(self, request, task_id):
        
        form = ReceptorConfModelForm(request.POST, request.FILES)
        # print(request.POST, request.FILES)
        
        if form.is_valid():
            existing_instance = ReceptorConfiguration.objects.filter(receptor_file__icontains=form.cleaned_data['receptor_file'].name).first()

            if existing_instance:
                # Delete the previous instance
                existing_instance.delete()
            form.save()
            latest_config = ReceptorConfiguration.objects.latest('id')

            # Define the file path for the receptor file
            file_path = latest_config.receptor_file.path

            # Open and write the configuration to the file
            with open('rest/receptor_conf.txt', 'w') as config_file:
                config_file.write(f'receptor = {file_path}\n')
                config_file.write(f'center_x = {latest_config.center_x}\n')
                config_file.write(f'size_x = {latest_config.size_x}\n')
                config_file.write(f'center_y = {latest_config.center_y}\n')
                config_file.write(f'size_y = {latest_config.size_y}\n')
                config_file.write(f'center_z = {latest_config.center_z}\n')
                config_file.write(f'size_z = {latest_config.size_z}\n')
                config_file.write(f'exhaustiveness = {latest_config.exhaustive_number}\n')
            return HttpResponse('<p class="text-green-500 mt-2">Receptor Configuration Updated.</p>')
        else:
            print(form.errors)
            return HttpResponse('Something went wrong')
        # return render(request, 'generate_rec_submit.html', context={'form':form})

class GenerateSmilesView(View):
    
    def get(self, request):
        form = GenerateSmilesForm()
        return render(request, 'generate_smiles.html', {'form': form})
    
    def post(self, request):
        form = GenerateSmilesForm(request.POST)
        generation_number = int(request.POST.get('generation_number'))
        if form.is_valid():
            sample_number = form.cleaned_data['sample_number']
            desired_length = form.cleaned_data['desired_length']
            active_tasks = default_app.control.inspect().active()
            if not any(active_tasks.values()):
            # Enqueue the Celery task with the provided arguments
                if generation_number == 0:
                    task = generate_smiles.delay(sample_number, desired_length)
                else:
                    task = generate_more_smiles.delay(generation_number ,sample_number, desired_length)
                return render(request, 'generate_progress.html', context={'task_id':task.task_id,'progress':0, 'current':0, 'total':0})  # Redirect to a success page
            return HttpResponse('Another task is already in progress.')
        print(form.errors.as_json())
        return HttpResponse(f'<div class="text-red-600 mt-4">{list(form.errors.values())[0]}</div>')
        # return render(request, context={'form':form})

class TrainProgressView(View):

    def get(self, request, task_id):
        tl = TrainLog.objects.get(task_id = task_id)
        if tl.cur_batch > tl.max_batch:
            tl.max_batch = tl.cur_batch
            tl.save()

        percent_complete = int(100*tl.epoch/tl.max_epoch)
        batch_percent_complete = int(100*tl.cur_batch/tl.max_batch)
        if tl.task_status == 'C':
            context = {'task_id':task_id,'progress':tl, 'value': percent_complete, 
                       'batch_value':batch_percent_complete, 'loss':zip(list(range(1,len(ast.literal_eval(tl.train_loss))+1)),ast.literal_eval(tl.train_loss),ast.literal_eval(tl.val_loss))}
            
            return render(request, 'train_progress.html',context=context)        
        
        # context = {'task_id':task_id,'epoch':epoch,'epochs':epochs, 'val_loss':val_loss, 'train_loss':train_loss, 'value': percent_complete}
        context = {'task_id':task_id,'progress':tl, 'value': percent_complete, 'batch_value':batch_percent_complete}
        return render(request, 'train_progress.html',context=context)
    
    def post(self, request, task_id):
        tl = TrainLog.objects.get(task_id = task_id)
        tl.task_status = 'F'
        tl.save()
        # default_app.control.revoke(task_id, terminate=True, signal='SIGKILL')
        context = {'task_id':task_id,'progress':tl}
        return render(request, 'train_progress.html',context=context)

class TrainView(View):
    def get(self, request):  
        
        cleaned = CleanedSmile.objects.filter(task_status__in=['C'])
        return render(request, 'train.html', {'cleaned': cleaned})
    
    def post(self, request):
        cleaned_file = request.POST.get('cleaned_file')
        epochs = request.POST.get('epochs')
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
            # request.session['task_id'] = task_id
            # request.session['last_position'] = 0
            print('suc')
            return render(request, 'train_progress.html', context={'task_id': task_id, 'value' : 0})
        print('nisuc')    
        return HttpResponse('Another task is already in progress.')

class GetProgress(View):
    def get(self, request, task_id):
        progress = Progress(AsyncResult(task_id)) 
        percent_complete = int(progress.get_info()['progress']['percent'])
            
        if percent_complete == 100:
            cleaned_smi = CleanedSmile.objects.get(task_id=task_id)           
            return HttpResponse(f"<p class='mb-4'>CSV processing complete. The cleaned smiles file is: <em>{cleaned_smi.cleaned_file}</em> </p>")
        
        context = {'task_id':task_id, 'value': percent_complete}
        return render(request, 'process_csv.html',context=context)
    
    def post(self, request, task_id):
        default_app.control.revoke(task_id, terminate=True, signal='SIGKILL')
        return HttpResponse('Stopped')

class ProcessCSVView(View):
    def post(self, request):      
        pk_list = request.POST.getlist('selected_csvs')
        if len(pk_list) != 0:
            active_tasks = default_app.control.inspect().active()
            if not any(active_tasks.values()):
            # There are no active tasks, so we can start a new one
                task = process_csv_task.delay(pk_list)
                return render(request, 'process_csv.html', context={'task_id': task.task_id, 'value' : 0})    
            return HttpResponse('Another task is already in progress.')
        return HttpResponse('Please select a file to proceed')
           
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
