# csv_processor/views.py
from django.shortcuts import render, redirect
from django.views import View
from .models import UploadedCSV
import os
from .tasks import process_csv_task
from celery.result import AsyncResult
from celery_progress.backend import Progress


class ProcessCSVView(View):
    def get(self, request, pk):
        uploaded_csv = UploadedCSV.objects.get(pk=pk)
        csv_path = uploaded_csv.csv_file.path
        task = process_csv_task.delay(csv_path,pk)
        uploaded_csv.task_id = task.task_id
        uploaded_csv.save()
        return render(request, 'process_csv.html', context={'task_id': task.task_id, 'value' : 0})

class GetProgress(View):
    def get(self, request, task_id):
        progress = Progress(AsyncResult(task_id)) 
        percent_complete = int(progress.get_info()['progress']['percent'])
        if percent_complete == 100:
            results = UploadedCSV.objects.get(task_id=task_id)
            print(results.cleaned_smiles_file)
            return render(request, 'csv_list.html', {'results': results})
        
        context = {'task_id':task_id, 'value': percent_complete}
        return render(request, 'process_csv.html',context=context)


class UploadCSVView(View):
    def get(self, request):
        uploaded_csv_list = UploadedCSV.objects.all()
        return render(request, 'upload_csv.html', {'uploaded_csv_list': uploaded_csv_list})
    
    def post(self, request):

        csv_id_to_delete = request.POST.get('delete_csv')
        csv_file = request.FILES.get('csv_file')
        if csv_file:
            if csv_file.name.endswith('.csv'):
                UploadedCSV.objects.create(csv_file=csv_file)
                return redirect('upload')
            else:
                return render(request, 'upload_csv.html', {'error_message': 'Please upload a valid CSV file.'})

        if csv_id_to_delete:
            uploaded_csv = UploadedCSV.objects.get(pk=csv_id_to_delete)

            # Delete the associated cleaned SMILES file
            cleaned_smiles_file = uploaded_csv.cleaned_smiles_file
            if cleaned_smiles_file:
                os.remove(cleaned_smiles_file)

            # Delete the UploadedCSV object
            uploaded_csv.delete()
        return redirect('upload')

        


