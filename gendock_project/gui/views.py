# csv_processor/views.py
from django.shortcuts import render, redirect
from django.views import View
from .models import UploadedCSV
import os
from .tasks import process_csv_task


class ProcessCSVView(View):
    def post(self, request, pk):
        uploaded_csv = UploadedCSV.objects.get(pk=pk)
        csv_path = uploaded_csv.csv_file.path
        task = process_csv_task.delay(csv_path,pk)
        uploaded_csv.task_id = task.task_id
        uploaded_csv.save()
        return render(request, 'process_csv.html', context={'task_id': task.task_id})


class UploadCSVView(View):
    def get(self, request):
        return render(request, 'upload_csv.html')
    
    def post(self, request):
        csv_file = request.FILES.get('csv_file')
        if csv_file and csv_file.name.endswith('.csv'):
            UploadedCSV.objects.create(csv_file=csv_file)
            return redirect('csv_list')
        else:
            return render(request, 'upload_csv.html', {'error_message': 'Please upload a valid CSV file.'})

    
class CSVListView(View):
    def get(self, request):
        uploaded_csv_list = UploadedCSV.objects.all()
        return render(request, 'csv_list.html', {'uploaded_csv_list': uploaded_csv_list})
    
    def post(self, request):
        csv_id_to_process = request.POST.get('process_csv')
        csv_id_to_delete = request.POST.get('delete_csv')

        if csv_id_to_process:
            return redirect('process_csv', pk=csv_id_to_process)
        elif csv_id_to_delete:
            uploaded_csv = UploadedCSV.objects.get(pk=csv_id_to_delete)

            # Delete the associated cleaned SMILES file
            cleaned_smiles_file = uploaded_csv.cleaned_smiles_file
            print(cleaned_smiles_file)
            if cleaned_smiles_file:
                os.remove(cleaned_smiles_file)

            # Delete the UploadedCSV object
            uploaded_csv.delete()
        
        return redirect('csv_list')
    

