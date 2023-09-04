# csv_processor/urls.py
from django.urls import path
from .views import UploadCSVView, ProcessCSVView, GetProgress, TrainView, TrainProgressView, generate_smiles_view, generate_progress_view

urlpatterns = [
    path('upload/', UploadCSVView.as_view(), name='upload'),
    path('process-csv/', ProcessCSVView.as_view(), name='process_csv'),
    path('get-progress/<task_id>', GetProgress.as_view(),name='get_progress'),
    path('train/', TrainView.as_view(), name='train'),
    path('train-progress/<task_id>', TrainProgressView.as_view(), name='train_progress'),
    path('generate/', generate_smiles_view, name='generate'),
     path('generate/progress/', generate_progress_view, name='generate_progress'),
]