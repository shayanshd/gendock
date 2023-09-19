# csv_processor/urls.py
from django.urls import path
from .views import UploadCSVView, ProcessCSVView, GetProgress, TrainView, TrainProgressView, GenerateSmilesView, GenerateProgressView, start_docking

urlpatterns = [
    path('upload/', UploadCSVView.as_view(), name='upload'),
    path('process-csv/', ProcessCSVView.as_view(), name='process_csv'),
    path('get-progress/<task_id>', GetProgress.as_view(),name='get_progress'),
    path('train/', TrainView.as_view(), name='train'),
    path('train-progress/<task_id>', TrainProgressView.as_view(), name='train_progress'),
    path('generate/', GenerateSmilesView.as_view(), name='generate'),
    path('generate/progress/<task_id>', GenerateProgressView.as_view(), name='generate_progress'),
    path('start_docking/', start_docking, name='start_docking'),
]