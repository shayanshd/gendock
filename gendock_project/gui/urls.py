# csv_processor/urls.py
from django.urls import path
from .views import *

urlpatterns = [
    path('upload/', UploadCSVView.as_view(), name='upload'),
    path('process-csv/', ProcessCSVView.as_view(), name='process_csv'),
    path('get-progress/<task_id>', GetProgress.as_view(),name='get_progress'),
    path('train/', TrainView.as_view(), name='train'),
    path('train-progress/<task_id>', TrainProgressView.as_view(), name='train_progress'),
    path('generate/', GenerateSmilesView.as_view(), name='generate'),
    path('generate/progress/<task_id>', GenerateProgressView.as_view(), name='generate_progress'),
    path('start-docking/', DockingView.as_view(), name='start_docking'),
    path('docking-progress/<str:dock_task_id>/<int:generation_number>', DockingProgressView.as_view(), name='docking_progress'),
    path('docking-master-table/<int:generation_number>', DockingMasterTableView.as_view(), name='docking_master_table'),
    path('instructions/', instructions_view, name='instructions'),
]