# csv_processor/urls.py
from django.urls import path
from .views import UploadCSVView, ProcessCSVView, GetProgress, TrainView, TrainProgressView

urlpatterns = [
    path('upload/', UploadCSVView.as_view(), name='upload'),
    path('process-csv/', ProcessCSVView.as_view(), name='process_csv'),
    path('get-progress/<task_id>', GetProgress.as_view(),name='get_progress'),
    path('train/', TrainView.as_view(), name='train'),
    path('train-progress/', TrainProgressView.as_view(), name='train_progress'),
]