# csv_processor/urls.py
from django.urls import path
from .views import UploadCSVView, ProcessCSVView, GetProgress

urlpatterns = [
    path('upload/', UploadCSVView.as_view(), name='upload'),
    # path('csv-list/', CSVListView.as_view(), name='csv_list'),
    path('process-csv/<int:pk>', ProcessCSVView.as_view(), name='process_csv'),
    path('get-progress/<task_id>', GetProgress.as_view(),name='get_progress')
]