from django.db import models
import os
# Create your models here.\
class UploadedCSV(models.Model):
    csv_file = models.FileField(upload_to='uploads/')
    # cleaned_smiles_file = models.CharField(max_length=100,null=True, blank=True)

    def __str__(self):
        return self.csv_file.name

    def delete(self, *args, **kwargs):
        # Delete the CSV file from the filesystem
        if self.csv_file:
            os.remove(self.csv_file.path)
        super().delete(*args, **kwargs)


class CleanedSmile(models.Model):
    cleaned_file = models.CharField(max_length=100,null=True, blank=True)
    csv_file = models.ManyToManyField(UploadedCSV)
    task_id = models.CharField(max_length=50, null=True, blank=True)

    TASK_STATUS_CHOICES = (
    ('P', 'Processing'),
    ('C', 'Completed'),
    ('F', 'Failed'),
    ('N', 'Not Started')
    )
    task_status = models.CharField(max_length=1, choices=TASK_STATUS_CHOICES, default='N')

    def __str__(self):
        return self.cleaned_file