from django.db import models
import os
# Create your models here.
class UploadedCSV(models.Model):
    csv_file = models.FileField(upload_to='uploads/')
    cleaned_smiles_file = models.CharField(max_length=255, blank=True, null=True)
    task_id = models.CharField(max_length=50, null=True, blank=True)

    def __str__(self):
        return self.csv_file.name

    def delete(self, *args, **kwargs):
        # Delete the CSV file from the filesystem
        if self.csv_file:
            os.remove(self.csv_file.path)
        super().delete(*args, **kwargs)