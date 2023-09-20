from django.db import models
import os
# Create your models here.\
class UploadedCSV(models.Model):
    csv_file = models.FileField(upload_to='uploads/')

    def __str__(self):
        return self.csv_file.name

    def delete(self, *args, **kwargs):
        # Delete the CSV file from the filesystem
        if self.csv_file:
            try:
                os.remove(self.csv_file.path)
            except:
                print('csv file not found')

        # Delete associated cleaned smile files
        cleaned_smiles = CleanedSmile.objects.filter(csv_file=self)
        for cleaned_smile in cleaned_smiles:
            if cleaned_smile.cleaned_file:
                try:
                    os.remove(cleaned_smile.cleaned_file)
                except:
                    print('smi file not found')
            cleaned_smile.delete()

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
    
class TrainLog(models.Model):
    task_id = models.CharField(max_length=50, unique=True)
    epoch = models.IntegerField(default=0)
    max_epoch = models.IntegerField(null=True)
    train_loss = models.CharField(max_length=20, null=True)
    val_loss = models.CharField(max_length=20, null=True)
    cur_batch = models.IntegerField(default=0)
    max_batch = models.IntegerField(default=1)

    TASK_STATUS_CHOICES = (
    ('P', 'Processing'),
    ('C', 'Completed'),
    ('F', 'Failed'),
    ('N', 'Not Started')
    )
    task_status = models.CharField(max_length=1, choices=TASK_STATUS_CHOICES, default='N')
   
class ReceptorConfiguration(models.Model):
    receptor_file = models.FileField(upload_to='receptor_files/')
    center_x = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    size_x = models.DecimalField(max_digits=10, decimal_places=2, default=30)
    center_y = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    size_y = models.DecimalField(max_digits=10, decimal_places=2, default=30)
    center_z = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    size_z = models.DecimalField(max_digits=10, decimal_places=2, default=30)
    exhaustive_number = models.IntegerField(default=8)

    def __str__(self):
        return f'Receptor Configuration {self.id}'
    
    def delete(self, *args, **kwargs):
        # Delete the CSV file from the filesystem
        if self.receptor_file:
            try:
                os.remove(self.receptor_file.path)
            except:
                print('csv file not found')
