# Generated by Django 4.2.4 on 2023-08-25 09:58

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('gui', '0006_alter_uploadedcsv_cleaned_smiles_file_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='CleanedSmile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('csv_file', models.CharField(max_length=200)),
                ('cleaned_file', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='gui.uploadedcsv')),
            ],
        ),
    ]
