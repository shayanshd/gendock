# Generated by Django 4.2.4 on 2023-09-18 23:32

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('gui', '0021_receptorconfiguration_receptor_file'),
    ]

    operations = [
        migrations.AlterField(
            model_name='receptorconfiguration',
            name='receptor_file',
            field=models.FileField(blank=True, null=True, upload_to='receptor_files/'),
        ),
    ]
