# Generated by Django 4.2.4 on 2023-09-09 19:51

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('gui', '0020_remove_receptorconfiguration_receptor_file'),
    ]

    operations = [
        migrations.AddField(
            model_name='receptorconfiguration',
            name='receptor_file',
            field=models.FileField(default='', upload_to='receptor_files/'),
        ),
    ]