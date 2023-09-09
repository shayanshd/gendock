# Generated by Django 4.2.4 on 2023-09-09 18:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('gui', '0018_alter_trainlog_max_batch'),
    ]

    operations = [
        migrations.CreateModel(
            name='ReceptorConfiguration',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('receptor_file', models.FileField(upload_to='receptor_files/')),
                ('center_x', models.DecimalField(decimal_places=2, default=0, max_digits=10)),
                ('size_x', models.DecimalField(decimal_places=2, default=0, max_digits=10)),
                ('center_y', models.DecimalField(decimal_places=2, default=0, max_digits=10)),
                ('size_y', models.DecimalField(decimal_places=2, default=0, max_digits=10)),
                ('center_z', models.DecimalField(decimal_places=2, default=0, max_digits=10)),
                ('size_z', models.DecimalField(decimal_places=2, default=0, max_digits=10)),
                ('exhaustive_number', models.IntegerField(default=8)),
            ],
        ),
    ]