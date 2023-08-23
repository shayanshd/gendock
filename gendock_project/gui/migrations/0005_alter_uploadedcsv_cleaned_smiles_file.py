# Generated by Django 4.2.4 on 2023-08-23 14:03

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('gui', '0004_cleansmiles'),
    ]

    operations = [
        migrations.AlterField(
            model_name='uploadedcsv',
            name='cleaned_smiles_file',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to='gui.cleansmiles'),
        ),
    ]
