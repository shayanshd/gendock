# Generated by Django 4.2.4 on 2023-08-23 14:18

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('gui', '0005_alter_uploadedcsv_cleaned_smiles_file'),
    ]

    operations = [
        migrations.AlterField(
            model_name='uploadedcsv',
            name='cleaned_smiles_file',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.DeleteModel(
            name='CleanSmiles',
        ),
    ]
