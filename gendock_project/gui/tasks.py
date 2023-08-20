# your_app/tasks.py
import pandas as pd
import os
from rdkit import Chem, RDLogger
from rdkit.Chem import MolStandardize
from celery import shared_task
from celery_progress.backend import ProgressRecorder
from .models import UploadedCSV

RDLogger.DisableLog('rdApp.*')



@shared_task(bind=True)
def process_csv_task(self, csv_path, pk):
    uploaded_csv = UploadedCSV.objects.get(pk=pk)
    print("Task started")
    progress_recorder = ProgressRecorder(self)

    dataset1 = pd.read_csv(csv_path, sep=',')
    try:
        dataset1 = dataset1[dataset1['smiles'].notnull()]
    except (ValueError, Exception) as e:
        print("Please set the column header name as 'smiles' in Dataset 1.")
        return True

    dataset1['smiles'] = dataset1["smiles"]
    dataset1['length'] = dataset1["smiles"].str.len()

    smiles = dataset1.drop_duplicates()["smiles"].tolist()

    print('total smiles in dataset 1: ' + str(dataset1.shape[0]))
    print('total number of smiles: ' + str(len(smiles)))

    normarizer = MolStandardize.normalize.Normalizer()
    lfc = MolStandardize.fragment.LargestFragmentChooser()
    uc = MolStandardize.charge.Uncharger()

    def process(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mol = normarizer.normalize(mol)
            mol = lfc.choose(mol)
            mol = uc.uncharge(mol)
            smi = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            return smi
        else:
            return None

    cl_smiles = []

    for i, smi in enumerate(smiles):
        cl_smi = process(smi)
        if cl_smi:
            cl_smiles.append(cl_smi)
        if (i + 1) % 100 == 0:
            progress_recorder.set_progress(i + 1, len(smiles))

    print('done.')
    print(f'output SMILES num: {len(cl_smiles)}')

    cleaned_smiles_dir = './cleaned_smiles'
    os.makedirs(cleaned_smiles_dir, exist_ok=True)

    cleaned_smiles_file = os.path.join(cleaned_smiles_dir, os.path.basename(csv_path).replace('.csv', '_clean.smi'))

    with open(cleaned_smiles_file, 'w') as f:
        for smi in cl_smiles:
            f.write(smi + '\n')
    
    uploaded_csv.cleaned_smiles_file = cleaned_smiles_file  # Set the cleaned_smiles_path
    uploaded_csv.save()
    return cleaned_smiles_file
