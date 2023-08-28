# your_app/tasks.py
import pandas as pd
import os
from rdkit import Chem, RDLogger
from rdkit.Chem import MolStandardize
from celery import shared_task
from celery_progress.backend import ProgressRecorder
from .models import UploadedCSV, CleanedSmile

RDLogger.DisableLog('rdApp.*')



@shared_task(bind=True)
def process_csv_task(self, pk_list):
    uploaded_csv = []
    pk_names = ''

    for pk in pk_list:
        uploaded_csv.append(UploadedCSV.objects.get(pk=pk))
        # csv_path.append(uploaded_csv.csv_file.path)
        pk_names = pk_names  +str(pk)+ '_'
    cleaned_smiles_dir = './cleaned_smiles'
    os.makedirs(cleaned_smiles_dir, exist_ok=True)
    cleaned_smiles_file = os.path.join(cleaned_smiles_dir, pk_names+'clean.smi')
    try:
        cs_chk =  CleanedSmile.objects.get(cleaned_file=cleaned_smiles_file )
    except:
        cs_chk = None
    if cs_chk:
        cs_chk.delete()
    cs = CleanedSmile.objects.create(cleaned_file = cleaned_smiles_file, task_id = self.request.id)
    for csv_file in uploaded_csv:
        cs.csv_file.add(csv_file)
    cs.save()

    print(pk_names)
    print(uploaded_csv)

    # uploaded_csv = UploadedCSV.objects.get(pk=pk)
    print("Task started")
    # print(uploaded_csv.csv_file)
    progress_recorder = ProgressRecorder(self)
    total_smiles = []
    for csv_path in uploaded_csv:
        dataset1 = pd.read_csv(csv_path.csv_file.path, sep=',')
        try:
            dataset1 = dataset1[dataset1['smiles'].notnull()]
        except (ValueError, Exception) as e:
            return "Please set the column header name as 'smiles' in Dataset 1."
        print(f'total smiles in dataset {csv_path}: ' + str(dataset1.shape[0]))
        dataset1['smiles'] = dataset1["smiles"]
        dataset1['length'] = dataset1["smiles"].str.len()

        smiles = dataset1.drop_duplicates()["smiles"].tolist()
        total_smiles.extend(smiles)
        print(len(smiles))
    total_smiles = list(set(total_smiles))

    
    print('total number of smiles: ' + str(len(total_smiles)))

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

    for i, smi in enumerate(total_smiles):
        cl_smi = process(smi)
        if cl_smi:
            cl_smiles.append(cl_smi)
        if (i + 1) % 100 == 0:
            progress_recorder.set_progress(i + 1, len(total_smiles))

    print('done.')
    print(f'output SMILES num: {len(cl_smiles)}')

    with open(cleaned_smiles_file, 'w') as f:
        for smi in cl_smiles:
            f.write(smi + '\n')
    cs.task_status = 'C'
    cs.save()

    

    return cleaned_smiles_file

