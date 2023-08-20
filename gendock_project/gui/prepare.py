import pandas as pd
import os
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import MolStandardize
from rest.lstm_chem.utils.smiles_tokenizer import SmilesTokenizer
from celery import Celery
import time

app = Celery('prepare', broker='redis://localhost:6379/0')

RDLogger.DisableLog('rdApp.*')


class Preprocessor(object):
    def __init__(self):
        self.normarizer = MolStandardize.normalize.Normalizer()
        self.lfc = MolStandardize.fragment.LargestFragmentChooser()
        self.uc = MolStandardize.charge.Uncharger()

    def process(self, smi):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mol = self.normarizer.normalize(mol)
            mol = self.lfc.choose(mol)
            mol = self.uc.uncharge(mol)
            smi = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            return smi
        else:
            return None


# class CleanClass:
#     def __init__(self):
#         self.pp = Preprocessor()

#     def clean_smiles(self, smiles_list):
#         pp_smiles = [self.pp.process(smi) for smi in tqdm(smiles_list)]
#         cl_smiles = list(set([s for s in pp_smiles if s]))
#         return cl_smiles
    
class CleanClass:
    def __init__(self):
        self.pp = Preprocessor()

    def clean_smiles(self, smiles_list, progress_callback):
        total = len(smiles_list)
        cleaned_smiles = []

        for idx, smi in enumerate(smiles_list):
            cleaned_smi = self.pp.process(smi)
            if cleaned_smi:
                cleaned_smiles.append(cleaned_smi)

            progress = (idx + 1) / total * 100
            progress_callback.update_state(state='PROGRESS', meta={'progress': progress})

        return cleaned_smiles


# prepare.py
class DSWorker:
    FN1 = str

    def DSW(self, task_id=None):
        dataset1 = pd.read_csv(self.FN1, sep=',')
        try:
            dataset1 = dataset1[dataset1['smiles'].notnull()]
        except (ValueError, Exception) as e:
            print("Please set the column header name as 'smiles' in Dataset 1.")
            return True

        dataset1['smiles'] = dataset1["smiles"]
        dataset1['length'] = dataset1["smiles"].str.len()

        smiles = dataset1.drop_duplicates()["smiles"].tolist()

        total_smiles = len(smiles)
        processed_count = 0

        for smi in smiles:
            time.sleep(1)  # Simulate processing time

            processed_count += 1
            progress = (processed_count / total_smiles) * 100

            if task_id:
                self.update_task_progress(task_id, progress)

        return

    def update_task_progress(self, task_id, progress):
        from celery import current_app
        current_app.backend.store_result(task_id, {'progress': progress}, 'PROGRESS')
    

