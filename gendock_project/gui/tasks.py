# your_app/tasks.py
import pandas as pd
import os
from rdkit import Chem, RDLogger
from rdkit.Chem import MolStandardize
from celery import shared_task
from celery_progress.backend import ProgressRecorder
from .models import UploadedCSV, CleanedSmile
from rest.lstm_chem.utils.config import process_config
from rest.lstm_chem.data_loader import DataLoader
from rest.lstm_chem.model import LSTMChem
from rest.lstm_chem.trainer import LSTMChemTrainer
from rest.lstm_chem.generator import LSTMChemGenerator
from copy import copy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow
from rest.gen_process import *

RDLogger.DisableLog('rdApp.*')

@shared_task(bind=True)
def generate_smiles(self, sample_number, desired_length):
    log_file_path = './celery.logs'  # Specify the correct path to your celery logs file   
    # Clear the log file before training starts
    with open(log_file_path, 'w') as log_file:
        log_file.write('')
    # Define the paths and configurations
    CONFIG_FILE = 'rest/experiments/LSTM_Chem/config.json'
    config = process_config(CONFIG_FILE)

    # Initialize the modeler and generator
    modeler = LSTMChem(config, session='generate')
    generator = LSTMChemGenerator(modeler)

    # Sample smiles
    sampled_smiles = generator.sample(num=sample_number)

    # Save sampled smiles to a file
    with open('./rest/generations/gen0notvalid.smi', 'w') as f:
        for item in sampled_smiles:
            f.write("%s\n" % item)

    # Read sampled smiles from the file and preprocess
    sampled_smiles = pd.read_csv('./rest/generations/gen0notvalid.smi', header=None)
    sampled_smiles = set(list(sampled_smiles[0]))

    # Validate valid smiles and calculate metrics
    valid_mols = GenProcess.validate_mols(sampled_smiles)
    print(sample_number)
    print('Validity: ', f'{len(valid_mols) / sample_number:.2%}')

    valid_smiles = [Chem.MolToSmiles(mol) for mol in valid_mols]
    print('Uniqueness: ', f'{len(set(valid_smiles)) / len(valid_smiles):.2%}')

    # Check originality against training data
    training_data = pd.read_csv('./rest/datasets/all_smiles_clean.smi', header=None)
    training_set = set(list(training_data[0]))
    original = [smile for smile in valid_smiles if smile not in training_set]
    print('Originality: ', f'{len(set(original)) / len(set(valid_smiles)):.2%}')

    # Save valid smiles to a file
    with open('./rest/generations/gen0.smi', 'w') as f:
        for item in valid_smiles:
            f.write("%s\n" % item)

    #########################################################
    # Additional processing steps
    gen0_table = pd.read_csv('./rest/generations/gen0.smi', sep=',', header=None)
    gen0_all = list(gen0_table[0])[0:10000]
    gen0 = []
    for smi in gen0_all:
        w = Chem.Descriptors.MolWt(Chem.MolFromSmiles(smi))
        if w < 500:
            gen0.append(smi)

    gen0_mols = GenProcess.validate_mols(gen0)
    gen0_mols = GenProcess.initialize_generation_from_mols(gen0_mols, desired_length)
    check_master_exists = os.path.exists('./rest/generations/master_results_table.csv')
    if check_master_exists:
        os.remove('./rest/generations/master_results_table.csv')

    raw_table = {}
    df = pd.DataFrame(raw_table, columns=['id', 'gen', 'smile', 'source', 'score'])
    df.to_csv('./rest/generations/master_results_table.csv', index=False)
    master_table = pd.read_csv('./rest/generations/master_results_table.csv', sep=',')
    new_mols_to_test = GenProcess.append_to_tracking_table(master_table, gen0_mols, 'generated', 0)
    mols_for_pd = new_mols_to_test[0]
    mols_for_export = new_mols_to_test[1]
    master_table = master_table.append(mols_for_pd)
    master_table = master_table.reset_index(drop=True)
    master_table.to_csv('./rest/generations/master_results_table.csv', index=False)
    master_table.to_csv('./rest/generations/master_results_table_gen0.csv', index=False)
    GenProcess.write_gen_to_sdf(mols_for_export, 0, 2000)
    return 'DONE'

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

@shared_task(bind=True)
def start_training(self):
    log_file_path = './celery.logs'  # Specify the correct path to your celery logs file
    
    # Clear the log file before training starts
    with open(log_file_path, 'w') as log_file:
        log_file.write('')

    CONFIG_FILE = 'rest/experiments/LSTM_Chem/config.json'
    config = process_config(CONFIG_FILE)
    modeler = LSTMChem(config, session='train')
    train_dl = DataLoader(config, data_type='train')
    valid_dl = copy(train_dl)
    valid_dl.data_type = 'valid'
    trainer = LSTMChemTrainer(modeler, train_dl, valid_dl)
    # Run trainer.train()
    trainer.train()
    # Save weights
    trainer.model.save_weights('rest/experiments/LSTM_Chem/checkpoints/LSTM_Chem-baseline-model-full.hdf5')

    return 'DONE'  # Returning logs to access in the view
