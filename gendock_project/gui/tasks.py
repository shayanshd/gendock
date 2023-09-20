# your_app/tasks.py
import pandas as pd
import os
from rdkit import Chem, RDLogger
from rdkit.Chem import MolStandardize, Descriptors, PropertyMol
from celery import shared_task
from celery_progress.backend import ProgressRecorder
from .models import UploadedCSV, CleanedSmile, TrainLog
from rest.lstm_chem.utils.config import process_config
from rest.lstm_chem.data_loader import DataLoader
from rest.lstm_chem.model import LSTMChem
from rest.lstm_chem.trainer import LSTMChemTrainer
from rest.lstm_chem.generator import LSTMChemGenerator
from rest.lstm_chem.finetuner import LSTMChemFinetuner

from copy import copy
import json
import shutil
import glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow
from rest.gen_process import *

RDLogger.DisableLog('rdApp.*')

@shared_task(bind=True)
def generate_more_smiles(self, global_generation, sample_number, desired_length):

    progress_recorder = ProgressRecorder(self)

    print('Starting process for generation ' + str(global_generation))
    master_table = pd.read_csv(
        './rest/generations/master_results_table_gen' + str(global_generation - 1) + '.csv',
        sep=','
    )

    new_scores = pd.read_csv('./rest/generations/results/results_gen' + str(global_generation - 1) + '.csv', sep=',')
    new_scores = new_scores.groupby("Ligand").min()["Binding Affinity"].reset_index()
    new_scores['id'] = new_scores['Ligand'].str.split("_").str[1].str.split("gen").str[0].str.split("id").str[1]
    new_scores['gen'] = new_scores['Ligand'].str.split("_").str[1].str.split("gen").str[1]
    new_scores['score'] = new_scores["Binding Affinity"]
    new_scores = new_scores[['id', 'gen', 'score']]
    new_scores.id = new_scores.id.astype(str)
    new_scores.gen = new_scores.gen.astype(int)
    master_table.id = master_table.id.astype(str)
    master_table.gen = master_table.gen.astype(int)
    new_table = pd.merge(master_table, new_scores, on=['id', 'gen'], suffixes=('_old', '_new'), how='left')
    new_table['score'] = np.where(new_table['score_new'].isnull(), new_table['score_old'], new_table['score_new'])
    new_table = new_table.drop(['score_old', 'score_new'], axis=1)
    new_table['weight'] = new_table['smile'].apply(lambda x: Chem.Descriptors.MolWt(Chem.MolFromSmiles(x)))
    new_table = new_table.sort_values('score', ascending=True)

    new_table.to_csv('./rest/generations/master_results_table_gen' + str(global_generation - 1) + '.csv', index=False)

    # Select top X ranked by score for training data to refine the molecule generator RNN
    training_smiles = list(set(list(new_table.head(36)['smile'])))
    len(training_smiles)

    training_fingerprints = []
    for smile in training_smiles:
        training_fingerprints.append(Chem.RDKFingerprint(Chem.MolFromSmiles(smile)))

    def calc_similarity_score(row):
        fingerprint = Chem.RDKFingerprint(Chem.MolFromSmiles(row['smile']))
        similarity = np.max(DataStructs.BulkTanimotoSimilarity(fingerprint, training_fingerprints))
        adj_factor = (1 / similarity) ** .333
        adj_score = row['score'] * adj_factor
        return adj_score

    similarity_adjusted = new_table.copy(deep=True)
    similarity_adjusted = similarity_adjusted[similarity_adjusted['weight'] < 500]
    similarity_adjusted['similarity_adj_score'] = similarity_adjusted.apply(calc_similarity_score, axis=1)
    similarity_adjusted = similarity_adjusted.sort_values('similarity_adj_score', ascending=True)

    # Select top X ranked by similarity adjusted score for training data to refine the molecule generator RNN (ensures diversity)
    training_smiles += list(similarity_adjusted.head(5)['smile'])
    len(training_smiles)

    weight_adjusted = new_table.copy(deep=True)
    weight_adjusted['weight_adj_score'] = weight_adjusted.apply(GenProcess.calc_weight_score, axis=1)
    weight_adjusted = weight_adjusted.sort_values('weight_adj_score', ascending=True)

    # Select top X ranked by similarity adjusted score for training data to refine the molecule generator RNN (ensures diversity)
    training_smiles += list(weight_adjusted.head(5)['smile'])
    len(training_smiles)

    # Generate some with the base original model
    CONFIG_FILE = 'rest/experiments/LSTM_Chem/config.json'
    config = process_config(CONFIG_FILE)
    modeler = LSTMChem(config, session='generate')
    generator = LSTMChemGenerator(modeler)

    sample_number = 20
    base_generated = generator.sample(progress_recorder, num=sample_number)

    base_generated_mols = GenProcess.validate_mols(base_generated)
    base_generated_smiles = GenProcess.convert_mols_to_smiles(base_generated_mols)
    random.shuffle(base_generated_smiles)
    random.shuffle(base_generated_smiles)

    # Select X for training data to refine the molecule generator RNN (ensures diversity)
    training_smiles += base_generated_smiles[0:5]
    len(training_smiles)

    master_table = pd.read_csv(
        './rest/generations/master_results_table_gen' + str(global_generation - 1) + '.csv',
        sep=','
    )

    # Save the list of smiles to train on
    with open('./rest/generations/training/gen' + str(global_generation) + '_training.smi', 'w') as f:
        for item in training_smiles:
            f.write("%s\n" % item)

    # Retrain the network to create molecules more like those selected above

    config = process_config('rest/experiments/LSTM_Chem/config.json')
    if global_generation == 1:
        config['model_weight_filename'] = 'rest/experiments/LSTM_Chem/checkpoints/' + 'LSTM_Chem-baseline-model-full.hdf5'
    else:
        config['model_weight_filename'] = 'rest/experiments/LSTM_Chem/checkpoints/finetuned_gen' + str(
            global_generation - 1) + '.hdf5'

    config['finetune_data_filename'] = './rest/generations/training/gen' + str(global_generation) + '_training.smi'

    modeler = LSTMChem(config, session='finetune')
    finetune_dl = DataLoader(config, data_type='finetune')

    finetuner = LSTMChemFinetuner(modeler, finetune_dl)
    finetuner.finetune()

    finetuner.model.save_weights(
        'rest/experiments/LSTM_Chem/checkpoints/finetuned_gen' + str(global_generation) + '.hdf5'
    )

    config['model_weight_filename'] = 'rest/experiments/LSTM_Chem/checkpoints/finetuned_gen' + str(
        global_generation) + '.hdf5'

    modeler = LSTMChem(config, session='generate')
    generator = LSTMChemGenerator(modeler)

    sample_number = sample_number
    sampled_smiles = generator.sample(progress_recorder, num=sample_number)

    valid_mols = []
    for smi in sampled_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_mols.append(mol)

    if len(valid_mols)  == 0:
        return [0,0,0]

    validity = len(valid_mols) / sample_number

    valid_smiles = [Chem.MolToSmiles(mol) for mol in valid_mols]
    uniqueness = len(set(valid_smiles)) / len(valid_smiles)

    # Of valid smiles generated, how many are truly original vs occurring in the training data

    training_data = pd.read_csv('./rest/datasets/all_smiles_clean.smi', header=None)
    training_set = set(list(training_data[0]))
    original = []
    for smile in list(set(valid_smiles)):
        if smile not in training_set:
            original.append(smile)
    
    originality = len(set(original)) / len(set(valid_smiles))

    valid_smiles = list(set(valid_smiles))
    gen0_all = valid_smiles
    gen0 = []
    for smi in gen0_all:
        w = Chem.Descriptors.MolWt(Chem.MolFromSmiles(smi))
        if w < 500:
            gen0.append(smi)
    len(gen0)
    valid_smiles = gen0
    len(valid_smiles)

    # take the valid smiles from above and run them through process to add to tracking table
    mols_for_next_generation = GenProcess.validate_mols(valid_smiles)

    master_table = pd.read_csv(
        './rest/generations/master_results_table_gen' + str(global_generation - 1) + '.csv',
        sep=','
    )

    new_mols_to_test = GenProcess.append_to_tracking_table(
        master_table, mols_for_next_generation, 'generated', global_generation
    )
    mols_for_pd = new_mols_to_test[0]
    mols_for_export = new_mols_to_test[1]

    master_table = master_table._append(mols_for_pd)
    master_table = master_table.reset_index(drop=True)
    master_table.to_csv('./rest/generations/master_results_table_gen' + str(global_generation) + '.csv', index=False)

    GenProcess.write_gen_to_sdf(mols_for_export, global_generation, desired_length)
    print('Writing generated smiles to ' + str(global_generation))
    print('Done!')

    return [validity, uniqueness, originality]

@shared_task(bind=True)
def process_nd_worker(self, global_generation):
    df = pd.read_csv('rest/checklist.csv')
    vina_address = str(df['Address'].loc[df['Product'].str.contains('Vina', case=False)].values[0])
    mgl_address = str(df['Address'].loc[df['Product'].str.contains('MGL', case=False)].values[0])
    MGL_ROOT_PATH = mgl_address  # enter root of mgltools installed on your device
    PYTHONSH = os.path.join(MGL_ROOT_PATH, 'bin/pythonsh')
    LIGAND_PREPARE = os.path.join(MGL_ROOT_PATH, r'MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py')
    VINA_PATH = vina_address

    # Create a ProgressRecorder instance
    progress_recorder = ProgressRecorder(self)

    workingdir = os.getcwd() + '/rest'

    gendir = workingdir + f'/generations/gen{global_generation}'
    workingdir = workingdir + r'/babeltest'

    try:
        shutil.rmtree(workingdir)
    except OSError:
        print("Deletion of the directory %s failed" % workingdir)

    try:
        os.mkdir(workingdir)
    except OSError:
        print("Creation of the directory %s failed" % workingdir)

    CONFIG_PATH = os.getcwd() + '/rest/receptor_conf.txt'
    print(CONFIG_PATH)

    # Convert smiles to PDB
    smi_to_pdb = f'(cd {workingdir} && obabel -isdf {gendir}.sdf -opdb -O mysm.pdb -h -m)'
    os.system(smi_to_pdb)

    # Prepare ligands
    counter = len(glob.glob1(workingdir, "*.pdb"))
    progress_recorder.set_progress(0, counter)
    for i in range(counter):
        prepare_command = f'(cd "{workingdir}" && ' \
                          f'"{PYTHONSH}" "{LIGAND_PREPARE}" -l mysm{i + 1}.pdb -o mysm{i + 1}.pdbqt)'
        os.system(prepare_command)

    # Dock ligands
    counter = len(glob.glob1(workingdir, "*.pdbqt"))
    print('Total number of smiles to dock: ' + str(counter))
    for i in range(counter):
        dock_command = fr'(cd {workingdir} && "{VINA_PATH}/vina" --ligand mysm{i + 1}.pdbqt ' \
                    f'--config {CONFIG_PATH} --log mysm{i + 1}.log)'
        os.system(dock_command)
        
    # Calculate progress as a percentage and update it
        progress_recorder.set_progress(i+1, counter)

    
    # Add to result table
    suppl = Chem.SDMolSupplier(gendir + '.sdf')
    print(type(suppl))
    print(suppl)
    df2 = pd.DataFrame(columns=['Ligand', 'Binding Affinity'])
    counter = len(glob.glob1(workingdir, "*.log"))
    if counter == len(suppl):
        for i in range(len(suppl)):
            mol = suppl[i].GetProp("Title")
            filename = fr'{workingdir}/mysm{i + 1}.log'
            with open(filename) as myFile:
                for num, line in enumerate(myFile, 1):
                    if '  1  ' in line:
                        k = line[line.find('-'):]
                        k = k[:k.find(' ')]
                        break
            df2.loc[i] = ['6lu7cov_' + mol, k]

    df2.to_csv('rest/all_smiles.csv', index=False)
    df2.to_csv('rest/generations/results/results_gen' + str(global_generation) + '.csv', index=False)
    return 'Done'

@shared_task(bind=True)
def generate_smiles(self, sample_number, desired_length):
    # Define the paths and configurations
    CONFIG_FILE = 'rest/experiments/LSTM_Chem/config.json'
    config = process_config(CONFIG_FILE)

    # Initialize the modeler and generator
    modeler = LSTMChem(config, session='generate')
    generator = LSTMChemGenerator(modeler)
    progress_recorder = ProgressRecorder(self)
    # Sample smiles
    sampled_smiles = generator.sample(progress_recorder,num=sample_number)

    # Save sampled smiles to a file
    with open('./rest/generations/gen0notvalid.smi', 'w') as f:
        for item in sampled_smiles:
            f.write("%s\n" % item)

    # Read sampled smiles from the file and preprocess
    sampled_smiles = pd.read_csv('./rest/generations/gen0notvalid.smi', header=None)
    sampled_smiles = set(list(sampled_smiles[0]))

    # Validate valid smiles and calculate metrics
    valid_mols = GenProcess.validate_mols(sampled_smiles)
    if len(valid_mols)  == 0:
        return [0,0,0]
    validity = len(valid_mols) / sample_number

    valid_smiles = [Chem.MolToSmiles(mol) for mol in valid_mols]
    uniqueness = len(set(valid_smiles)) / len(valid_smiles)

    # Check originality against training data
    training_data = pd.read_csv('./rest/datasets/all_smiles_clean.smi', header=None)
    training_set = set(list(training_data[0]))
    original = [smile for smile in valid_smiles if smile not in training_set]
    originality = len(set(original)) / len(set(valid_smiles))

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
    new_mols_to_test = GenProcess.append_to_tracking_table(master_table, gen0_mols, 'generated', 0, config.data_filename)

    mols_for_pd = new_mols_to_test[0]
    mols_for_export = new_mols_to_test[1]
    master_table = master_table._append(mols_for_pd)
    master_table = master_table.reset_index(drop=True)
    master_table.to_csv('./rest/generations/master_results_table.csv', index=False)
    master_table.to_csv('./rest/generations/master_results_table_gen0.csv', index=False)
    GenProcess.write_gen_to_sdf(mols_for_export, 0, 2000)
    return [validity, uniqueness, originality]

@shared_task(bind=True)
def start_training(self):
  
    log_file_path = './celery.logs'  # Specify the correct path to your celery logs file
    
    # Clear the log file before training starts
    with open(log_file_path, 'w') as log_file:
        log_file.write('')

    CONFIG_FILE = 'rest/experiments/LSTM_Chem/config.json'
    config = process_config(CONFIG_FILE)
    tl = TrainLog.objects.create(task_id = self.request.id, task_status = 'P')  
    tl.max_epoch = int(config.num_epochs) 
    tl.save()
    modeler = LSTMChem(config, session='train')
    train_dl = DataLoader(config, data_type='train')
    valid_dl = copy(train_dl)
    valid_dl.data_type = 'valid'
    trainer = LSTMChemTrainer(modeler, train_dl, valid_dl,task_id = self.request.id)
    # Run trainer.train()
 
    h = trainer.train()
    # Save weights
 
    tl.epoch = int(h.params['epochs'])
    tl.val_loss = h.history['val_loss']
    tl.train_loss = h.history['loss']
    tl.task_status = 'C'
    tl.save()
    trainer.model.save_weights('rest/experiments/LSTM_Chem/checkpoints/LSTM_Chem-baseline-model-full.hdf5')
    with open(CONFIG_FILE, 'r') as config_file:
        config_data = json.load(config_file)
        config_data['model_weight_filename'] = 'rest/experiments/LSTM_Chem/checkpoints/LSTM_Chem-baseline-model-full.hdf5'
    with open(CONFIG_FILE, 'w') as config_file:
        json.dump(config_data, config_file,indent=2)

    return 'DONE'  # Returning logs to access in the view

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
        dataset1 = dataset1['smiles']
        smiles = dataset1.drop_duplicates()
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
    cl_smiles = list(set([s for s in cl_smiles if s]))
    print('done.')
    print(f'output SMILES num: {len(cl_smiles)}')

    with open(cleaned_smiles_file, 'w') as f:
        for smi in cl_smiles:
            f.write(smi + '\n')
    cs.task_status = 'C'
    cs.save()
    return cleaned_smiles_file
