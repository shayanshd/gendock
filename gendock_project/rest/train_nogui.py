import sys
import os
import glob
import pandas as pd
import numpy as np
from copy import copy
from rdkit import RDLogger, Chem, DataStructs
from lstm_chem.utils.config import process_config
from lstm_chem.model import LSTMChem
from lstm_chem.generator import LSTMChemGenerator
from lstm_chem.trainer import LSTMChemTrainer
from lstm_chem.data_loader import DataLoader
from cleanup_smiles import CleanClass
import random
import shutil
from gen_process import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow

RDLogger.DisableLog('rdApp.*')
GLOBAL_GENERATION = 0


class DSWorker():

    FN1 = str
    FN2 = str
    def DSW(self):
        # print(self.FN1)
        dataset1 = pd.read_csv(self.FN1, sep=',')
        try:

            dataset1 = dataset1[dataset1['smiles'].notnull()]
        except (ValueError, Exception) as e:
            print("Please set the column header name as 'smiles' in Dataset 1.")
            return True

        dataset1['smiles'] = dataset1["smiles"]
        dataset1['length'] = dataset1["smiles"].str.len()
        dataset1.head()
        # They are ready to be appended once run through a canonizer and then drop duplicates
        dataset1 = dataset1['smiles']
        smiles = dataset1.drop_duplicates()
        print('total smiles in dataset 1: ' + str(dataset1.shape[0]))
        # if self.FN2 is not None:
        #     # print(self.FN2)
        #     dataset2 = pd.read_csv(self.FN2, sep=',')
        #     # dataset2 = dataset2[dataset2['smiles'].notnull()]
        #     try:

        #         dataset2 = dataset2[dataset2['smiles'].notnull()]
        #     except (ValueError, Exception) as e:
        #         # print(e)
        #         print("Please set the column header name as 'smiles' in Dataset 2.")
        #         return True
        #     dataset2['smiles'] = dataset2["smiles"]
        #     dataset2['length'] = dataset2["smiles"].str.len()
        #     dataset2.head()

        #     dataset2 = dataset2['smiles']
        #     dataset2 = dataset2.drop_duplicates()
        #     print('total smiles in dataset 2: ' + str(dataset2.shape[0]))
        #     smiles = smiles.append(dataset2)
        smiles = smiles.drop_duplicates()
        print('total number of smiles: ' + str(smiles.shape[0]))
        smiles.to_csv(r'./datasets/all_smiles.smi', header=None, index=None, sep='\t', mode='w')
        cc = CleanClass(r'./datasets/all_smiles.smi', r'./datasets/all_smiles_clean.smi')
        cc.cleaner()


class INWorker():

    def IN(self):
        CONFIG_FILE = 'experiments/LSTM_Chem/config.json'
        config = process_config(CONFIG_FILE)

        modeler = LSTMChem(config, session='train')

        train_dl = DataLoader(config, data_type='train')

        valid_dl = copy(train_dl)
        valid_dl.data_type = 'valid'

        trainer = LSTMChemTrainer(modeler, train_dl, valid_dl)

        trainer.train()

        # Save weights of the trained model
        trainer.model.save_weights('experiments/LSTM_Chem/checkpoints/LSTM_Chem-baseline-model-full.hdf5')


class IGWorker():

    sample_number = int
    desired_length = int
    def IGW(self):
        CONFIG_FILE = 'experiments/LSTM_Chem/config.json'
        config = process_config(CONFIG_FILE)
        config['model_weight_filename'] = 'experiments/LSTM_Chem/checkpoints/LSTM_Chem-23-0.21.hdf5'
        # print(config)

        modeler = LSTMChem(config, session='generate')
        generator = LSTMChemGenerator(modeler)
        # print(config)

        sampled_smiles = generator.sample(num=self.sample_number)

        with open('./generations/gen0notvalid.smi', 'w') as f:
            for item in sampled_smiles:
                f.write("%s\n" % item)

        sampled_smiles = pd.read_csv('./generations/gen0notvalid.smi', header=None)
        sampled_smiles = set(list(sampled_smiles[0]))
        # print(sampled_smiles.head())

        valid_mols = []
        for smi in sampled_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_mols.append(mol)
        # low validity
        print('Validity: ', f'{len(valid_mols) / self.sample_number:.2%}')

        valid_smiles = [Chem.MolToSmiles(mol) for mol in valid_mols]
        # high uniqueness
        print('Uniqueness: ', f'{len(set(valid_smiles)) / len(valid_smiles):.2%}')

        # Of valid smiles generated, how many are truly original vs ocurring in the training data
        training_data = pd.read_csv('./datasets/all_smiles_clean.smi', header=None)
        training_set = set(list(training_data[0]))
        original = []
        for smile in valid_smiles:
            if not smile in training_set:
                original.append(smile)
        print('Originality: ', f'{len(set(original)) / len(set(valid_smiles)):.2%}')

        with open('./generations/gen0.smi', 'w') as f:
            for item in valid_smiles:
                f.write("%s\n" % item)
        #########################################################
        gen0_table = pd.read_csv('./generations/gen0.smi', sep=',', header=None)
        gen0_all = list(gen0_table[0])[0:10000]
        gen0 = []
        for smi in gen0_all:
            w = Chem.Descriptors.MolWt(Chem.MolFromSmiles(smi))
            if w < 500:
                gen0.append(smi)
        # print(len(gen0))
        gen0_mols = GenProcess.validate_mols(gen0)
        # print(len(gen0_mols))

        gen0_mols = GenProcess.initialize_generation_from_mols(gen0_mols, self.desired_length)
        # print(len(gen0_mols))\
        check_master_exits = os.path.exists('./generations/master_results_table.csv')
        if check_master_exits:
            os.remove('./generations/master_results_table.csv')
        raw_table = {}
        df = pd.DataFrame(raw_table, columns = ['id','gen','smile','source','score'])
        df.to_csv('./generations/master_results_table.csv', index=False)
        master_table = pd.read_csv('./generations/master_results_table.csv', sep=',')
        # print(master_table.shape[0])

        new_mols_to_test = GenProcess.append_to_tracking_table(master_table, gen0_mols, 'generated', 0)
        mols_for_pd = new_mols_to_test[0]
        mols_for_export = new_mols_to_test[1]
        master_table = master_table._append(mols_for_pd)
        # print(len(mols_for_export))

        master_table = master_table.reset_index(drop=True)
        master_table.to_csv(r'./generations/master_results_table.csv', index=False)
        master_table.to_csv(r'./generations/master_results_table_gen0.csv', index=False)
        GenProcess.write_gen_to_sdf(mols_for_export, 0, 2000)
        print('Done!')


class IDWorker():

    def IDW(self):
        df = pd.read_csv('checklist.csv')
        vina_address = str(df['Address'].loc[df['Product'].str.contains('Vina', case=False)].values[0])
        mgl_address = str(df['Address'].loc[df['Product'].str.contains('MGL', case=False)].values[0])
        MGL_ROOT_PATH = mgl_address  # enter root of mgltools installed on your device
        PYTHONSH = os.path.join(MGL_ROOT_PATH, 'bin/pythonsh')
        LIGAND_PREPARE = os.path.join(MGL_ROOT_PATH, r'MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py')
        VINA_PATH = vina_address
        # print(os.getcwd())
        workingdir = os.getcwd()
        print(workingdir)
        gendir = workingdir + r'/generations/gen0'
        workingdir = workingdir + r'/babeltest'
        try:
            shutil.rmtree(workingdir)
        except OSError:
            print("Deletion of the directory %s failed" % workingdir)
        else:
            print("Successfully deleted the directory %s" % workingdir)
        try:
            os.mkdir(workingdir)
        except OSError:
            print("Creation of the directory %s failed" % workingdir)
        else:
            print("Successfully created the directory %s " % workingdir)
        # print(gendir)
        # print(workingdir)
        CONFIG_PATH = os.getcwd() + r'/receptor_conf.txt'
        print(CONFIG_PATH)
        def smi_to_pdb(smiles: str, output_name='output'):  # define output name without file extension

            command = f'(cd {workingdir} && obabel -isdf {smiles}.sdf ' \
                      f'-opdb -O {output_name}.pdb -h -m)'
            os.system(command)

        def prepare(pdb_path, pythonsh=PYTHONSH, mglprepare=LIGAND_PREPARE):
            counter = len(glob.glob1(workingdir, "*.pdb"))
            # print(counter)
            for i in range(counter):
                command = f'(cd "{workingdir}" && ' \
                          f'"{pythonsh}" "{mglprepare}" -l {pdb_path}{i + 1}.pdb -o {pdb_path}{i + 1}.pdbqt)'
                os.system(command)

        def dock(smiles):
            counter = len(glob.glob1(workingdir, "*.pdbqt"))
            print('Total number of smiles to dock: ' + str(counter))
            for i in range(counter):
                command = fr'(cd {workingdir} && "{VINA_PATH}/vina" --ligand {smiles}{i + 1}.pdbqt ' \
                          f'--config {CONFIG_PATH} --log {smiles}{i + 1}.log)'
                os.system(command)

        def addnames():
            suppl = Chem.SDMolSupplier(gendir + '.sdf')
            # df = pd.DataFrame(columns=['mol_name'])
            df2 = pd.DataFrame(columns=['Ligand', 'Binding Affinity'])
            counter = len(glob.glob1(workingdir, "*.log"))
            if counter == len(suppl):
                for i in range(len(suppl)):
                    mol = suppl[i].GetProp("Title")
                    filename = fr'{workingdir}/mysm{i + 1}.log'
                    with open(filename) as myFile:
                        for num, line in enumerate(myFile, 1):

                            if '  1  ' in line:
                                # print('found at line:', num, line)
                                k = line[line.find('-'):]
                                k = k[:k.find(' ')]
                                break

                    df2.loc[i] = ['6lu7cov_' + mol, k]
            df2.to_csv(r'all_smiles.csv', index=False)
            df2.to_csv(r'generations/results/results_gen0.csv', index=False)
            return df2

        smi_to_pdb(gendir, 'mysm')
        prepare('mysm')
        dock('mysm')
        addnames()


class NGWorker():
    sample_number = int
    desired_length = int
    GLOBAL_GENERATION = int

    def NGW(self):
        def proc1():
            # GLOBAL_GENERATION = self.gennum.text()
            print('Starting process for generation ' + str(self.GLOBAL_GENERATION))
            master_table = pd.read_csv(
                './generations/master_results_table_gen' + str(self.GLOBAL_GENERATION - 1) + '.csv',
                sep=',')
            # master_table = pd.read_csv('./generations/master_results_table' + '.csv',sep=',')

            new_scores = pd.read_csv('./generations/results/results_gen' + str(self.GLOBAL_GENERATION - 1) + '.csv',
                                     sep=',')

            new_scores = new_scores.groupby("Ligand").min()["Binding Affinity"].reset_index()
            new_scores['id'] = new_scores['Ligand'].str.split("_").str[1].str.split("gen").str[0].str.split("id").str[1]
            new_scores['gen'] = new_scores['Ligand'].str.split("_").str[1].str.split("gen").str[1]
            new_scores['score'] = new_scores["Binding Affinity"]
            new_scores = new_scores[['id', 'gen', 'score']]
            print(new_scores.head())

            new_scores.id = new_scores.id.astype(str)
            new_scores.gen = new_scores.gen.astype(int)
            master_table.id = master_table.id.astype(str)
            master_table.gen = master_table.gen.astype(int)
            new_table = pd.merge(master_table, new_scores, on=['id', 'gen'], suffixes=('_old', '_new'), how='left')
            new_table['score'] = np.where(new_table['score_new'].isnull(), new_table['score_old'],
                                          new_table['score_new'])
            new_table = new_table.drop(['score_old', 'score_new'], axis=1)
            new_table['weight'] = new_table['smile'].apply(lambda x: Chem.Descriptors.MolWt(Chem.MolFromSmiles(x)))
            new_table = new_table.sort_values('score', ascending=True)
            print(new_table.head())

            new_table.to_csv(r'./generations/master_results_table_gen' + str(self.GLOBAL_GENERATION - 1) + '.csv',
                             index=False)

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
            similarity_adjusted.head()

            # Select top X ranked by similarity adjusted score for training data to refine the molecule generator RNN (ensures diverity)
            training_smiles += list(similarity_adjusted.head(5)['smile'])
            len(training_smiles)

            weight_adjusted = new_table.copy(deep=True)
            weight_adjusted['weight_adj_score'] = weight_adjusted.apply(GenProcess.calc_weight_score, axis=1)
            weight_adjusted = weight_adjusted.sort_values('weight_adj_score', ascending=True)
            weight_adjusted.head()

            # Select top X ranked by similarity adjusted score for training data to refine the molecule generator RNN (ensures diverity)
            training_smiles += list(weight_adjusted.head(5)['smile'])
            len(training_smiles)

            # Generate some with the base original model
            CONFIG_FILE = 'experiments/LSTM_Chem/config.json'
            config = process_config(CONFIG_FILE)
            modeler = LSTMChem(config, session='generate')
            generator = LSTMChemGenerator(modeler)

            sample_number = 20

            base_generated = generator.sample(num=sample_number)

            base_generated_mols = GenProcess.validate_mols(base_generated)
            base_generated_smiles = GenProcess.convert_mols_to_smiles(base_generated_mols)
            random.shuffle(base_generated_smiles)
            random.shuffle(base_generated_smiles)
            # Select X for training data to refine the molecule generator RNN (ensures diverity)
            training_smiles += base_generated_smiles[0:5]
            len(training_smiles)

            master_table = pd.read_csv(
                './generations/master_results_table_gen' + str(self.GLOBAL_GENERATION - 1) + '.csv',
                sep=',')
            master_table.head()
            # Save the list of smiles to train on
            with open('./generations/training/gen' + str(self.GLOBAL_GENERATION) + '_training.smi', 'w') as f:
                for item in training_smiles:
                    f.write("%s\n" % item)

            # ## Retrain the network to create molecules more like those selected above

        def proc2():
            from lstm_chem.finetuner import LSTMChemFinetuner

            config = process_config('experiments/LSTM_Chem/config.json')
            if self.GLOBAL_GENERATION == 1:
                config[
                    'model_weight_filename'] = 'experiments/LSTM_Chem/checkpoints/' + 'LSTM_Chem-baseline-model-full.hdf5'

            else:
                config['model_weight_filename'] = 'experiments/LSTM_Chem/checkpoints/finetuned_gen' + str(
                    self.GLOBAL_GENERATION - 1) + '.hdf5'
            config['finetune_data_filename'] = './generations/training/gen' + str(
                self.GLOBAL_GENERATION) + '_training.smi'
            # print(config)

            modeler = LSTMChem(config, session='finetune')
            finetune_dl = DataLoader(config, data_type='finetune')

            finetuner = LSTMChemFinetuner(modeler, finetune_dl)
            finetuner.finetune()

            finetuner.model.save_weights(
                'experiments/LSTM_Chem/checkpoints/finetuned_gen' + str(self.GLOBAL_GENERATION) + '.hdf5')

        def proc3():
            config = process_config('experiments/LSTM_Chem/config.json')
            config['model_weight_filename'] = 'experiments/LSTM_Chem/checkpoints/finetuned_gen' + str(
                self.GLOBAL_GENERATION) + '.hdf5'
            modeler = LSTMChem(config, session='generate')
            generator = LSTMChemGenerator(modeler)
            # print(config)

            sample_number = self.sample_number
            sampled_smiles = generator.sample(num=sample_number)

            valid_mols = []
            for smi in sampled_smiles:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    valid_mols.append(mol)
            # low validity
            print('Validity: ', f'{len(valid_mols) / sample_number:.2%}')

            valid_smiles = [Chem.MolToSmiles(mol) for mol in valid_mols]
            # high uniqueness
            print('Uniqueness: ', f'{len(set(valid_smiles)) / len(valid_smiles):.2%}')

            # Of valid smiles generated, how many are truly original vs ocurring in the training data

            training_data = pd.read_csv('./datasets/all_smiles_clean.smi', header=None)
            training_set = set(list(training_data[0]))
            original = []
            for smile in list(set(valid_smiles)):
                if not smile in training_set:
                    original.append(smile)
            print('Originality: ', f'{len(set(original)) / len(set(valid_smiles)):.2%}')

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

            # take the valid smiles from above and run them through process to add to tracking table and to generate next PyRx testing data
            mols_for_next_generation = GenProcess.validate_mols(valid_smiles)

            master_table = pd.read_csv(
                './generations/master_results_table_gen' + str(self.GLOBAL_GENERATION - 1) + '.csv',
                sep=',')
            new_mols_to_test = GenProcess.append_to_tracking_table(master_table, mols_for_next_generation, 'generated',
                                                                   self.GLOBAL_GENERATION)
            mols_for_pd = new_mols_to_test[0]
            mols_for_export = new_mols_to_test[1]

            master_table = master_table._append(mols_for_pd)
            master_table = master_table.reset_index(drop=True)
            master_table.to_csv(r'./generations/master_results_table_gen' + str(self.GLOBAL_GENERATION) + '.csv',
                                index=False)

            len(mols_for_export)

            GenProcess.write_gen_to_sdf(mols_for_export, self.GLOBAL_GENERATION, self.desired_length)
            print('Writing generated smiles to '+str(self.GLOBAL_GENERATION))
            print('Done!')

        proc1()
        proc2()
        proc3()


class NDWorker():
    GLOBAL_GENERATION = int

    def NDW(self):
        df = pd.read_csv('checklist.csv')
        vina_address = str(df['Address'].loc[df['Product'].str.contains('Vina', case=False)].values[0])
        mgl_address = str(df['Address'].loc[df['Product'].str.contains('MGL', case=False)].values[0])
        MGL_ROOT_PATH = mgl_address  # enter root of mgltools installed on your device
        PYTHONSH = os.path.join(MGL_ROOT_PATH, 'bin/pythonsh')
        LIGAND_PREPARE = os.path.join(MGL_ROOT_PATH, r'MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py')
        VINA_PATH = vina_address

        # print(os.getcwd())
        workingdir = os.getcwd()
        print(workingdir)
        gendir = workingdir + fr'/generations/gen{self.GLOBAL_GENERATION}'
        workingdir = workingdir + r'/babeltest'
        try:
            shutil.rmtree(workingdir)
        except OSError:
            print("Deletion of the directory %s failed" % workingdir)
        # else:
        #     print("Successfully deleted the directory %s" % workingdir)
        try:
            os.mkdir(workingdir)
        except OSError:
            print("Creation of the directory %s failed" % workingdir)
        # else:
        #     print("Successfully created the directory %s " % workingdir)
        # print(gendir)
        # print(workingdir)
        CONFIG_PATH = os.getcwd() + r'/receptor_conf.txt'

        def smi_to_pdb(smiles: str, output_name='output'):  # define output name without file extension

            command = f'(cd {workingdir} && obabel -isdf {smiles}.sdf ' \
                      f'-opdb -O {output_name}.pdb -h -m)'
            os.system(command)

        def prepare(pdb_path, pythonsh=PYTHONSH, mglprepare=LIGAND_PREPARE):
            counter = len(glob.glob1(workingdir, "*.pdb"))
            # print(counter)
            for i in range(counter):
                command = f'(cd "{workingdir}" && ' \
                          f'"{pythonsh}" "{mglprepare}" -l {pdb_path}{i + 1}.pdb -o {pdb_path}{i + 1}.pdbqt)'
                os.system(command)

        def dock(smiles):
            counter = len(glob.glob1(workingdir, "*.pdbqt"))
            print('Total number of smiles to dock: ' + str(counter))
            for i in range(counter):
                command = fr'(cd {workingdir} && "{VINA_PATH}/vina" --ligand {smiles}{i + 1}.pdbqt ' \
                          f'--config {CONFIG_PATH} --log {smiles}{i + 1}.log)'
                os.system(command)

        def addnames():
            
            suppl = Chem.SDMolSupplier(gendir + '.sdf')
            print(type(suppl))
            print(suppl)
            # df = pd.DataFrame(columns=['mol_name'])
            df2 = pd.DataFrame(columns=['Ligand', 'Binding Affinity'])
            counter = len(glob.glob1(workingdir, "*.log"))
            if counter == len(suppl):
                for i in range(len(suppl)):
                    mol = suppl[i].GetProp("Title")
                    filename = fr'{workingdir}/mysm{i + 1}.log'
                    with open(filename) as myFile:
                        for num, line in enumerate(myFile, 1):

                            if '  1  ' in line:
                                # print('found at line:', num, line)
                                k = line[line.find('-'):]
                                k = k[:k.find(' ')]
                                break

                    df2.loc[i] = ['6lu7cov_' + mol, k]
            df2.to_csv(r'all_smiles.csv', index=False)
            df2.to_csv(r'generations/results/results_gen' + str(self.GLOBAL_GENERATION) + '.csv', index=False)
            return df2

        smi_to_pdb(gendir, 'mysm')
        prepare('mysm')
        dock('mysm')
        addnames()


class MainWindow():

    def __init__(self):
        super(MainWindow, self).__init__()

    def ds(self):
        fn2 = None
        if self.cb1.isChecked():

            if self.cb2.isChecked() and self.fileName2.text() != '':
                fn2 = self.fileName2.text()
            else:
                print('No file selected for Dataset 2')

            if self.fileName1.text() != '':
                fn1 = self.fileName1.text()

            else:
                print('No file selected')
                return
        else:
            print("Dataset 1 not selected")
            return

        self.worker = DSWorker()
        self.worker.FN1 = fn1
        self.worker.FN2 = fn2
        self.worker.DSW

    def InitialNetworks(self):
        self.worker = INWorker()
        self.worker.IN

    def InitialGenerate(self):
        
        if self.init_sample_number.text() != '':
            sample_number = int(self.init_sample_number.text())
        else:
            print("Please determine sample number")
            return
        # print(self.init_desired_num.text())
        if self.init_desired_num.text() != '':
            desired_length = int(self.init_desired_num.text())
        else:
            print("Please determine sample number")
            return
        # load worker
        # self.thread = QThread()

        self.worker = IGWorker()
        self.worker.sample_number = sample_number
        self.worker.desired_length = desired_length
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.IGW)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        # self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

        self.init_dock.setEnabled(True)

    def InitialDocking(self):
        # load worker
        # self.thread = QThread()
        if self.thread.isRunning():
            print("Another process is already running.")
            return
        self.worker = IDWorker()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.IDW)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        # self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def Evaluation(self):
        if self.init_sample_number.text() != '':
            sample_number = int(self.init_sample_number.text())
        else:
            print("Please determine sample number")
            return
        # print(self.init_desired_num.text())
        if self.init_desired_num.text() != '':
            desired_length = int(self.init_desired_num.text())
        else:
            print("Please determine sample number")
            return
        GLOBAL_GENERATION = int(self.gennum.text()) + 1
        self.gennum.setValue(GLOBAL_GENERATION)

        # load worker
        # self.thread = QThread()
        if self.thread.isRunning():
            print("Another process is already running.")
            return
            # self.thread.exit()
        self.worker = NGWorker()
        self.worker.sample_number = sample_number
        self.worker.desired_length = desired_length
        self.worker.GLOBAL_GENERATION = GLOBAL_GENERATION
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.NGW)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        # self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def NextDocking(self):

        GLOBAL_GENERATION = int(self.gennum.text())
        # self.thread = QThread()
        self.worker = NDWorker()
        self.worker.GLOBAL_GENERATION = GLOBAL_GENERATION
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.NDW)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        # self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    widget = QtWidgets.QStackedWidget()
    widget.addWidget(mainWindow)
    widget.setFixedWidth(400)
    widget.setFixedHeight(600)
    widget.show()
    sys.exit(app.exec_())
