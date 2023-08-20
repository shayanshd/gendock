import random
from rdkit import Chem, DataStructs
import pandas as pd
import numpy as np

class GenProcess:

    def validate_mols(list_of_smiles):
        valid_mols = []
        for smi in list_of_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_mols.append(mol)
        return valid_mols

    def convert_mols_to_smiles(list_of_mols):
        valid_smiles = [Chem.MolToSmiles(mol) for mol in list_of_mols]
        return valid_smiles

    def initialize_generation_from_mols(list_of_mols, desired_length):
        # assert desired_length > 30
        random.shuffle(list_of_mols)
        random.shuffle(list_of_mols)

        # Prepare fingerprints for similarity calcs
        mol_fingerprints = []
        for mol in list_of_mols:
            mol_fingerprints.append(Chem.RDKFingerprint(mol))

        selected_mols = list_of_mols[0:5]
        selected_fingerprints = mol_fingerprints[0:5]
        remaining_mols = list_of_mols[5:]
        remaining_fingerprints = mol_fingerprints[5:]

        similarity_threshold = .05
        while len(selected_mols) < desired_length:
            for fingerprint, mol in zip(remaining_fingerprints, remaining_mols):
                max_similarity = np.max(DataStructs.BulkTanimotoSimilarity(fingerprint, selected_fingerprints))
                if (max_similarity <= similarity_threshold) and (max_similarity < 1):
                    selected_fingerprints.append(fingerprint)
                    selected_mols.append(mol)
            # print("Completed loop with threshold at: ", similarity_threshold, ". Length is currently: ",
            #       len(selected_mols))
            similarity_threshold += .05
        return selected_mols

    def iterate_alpha(alpha_code):
        numbers = []
        for letter in alpha_code:
            number = ord(letter)
            numbers.append(number)

        if numbers[3] + 1 > 90:
            if numbers[2] + 1 > 90:
                if numbers[1] + 1 > 90:
                    if numbers[0] + 1 > 90:
                        raise ValueError('Too long for alpha code')
                    else:
                        numbers[3] = 65
                        numbers[2] = 65
                        numbers[1] = 65
                        numbers[0] = numbers[0] + 1
                else:
                    numbers[3] = 65
                    numbers[2] = 65
                    numbers[1] = numbers[1] + 1
            else:
                numbers[3] = 65
                numbers[2] = numbers[2] + 1
        else:
            numbers[3] = numbers[3] + 1

        new_code = ""
        for number in numbers:
            new_code += chr(number)
        return new_code

    def append_to_tracking_table(master_table, mols_to_append, source, generation):
        # Assign IDs for tracking to each mol, and assign a pandas table entry for each
        mols_to_export = []
        rows_list = []

        master_table_gen = master_table[master_table['gen'] == generation]
        if master_table_gen.shape[0] == 0:
            id_code = 'AAAA'
        else:
            master_table_gen_ids = master_table_gen.sort_values('id', ascending=True)
            master_table_gen_max_id = master_table_gen_ids.tail(1)
            key = master_table_gen_max_id['id'].keys()[0]
            id_code = GenProcess.iterate_alpha(str(master_table_gen_max_id['id'][key]))

        training_data = pd.read_csv('./datasets/all_smiles_clean.smi', header=None)
        training_set = set(list(training_data[0]))

        for mol in mols_to_append:
            pm = Chem.PropertyMol.PropertyMol(mol)
            title = 'id' + str(id_code) + 'gen' + str(generation)
            # print(title)
            # Enables for tracking which molecule is which in PyRx GUI and PyRx results export
            pm.SetProp('Title', title)
            mols_to_export.append(pm)

            # And track in pandas
            mol_dict = {}
            mol_dict['id'] = id_code
            mol_dict['gen'] = generation
            smile = Chem.MolToSmiles(mol)
            assert type(smile) == type('string')
            mol_dict['smile'] = smile

            if (source != 'hiv' and source != 'manual' and source != 'baseline') and (smile in training_set):
                mol_dict['source'] = 'training'
            else:
                mol_dict['source'] = source
            mol_dict['score'] = 99.9

            rows_list.append(mol_dict)
            id_code = GenProcess.iterate_alpha(id_code)

        df = pd.DataFrame(rows_list)
        return df, mols_to_export

    def write_gen_to_sdf(mols_for_export, generation, batch_size):
        if len(mols_for_export) > batch_size:
            batches = (len(mols_for_export) // 1000) + 1
            for i in range(0, batches):
                batch_to_export = mols_for_export[i * batch_size:(i + 1) * batch_size]
                w = Chem.SDWriter('./generations/gen' + str(generation) + '_batch_' + str(i + 1) + '.sdf')
                for m in batch_to_export: w.write(m)
        else:
            w = Chem.SDWriter('./generations/gen' + str(generation) + '.sdf')
            for m in mols_for_export:
                w.write(m)

        # w = Chem.SDWriter('./generations/junk/test.sdf')
        # w.write(m)

        return mols_for_export

    def calc_weight_score(row):
        adj_factor = (500 / row['weight']) ** .333
        if adj_factor < 1:
            adj_score = 0
        else:
            adj_score = row['score'] * adj_factor
        return adj_score