import os
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import MolStandardize
from lstm_chem.utils.smiles_tokenizer import SmilesTokenizer

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


class CleanClass:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file

    def cleaner(self):
        assert os.path.exists(self.input_file)
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

        pp = Preprocessor()

        with open(self.input_file, 'r') as f:
            smiles = [l.rstrip() for l in f]

        print(f'input SMILES num: {len(smiles)}')
        print('starting to clean up')

        pp_smiles = [pp.process(smi) for smi in tqdm(smiles)]
        print('Step 1 / 3 completed')
        cl_smiles = list(set([s for s in pp_smiles if s]))
        print('Step 2 / 3 completed')

        # token limits (34 to 128)
        out_smiles = []
        print('Initiating tokenizer')
        st = SmilesTokenizer()
        print('Tokenizer initiated')
        out_smiles = cl_smiles

        print('done.')
        print(f'output SMILES num: {len(out_smiles)}')

        with open(self.output_file, 'w') as f:
            for smi in out_smiles:
                f.write(smi + '\n')

        return

