from tqdm import tqdm
import numpy as np
from rest.lstm_chem.utils.smiles_tokenizer2 import SmilesTokenizer
from celery_progress.backend import ProgressRecorder
from gui.models import GenerateLog

class LSTMChemGenerator(object):
    def __init__(self, modeler):
        self.session = modeler.session
        self.model = modeler.model
        self.config = modeler.config
        self.st = SmilesTokenizer()

    def _generate(self, sequence):
        while (sequence[-1] != 'E') and (len(self.st.tokenize(sequence)) <=
                                         self.config.smiles_max_length):
            x = self.st.one_hot_encode(self.st.tokenize(sequence))
            preds = self.model.predict_on_batch(x)[0][-1]
            next_idx = self.sample_with_temp(preds)
            sequence += self.st.table[next_idx]

        sequence = sequence[1:].rstrip('E')
        return sequence

    def sample_with_temp(self, preds):
        streched = np.log(preds) / self.config.sampling_temp
        streched_probs = np.exp(streched) / np.sum(np.exp(streched))
        return np.random.choice(range(len(streched)), p=streched_probs)

    def sample(self, progress_recorder, task_id, num=1, start='G'):
        sampled = []
        if self.session == 'generate':
            for i in tqdm(range(num)):
                gl = GenerateLog.objects.get(task_id = task_id)
                if gl.task_status == 'F':
                    return sampled
                progress_recorder.set_progress(i + 1, num)
                sampled.append(self._generate(start))
            return sampled
        else:
            from rdkit import Chem, RDLogger
            RDLogger.DisableLog('rdApp.*')
            while len(sampled) < num:
                sequence = self._generate(start)
                mol = Chem.MolFromSmiles(sequence)
                if mol is not None:
                    canon_smiles = Chem.MolToSmiles(mol)
                    sampled.append(canon_smiles)
            return sampled
