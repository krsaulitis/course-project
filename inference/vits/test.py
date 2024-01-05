import os
import sys
import torch
import scipy
import librosa
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.append('./model')
import utils as utils
import commons as commons
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

# from model import utils as utils
# from model import commons as commons
# from model.models import SynthesizerTrn
# from model.text.symbols import symbols
# from model.text import text_to_sequence

os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = '/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.dylib'


class VITSGenerator:
    def __init__(self, type='ljs'):
        self.type = type
        if type == 'ljs':
            self.hps = utils.get_hparams_from_file('./model/configs/ljs_base.json')
        else:
            self.hps = utils.get_hparams_from_file('./model/configs/vctk_base.json')

        self.model = SynthesizerTrn(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model
        )

        self.model.eval()

        if type == 'ljs':
            utils.load_checkpoint('./checkpoints/pretrained_ljs.pth', self.model, None)
        else:
            utils.load_checkpoint('./checkpoints/pretrained_vctk.pth', self.model, None)

    def generate(self, text, path):
        with torch.no_grad():
            tokens = self.get_text(text)
            t_in = tokens.unsqueeze(0)
            t_in_len = torch.LongTensor([tokens.size(0)])
            sid = torch.LongTensor([4])
            audio = self.model.infer(
                t_in,
                t_in_len,
                sid=None if self.type == 'ljs' else sid,
                noise_scale=.667,
                noise_scale_w=0.8,
                length_scale=1
            )[0][0, 0]

            # Normalize to int16 for 16bit audio file
            resampled = librosa.resample(audio.numpy(), orig_sr=self.hps.data.sampling_rate, target_sr=16000)
            waveform = (resampled / np.max(np.abs(resampled)) * 32767).astype(np.int16)
            scipy.io.wavfile.write(path, rate=16000, data=waveform)

    def get_text(self, text):
        text_norm = text_to_sequence(text, self.hps.data.text_cleaners)
        if self.hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm


generator = VITSGenerator()
# generator.forward()


def generate():
    with open('../_data/filtered_no_duplicates_max_upvotes.tsv', 'r') as file:
        rows = file.readlines()

        for i, row in tqdm(enumerate(rows), total=len(rows)):
            sentence = row.split('\t')[2]
            file_id = row.split('\t')[1].split("_")[-1].split(".")[0]
            path = f'./audios_ljs_real/audio_{file_id}.wav'
            generator.generate(sentence, path)


generate()
