import sys
import os
import torch
import scipy
import librosa
import numpy as np

os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = '/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.dylib'

sys.path.append('./model')
import utils as utils
import commons as commons
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


hps = utils.get_hparams_from_file('./model/configs/ljs_base.json')

model = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model
)

_ = model.eval()
_ = utils.load_checkpoint('./checkpoints/pretrained_ljs.pth', model, None)

with torch.no_grad():
    with open('../data/sentences.txt', 'r') as f:
        sentences = f.readlines()

        for i, sentence in enumerate(sentences):
            tokens = get_text(sentence, hps)
            t_in = tokens.unsqueeze(0)
            t_in_len = torch.LongTensor([tokens.size(0)])
            audio = model.infer(t_in, t_in_len, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0, 0]

            # Normalize to int16 for 16bit audio file
            resampled = librosa.resample(audio.numpy(), orig_sr=hps.data.sampling_rate, target_sr=16000)
            waveform = (resampled / np.max(np.abs(resampled)) * 32767).astype(np.int16)
            scipy.io.wavfile.write(f'./samples/sample_{i}.wav', rate=16000, data=waveform)
