import sys
import torch
import scipy
import numpy as np
from tqdm.auto import tqdm
from TTS.api import TTS

class OverflowGenerator:
    def __init__(self):
        self.tts = TTS("tts_models/en/ljspeech/overflow", vocoder_path="vocoder_models/en/ljspeech/hifigan_v2")

    def generate(self, text, path):
        self.tts.tts_to_file(text, file_path=path)


generator = OverflowGenerator()

def generate():
    with open('../_data/filtered_no_duplicates_max_upvotes.tsv', 'r') as file:
        rows = file.readlines()

        for i, row in tqdm(enumerate(rows), total=len(rows)):
            sentence = row.split('\t')[2]
            file_id = row.split('\t')[1].split("_")[-1].split(".")[0]
            path = f'./audios/audio_{file_id}.wav'
            generator.generate(sentence, path)


generate()

