import torch
import scipy
from transformers import VitsTokenizer, VitsModel, set_seed

# NOTE: this is not the OG implementation. This is FB massively multilingual speech project (single speaker)
tokenizer = VitsTokenizer.from_pretrained('facebook/mms-tts-eng')
model = VitsModel.from_pretrained('facebook/mms-tts-eng')

with open('../sentences.txt', 'r') as f:
    sentences = f.readlines()

    for i, sentence in enumerate(sentences):
        inputs = tokenizer(text=sentence, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)

        waveform = outputs.waveform[0]
        # Normalize to int16 for 16bit audio file
        waveform = (waveform / torch.max(torch.abs(waveform)) * 32767).short()
        scipy.io.wavfile.write(f'./samples/sample_{i}.wav', rate=model.config.sampling_rate, data=waveform.numpy())
