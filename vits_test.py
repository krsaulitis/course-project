import torch
import scipy
from torchinfo import summary
from transformers import VitsTokenizer, VitsModel, set_seed
from playsound import playsound

tokenizer = VitsTokenizer.from_pretrained('facebook/mms-tts-eng')
model = VitsModel.from_pretrained('facebook/mms-tts-eng')

inputs = tokenizer(text='This is a test for VITS model.', return_tensors='pt')

summary(model, mode='train', depth=4, input_data={'input_ids': inputs.data['input_ids'], 'attention_mask': inputs.data['attention_mask']})

set_seed(42)  # make deterministic

with torch.no_grad():
   outputs = model(**inputs)

waveform = outputs.waveform[0]
scipy.io.wavfile.write('audios/vits_test.wav', rate=model.config.sampling_rate, data=waveform.numpy())

playsound('audios/vits_test.wav')

