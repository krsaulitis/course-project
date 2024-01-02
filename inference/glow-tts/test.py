# todo: create a wrapper for ./model/test.py
import sys

sys.path.append('./model')
from inference import GlowTTS


class GlowTTSGenerator:
    def __init__(self):
        self.model = GlowTTS('./pretrained.pth', './model/waveglow/waveglow_256channels_ljs_v3.pt')

    def generate(self, text, path):
        self.model.generate(text, path)


model = GlowTTSGenerator()
model.generate("Hello world!", "test.wav")