from tqdm.auto import tqdm
from TTS.api import TTS


print(TTS().list_models())  # not available here
exit(0)


class FastSpeech2Generator:
    def __init__(self):
        self.tts = TTS("tts_models/multilingual/multi-dataset/your_tts")

    def generate(self, text, path):
        self.tts.tts_to_file(text, speaker="male-en-2", language="en", file_path=path)


generator = FastSpeech2Generator()


def generate():
    with open('../_data/filtered_no_duplicates_max_upvotes.tsv', 'r') as file:
        rows = file.readlines()

        for i, row in tqdm(enumerate(rows)):
            sentence = row.split('\t')[2]
            file_id = row.split('\t')[1].split("_")[-1].split(".")[0]
            path = f'./audios/audio_{file_id}.wav'
            generator.generate(sentence, path)


generate()
