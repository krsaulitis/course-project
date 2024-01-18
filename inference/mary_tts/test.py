import requests
from tqdm.auto import tqdm


def text_to_speech(text, output_path):
    response = requests.post('http://localhost:59125/process', params={
        "INPUT_TEXT": text,
        "INPUT_TYPE": "TEXT",
        "LOCALE": "en_US",
        "OUTPUT_TYPE": "AUDIO",
        "AUDIO": "WAVE",
    })

    with open(output_path, 'wb') as file:
        file.write(response.content)


def generate():
    with open('../_data/common_voice.tsv', 'r') as file:
        rows = file.readlines()

        for i, row in tqdm(enumerate(rows), total=len(rows)):
            sentence = row.split('\t')[2]
            file_id = row.split('\t')[1].split("_")[-1].split(".")[0]
            path = f'./audios/audio_{file_id}.wav'
            text_to_speech(sentence, path)


text_to_speech('data', 'data.wav')
