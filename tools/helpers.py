import re
import csv
import os
import json
import soundfile as sf
import numpy as np
import wavfile
import librosa
import pandas as pd
from pydub import AudioSegment
from tqdm.auto import tqdm


def filter_common_voice():
    df = pd.read_csv('../datasets/common_voice/validated.tsv', sep='\t')

    def filter_rows(row):
        up_votes = row['up_votes']
        down_votes = row['down_votes']
        segment = row['segment']

        return up_votes >= 7 and ((up_votes + down_votes) / up_votes) <= 1.1 and segment != 'Benchmark'

    filtered_df = df[df.apply(filter_rows, axis=1)]

    idx = filtered_df.groupby('sentence')['up_votes'].idxmax()
    filtered_df = filtered_df.loc[idx]

    filtered_df.to_csv('../inference/_data/filtered_no_duplicates_max_upvotes.tsv', sep='\t', index=False)


def delete_duplicates():
    tsv_file_path = './filtered_no_duplicates_max_upvotes.tsv'
    directory_path = '../../datasets/common_voice/clips'

    files_to_keep = set()
    with open(tsv_file_path, 'r') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            # match = re.search(r'common_voice_en_(\d+).mp3', row['path'])
            # file_id = match.group(1)
            # file_name = f'audio_{file_id}.wav'

            files_to_keep.add(row['path'])

    deleted_files = []
    for file_name in os.listdir(directory_path):
        if file_name not in files_to_keep:
            os.remove(os.path.join(directory_path, file_name))
            deleted_files.append(file_name)


# Function to convert MP3 to WAV
def convert_mp3_to_wav():
    source_folder = "./clips"
    target_folder = "./audios_hd"

    # Create target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for file in os.listdir(source_folder):
        if file.endswith(".mp3"):
            mp3_sound = AudioSegment.from_mp3(os.path.join(source_folder, file))
            mp3_sound = mp3_sound.set_frame_rate(48000)
            wav_filename = file.replace(".mp3", ".wav")
            mp3_sound.export(os.path.join(target_folder, wav_filename), format="wav")
            print(f"Converted: {file} to {wav_filename}")


def convert_sample_rate():
    audio, sr = librosa.load('../inference/mqtts/model/lj_speech.wav', sr=22050)
    audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    sf.write('../inference/mqtts/model/lj_speech_16000.wav', audio_resampled, 16000)


def convert_sample_rate_for_folder():
    # Directory containing the audio files
    directory = "../inference/tacotron_2/audios"
    new_directory = "../inference/tacotron_2/audios_22050"

    # Target sample rate
    target_sample_rate = 22050

    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory, filename)

            # Load the audio file
            audio = AudioSegment.from_file(file_path)

            # Change the sample rate
            audio = audio.set_frame_rate(target_sample_rate)

            new_filepath = os.path.join(new_directory, filename)

            # Export the modified file
            audio.export(new_filepath, format="wav")  # Change format if needed

            print(f"Processed {filename}")

    print("All files have been processed.")


# Helper function to extract responses from request/response logs file for ground truth audios
def extract_responses_from_logs():
    log_file_path = '../inference/_metrics/cer_wer/logs/request_gt_logs.log'

    # Regular expression to find request and response lines with JSON content
    request_pattern = re.compile(r"Request: .* 'task_uuid': '([^']*)'")
    response_pattern = re.compile(r"Response: Status Code=\d{3}, Content=(\{.*\})")

    ready_responses = []
    last_task_uuid = None

    with open(log_file_path, 'r') as file:
        for line in file:
            request_match = request_pattern.search(line)
            if request_match:
                last_task_uuid = request_match.group(1)

            response_match = response_pattern.search(line)
            if response_match:
                response_json_str = response_match.group(1)
                try:
                    response_data = json.loads(response_json_str)
                    if response_data.get("request_status") == "READY":
                        segments = response_json_str['results']['segments']
                        ready_responses.append([
                            last_task_uuid,
                            ' '.join(segment["text"] for segment in segments if segment["text"]),
                        ])
                except json.JSONDecodeError:
                    print("Error decoding JSON from response")

    with open('./results.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerows(ready_responses)


def merge_two_csv_files():
    path1 = '../inference/_metrics/cer_wer/results.csv'
    df1 = pd.read_csv(path1, header=None, names=['ID', 'AudioPath', 'Status', 'Text1'])

    path2 = './results.csv'  # Replace with the path to your second CSV file
    df2 = pd.read_csv(path2, header=None, names=['ID', 'Text2'])

    merged_df = pd.merge(df1, df2, on='ID')
    merged_df = merged_df[['ID', 'AudioPath', 'Status', 'Text2']]

    merged_df.to_csv('merged_results.csv', index=False)


convert_sample_rate_for_folder()
