import csv
import re
from jiwer import wer, cer
import numpy as np


def get_file_truth(column_index, criteria):
    # with open('../_data/filtered_no_duplicates.tsv', 'r', encoding='utf-8') as file:  # for MaryTTS
    with open('../_data/filtered_no_duplicates_max_upvotes.tsv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            if row[column_index] == criteria:
                return row
    return None


def normalize_text(text):
    # remove non-alphabet characters
    normalized_text = ''.join([char if char.isalpha() else ' ' for char in text])
    # lowercase all letters
    normalized_text = normalized_text.lower()
    # remove multiple spaces
    while '  ' in normalized_text:
        normalized_text = normalized_text.replace('  ', ' ')

    return normalized_text


def calculate_cer_wer(csv_file):
    modified_rows = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)

        total_count = 0
        skip_count = 0
        cer_scores = []
        wer_scores = []

        for row in reader:
            try:
                total_count += 1
                prediction = normalize_text(row[3])

                match = re.search(r'audio_(\d+).wav', row[1])
                # match = re.search(r'common_voice_en_(\d+).wav', row[1])  # for Ground Truth
                file_id = match.group(1)
                cv_file_name = f'common_voice_en_{file_id}.mp3'

                truth_row = get_file_truth(1, cv_file_name)
                if not truth_row:
                    skip_count += 1
                    continue

                truth = normalize_text(truth_row[2])

                if prediction:
                    row_cer = cer(prediction, truth)
                    row_wer = wer(prediction, truth)
                elif len(truth.split()) > 1:
                    row_cer = 1
                    row_wer = 1
                    print(f'Empty prediction: {prediction}, truth: {truth} for file: {row[1]}')
                else:
                    skip_count += 1
                    continue

                # if row_cer > 0:
                #     # cer_count += 1
                #     print(f'Prediction: {prediction}')
                #     print(f'Truth:      {truth}')
                #     print(f'CER:        {row_cer}')

                cer_scores.append(row_cer)
                wer_scores.append(row_wer)

                row.append(row_cer)
                row.append(row_wer)
                modified_rows.append(row)
            except Exception as e:
                skip_count += 1
                print('Error on row: ', row[0])
                print(e)

        avg_cer = np.mean(cer_scores)
        avg_wer = np.mean(wer_scores)

        print(f'Skipped {skip_count}/{total_count}')
        print(f'Average CER: {avg_cer}')
        print(f'Average WER: {avg_wer}')

    with open('../fastspeech_2/cer_wer.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerows(modified_rows)


calculate_cer_wer('cer_wer/results/results_fast_speech.csv')
