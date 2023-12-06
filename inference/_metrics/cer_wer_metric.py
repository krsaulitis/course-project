import csv
from jiwer import wer, cer
import numpy as np


def get_file_truth(column_index, criteria):
    with open('cer_wer/labels.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[column_index] == criteria:
                return row
    return None


def calculate_cer_wer(csv_file):
    modified_rows = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)

        cer_scores = []
        wer_scores = []

        for row in reader:
            prediction = row[3]
            truth = get_file_truth(0, row[1])[1]

            row_cer = cer(prediction, truth)
            row_wer = wer(prediction, truth)

            cer_scores.append(row_cer)
            wer_scores.append(row_wer)

            row.append(row_cer)
            row.append(row_wer)
            modified_rows.append(row)

        avg_cer = np.mean(cer_scores)
        avg_wer = np.mean(wer_scores)

        print(f'Average CER: {avg_cer}')
        print(f'Average WER: {avg_wer}')

    with open(csv_file, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(modified_rows)


calculate_cer_wer('cer_wer/results.csv')
