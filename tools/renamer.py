import os
import csv


class DuplicateSentenceException(Exception):
    pass


def read_tsv_to_dict(file_path, file_id_index, sentence_index):
    mapping = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)  # Skip header
        for row in reader:
            file_id = row[file_id_index].split('.')[0].split('_')[-1]
            sentence = row[sentence_index]
            if sentence in mapping:
                raise DuplicateSentenceException(f"Duplicate sentence found in file {file_path}: '{sentence}'")
            mapping[sentence] = file_id
    return mapping


def rename_files_in_directory(directory, file_mapping):
    for file in os.listdir(directory):
        if file.endswith('.wav'):
            file_id = file.split('_')[-1].split('.')[0]
            if file_id in file_mapping:
                new_file_id = file_mapping[file_id]
                new_file_name = f'audio_{new_file_id}.wav'
                os.rename(os.path.join(directory, file), os.path.join(directory, new_file_name))


def update_file_names_in_csv(csv_file_path, file_mapping):
    updated_data = []
    with open(csv_file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            file_id = row[1].split('/')[-1].split('.')[0].split('_')[-1]
            if file_id in file_mapping:
                new_file_id = file_mapping[file_id]
                row[1] = row[1].replace(f'audio_{file_id}', f'audio_{new_file_id}')
            updated_data.append(row)
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(updated_data)


# File paths
no_duplicates_tsv = '../inference/_data/filtered_no_duplicates.tsv'
max_upvotes_tsv = '../inference/_data/filtered_no_duplicates_max_upvotes.tsv'
audio_directory = '../inference/mary_tts/audios/'
results_csv = '../inference/_metrics/cer_wer/results/results_mary_tts.csv'

# Create mappings
no_duplicates_map = read_tsv_to_dict(no_duplicates_tsv, 1, 2)
max_upvotes_map = read_tsv_to_dict(max_upvotes_tsv, 1, 2)

# Find file ID mapping
file_id_mapping = {no_duplicates_map[sentence]: max_upvotes_map[sentence] for sentence in no_duplicates_map if
                   sentence in max_upvotes_map}

file_id_mapping_diff = {}
for sentence in no_duplicates_map:
    if sentence in max_upvotes_map:
        source_id = no_duplicates_map[sentence]
        target_id = max_upvotes_map[sentence]
        if source_id != target_id:
            file_id_mapping_diff[source_id] = target_id

print(file_id_mapping_diff)

# Rename files in the directory
# rename_files_in_directory(audio_directory, file_id_mapping)

# Update file names in the CSV file
# update_file_names_in_csv(results_csv, file_id_mapping)
