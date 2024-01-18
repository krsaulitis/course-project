import csv
import random
import time
import requests
import uuid
import os
import requests
import logging
import json
from dotenv import load_dotenv


class ClientWithLogger:
    def __init__(self):
        log_directory = './logs'
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)

        log_file_path = os.path.join(log_directory, 'request_logs.log')
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()])
        self.logger = logging.getLogger()

    def log_request_response(self, url, method='get', params=None, data=None, headers=None, files=None):
        try:
            self.logger.info(
                f"Request: Method={method.upper()}, URL={url}, Params={params}, Data={data}, Headers={headers}")

            if method.lower() == 'get':
                response = requests.get(url, params=params, headers=headers)
            elif method.lower() == 'post':
                response = requests.post(url, params=params, data=data, headers=headers, files=files)

            self.logger.info(f"Response: Status Code={response.status_code}, Content={response.text}")

            return response
        except Exception as e:
            self.logger.error(f"Error during request: {e}")


class AsyaTaskProcessor:
    def __init__(self, base_url, api_key, file_list, results_file, max_parallel_tasks, status_check_interval):
        self.base_url = base_url
        self.api_key = api_key
        self.queue_files = file_list
        self.results_file = results_file
        self.max_parallel_tasks = max_parallel_tasks
        self.status_check_interval = status_check_interval
        self.active_tasks = []
        self.completed_tasks = []
        self.client = ClientWithLogger()

    def task_status(self, task_id):
        task_response = self.client.log_request_response(
            url=f"{self.base_url}/task_status",
            params={
                "api_key": self.api_key,
                "task_uuid": task_id,
            },
            method='post',
        )

        return task_response.json()

        # return {
        #     'request_status': random.choice(['COMPLETED', 'PROCESSING']),
        #     'results': {
        #         'segments': [
        #             {
        #                 'text': 'Hello world',
        #             },
        #         ],
        #     },
        # }

    def task_submit(self, file_path):
        with open(file_path, 'rb') as file:
            task_response = self.client.log_request_response(
                url=f"{self.base_url}/task_submit",
                params={
                    "api_key": self.api_key,
                    "features": [
                        "audio_denoise",
                        "audio_diarisation",
                        "audio_text",
                        "text_grammar",  # should try with and without
                    ],
                    "known_and_unknown_users_count": 1,
                    "language_codes": ["en"],
                },
                files={'file': file},
                method='post',
            )

            return task_response.json()

            # return {
            #     'is_success': True,
            #     'task_uuid': uuid.uuid4(),
            # }

    def write_to_csv(self, task_id, file_path, status, text):
        updated = False
        data = []

        file_number = os.path.basename(file_path).replace('.wav', '')

        if os.path.exists(self.results_file):
            with open(self.results_file, 'r', newline='') as file:
                reader = csv.reader(file)
                for row in reader:
                    existing_file_number = os.path.basename(row[1]).replace('.wav', '')
                    if row and existing_file_number == file_number:
                        data.append([task_id, file_path, status, text])
                        updated = True
                    else:
                        data.append(row)

        if not updated:
            data.append([task_id, file_path, status, text])

        with open(self.results_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)

    def _init_tasks(self):
        for _ in range(min(self.max_parallel_tasks - len(self.active_tasks), len(self.queue_files))):
            file_path = self.queue_files.pop(0)
            print(f'Submitting task {_} (file: {file_path})')
            try:
                response = self.task_submit(file_path)
                is_success = response['is_success']
                task_id = response['task_uuid']
            except:
                is_success = False

            if is_success:
                print(f'Submitted task {task_id} (file: {file_path})')
                self.active_tasks.append({
                    'task_id': task_id,
                    'file_path': file_path,
                })

    def merge_segment_texts(self, segments):
        return ' '.join(segment["text"] for segment in segments if segment["text"])

    def process_files(self):
        self._init_tasks()

        while self.active_tasks or self.queue_files:
            just_completed_tasks = []

            for task in self.active_tasks:
                try:
                    response = self.task_status(task['task_id'])
                    status = response['request_status']
                    error = response['error_code']
                    if error != 0:
                        status = 'ERROR'
                except:
                    status = 'ERROR'

                if status == 'READY':
                    self.completed_tasks.append(task)
                    just_completed_tasks.append(task)
                    print(f'Completed task {task["task_id"]} (file: {task["file_path"]}) with status {status}')
                    self.write_to_csv(
                        task['task_id'],
                        task['file_path'],
                        status,
                        self.merge_segment_texts(response['results']['segments']),
                    )

                if status == 'ERROR':
                    self.completed_tasks.append(task)
                    just_completed_tasks.append(task)
                    print(f'Completed task {task["task_id"]} (file: {task["file_path"]}) with status {status}')
                    self.write_to_csv(
                        task['task_id'],
                        task['file_path'],
                        status,
                        'Err',
                    )

            for task in just_completed_tasks:
                self.active_tasks.remove(task)

            self._init_tasks()
            time.sleep(self.status_check_interval)


load_dotenv()


def get_files_to_process(csv_path, audio_directory):
    processed_files = set()
    error_files = set()

    # Read the CSV file
    with open(csv_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            file_id, file_path, status, _ = row
            if status == 'ERROR':
                error_files.add(file_path.split('/')[-1])
            processed_files.add(file_path.split('/')[-1])

    # List all .wav files in the directory
    all_files = set(f for f in os.listdir(audio_directory) if f.endswith('.wav'))

    # Files to process are those which are either not processed or have an ERROR status
    return [f'{audio_directory}/{file}' for file in all_files if file not in processed_files or file in error_files]


audio_dir = '../../mqtts/audios_alt'
# wav_files = [f'{audio_dir}/{file}' for file in os.listdir(audio_dir) if file.endswith('.wav')]
wav_files = get_files_to_process('./results.csv', audio_dir)
wav_files.sort()

processor = AsyaTaskProcessor(
    os.getenv('ASYA_API_BASE_URL'),
    os.getenv('ASYA_API_KEY'),
    wav_files,
    './results.csv',
    5,
    20,
)

processor.process_files()
