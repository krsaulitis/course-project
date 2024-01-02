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
        with open(self.results_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([task_id, file_path, status, text])

    def _init_tasks(self):
        for _ in range(min(self.max_parallel_tasks - len(self.active_tasks), len(self.queue_files))):
            file_path = self.queue_files.pop(0)
            print(f'Submitting task {_} (file: {file_path})')
            response = self.task_submit(file_path)
            is_success = response['is_success']
            task_id = response['task_uuid']

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

            for task in just_completed_tasks:
                self.active_tasks.remove(task)

            self._init_tasks()
            time.sleep(self.status_check_interval)


load_dotenv()

audio_dir = '../../your_tts/audios'
wav_files = [f'{audio_dir}/{file}' for file in os.listdir(audio_dir) if file.endswith('.wav')]
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
