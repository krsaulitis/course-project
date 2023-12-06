import csv
import random
import time
import requests
import uuid
import os
from dotenv import load_dotenv


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

    def task_status(self, task_id):
        # task_response = requests.post(
        #     f"{self.base_url}/task_status",
        #     params={
        #         "api_key": self.api_key,
        #         "task_uuid": task_id,
        #     })

        return {
            'request_status': random.choice(['COMPLETED', 'PROCESSING']),
            'results': {
                'text': 'Hello world',
            }
        }

    def task_submit(self, file_path):
        with open(file_path, 'rb') as file:
            # task_response = requests.post(
            #     f"{self.base_url}/task_submit",
            #     params={
            #         "api_key": self.api_key,
            #         "features": [
            #             "audio_denoise",
            #             "audio_diarisation",
            #             "audio_text",
            #             "text_grammar",  # should try with and without
            #         ],
            #         "known_and_unknown_users_count": 1,
            #         "language_codes": ["en"],
            #     },
            #     files={'file': file})
            #
            # return task_response.json()

            return {
                'is_success': True,
                'task_uuid': uuid.uuid4(),
            }

    def write_to_csv(self, task_id, file_path, status, text):
        with open(self.results_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([task_id, file_path, status, text])

    def _init_tasks(self):
        for _ in range(min(self.max_parallel_tasks - len(self.active_tasks), len(self.queue_files))):
            file_path = self.queue_files.pop(0)
            response = self.task_submit(file_path)
            is_success = response['is_success']
            task_id = response['task_uuid']

            if is_success:
                self.active_tasks.append({
                    'task_id': task_id,
                    'file_path': file_path,
                })

    def process_files(self):
        self._init_tasks()

        while self.active_tasks or self.queue_files:
            just_completed_tasks = []

            for task in self.active_tasks:
                response = self.task_status(task['task_id'])
                status = response['request_status']

                if status != 'PROCESSING':
                    self.completed_tasks.append(task)
                    just_completed_tasks.append(task)
                    self.write_to_csv(task['task_id'], task['file_path'], status, response['results']['text'])

            for task in just_completed_tasks:
                self.active_tasks.remove(task)

            self._init_tasks()
            time.sleep(self.status_check_interval)


load_dotenv()
processor = AsyaTaskProcessor(
    os.getenv('ASYA_API_BASE_URL'),
    os.getenv('ASYA_API_KEY'),
    [
        './audio_1.wav',
        './audio_2.wav',
        './audio_3.wav',
        './audio_4.wav',
        './audio_5.wav',
        './audio_6.wav',
    ],
    './results.csv',
    3,
    1,
)

processor.process_files()
