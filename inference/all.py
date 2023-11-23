import os

for path in os.listdir('.'):
    if os.path.isdir(path):
        print(f'Running {path}...')
        os.system(f'python {path}/test.py')
        print(f'Finished {path}.\n')
