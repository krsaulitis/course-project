Here are some of the tools required and the installation instructions for them.

### NISQA
Clone - https://github.com/gabrielmittag/NISQA.git
Follow the instruction manual in the repository
- `python run_predict.py --mode predict_dir --pretrained_model weights/nisqa.tar --data_dir ../../inference/_data/audios_hd --num_workers 0 --bs 10 --output_dir ../../inference/_data`

### FFMPEG
- `ffmpeg -i ./datasets/ljspeech/wavs/LJ038-0050.wav -ar 16000 ./inference/output.wav`