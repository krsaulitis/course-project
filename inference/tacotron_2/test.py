import torch
import scipy
import torchaudio
from tqdm.auto import tqdm


# Tacotron2 + WaveRNN


class Tacotron2Generator:
    def __init__(self):
        # bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
        # self.tacotron2 = bundle.get_tacotron2().to('cpu')
        # self.tacotron2.eval()

        self.tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
                                        'nvidia_tacotron2',
                                        model_math='fp16',
                                        pretrained=False)
        checkpoint = torch.hub.load_state_dict_from_url(
            'https://api.ngc.nvidia.com/v2/models/nvidia/tacotron2pyt_fp32/versions/1/files/nvidia_tacotron2pyt_fp32_20190306.pth',
            map_location="cpu")
        # self.tacotron2 = self.tacotron2.to('cuda')
        state_dict = {key.replace("module.", ""): value for key, value in checkpoint["state_dict"].items()}
        self.tacotron2.load_state_dict(state_dict)
        self.tacotron2.eval()

        # self.wave_rnn = bundle.get_vocoder().to('cpu')
        # self.wave_rnn.eval()

        self.waveglow = torch.hub.load(
            "NVIDIA/DeepLearningExamples:torchhub",
            "nvidia_waveglow",
            model_math="fp16",
        )

        # checkpoint = torch.hub.load_state_dict_from_url(
        #     "https://api.ngc.nvidia.com/v2/models/nvidia/waveglowpyt_fp32/versions/1/files/nvidia_waveglowpyt_fp32_20190306.pth",
        #     # noqa: E501
        #     progress=False,
        #     map_location='cpu',
        # )
        #
        # state_dict = {key.replace("module.", ""): value for key, value in checkpoint["state_dict"].items()}

        # self.waveglow.load_state_dict(state_dict)
        self.waveglow = self.waveglow.remove_weightnorm(self.waveglow)
        self.waveglow.eval()

        # with torch.no_grad():
        #     waveforms = waveglow.infer(spec)

        self.utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')
        # self.processor = bundle.get_text_processor()

    def generate(self, text, path):
        # processed, lengths = self.processor(text)
        processed, lengths = self.utils.prepare_input_sequence([text])
        processed = processed.to('cpu')
        lengths = lengths.to('cpu')

        with torch.no_grad():
            # run the models
            spec, spec_lengths, _ = self.tacotron2.infer(processed, lengths)
            waveforms = self.waveglow.infer(spec)
            # waveforms, lengths = self.wave_rnn(spec, spec_lengths)

        audio_numpy = waveforms[0].data.cpu().numpy()
        rate = 22050
        scipy.io.wavfile.write(path, rate, audio_numpy)


generator = Tacotron2Generator()


def generate():
    with open('../_data/filtered_no_duplicates_max_upvotes.tsv', 'r') as file:
        rows = file.readlines()

        for i, row in tqdm(enumerate(rows), total=len(rows)):
            sentence = row.split('\t')[2]
            file_id = row.split('\t')[1].split("_")[-1].split(".")[0]
            path = f'./audios_alt/audio_{file_id}.wav'
            generator.generate(sentence, path)


generate()
