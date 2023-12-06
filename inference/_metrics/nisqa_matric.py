from tools.nisqa.nisqa.NISQA_model import nisqaModel

args = {
    # 'mode': 'predict_file',
    'mode': 'predict_dir',
    'pretrained_model': './nisqa/weights/nisqa.tar',
    # 'deg': '../inference/data/output_1.wav',
    # 'deg': '../inference/vits/samples/sample_1.wav',
    # 'data_dir': '../inference/vits/samples',
    # 'output_dir': '../inference/vits/samples',
    'data_dir': '../inference/data',
    'output_dir': '../inference/data',
    'ms_channel': 0,
}

nisqa = nisqaModel(args)
nisqa.predict()
