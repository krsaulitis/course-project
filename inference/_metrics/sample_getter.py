import os
import pandas as pd


def top_worst_10():
    metric = 'cer'
    df = pd.read_csv('average_precision_cer_scores.csv')

    top_10 = df.sort_values(by=metric, ascending=False).head(20)
    last_10 = df.sort_values(by=metric, ascending=True).head(10)

    print(top_10[['deg', metric]])
    print(last_10[['deg', metric]])


def calculate_average_scores(directory_list, file_name, output_file):
    # Initialize an empty DataFrame to store data from all files
    combined_df = pd.DataFrame()

    # Loop through each file and append the data to the combined DataFrame
    for file, model in directory_list.items():
        file_path = os.path.join(file, file_name)
        temp_df = pd.read_csv(file_path)
        combined_df = combined_df.append(temp_df, ignore_index=True)

    # Group by the 'deg' column and calculate the average mos_pred score
    averages = combined_df.groupby('deg')['cer'].mean().reset_index()

    # Save the result to a new file
    averages.to_csv(output_file, index=False)


nisqa_full_file = 'NISQA_full_results.csv'
nisqa_tts_file = 'NISQA_tts_results.csv'
cer_wer_file = 'cer_wer_fix.csv'

folder_to_model_names = {
    '../comospeech': 'CoMoSpeech',
    '../mqtts': 'MQTTS',
    '../overflow': 'OverFlow',
    '../your_tts': 'YourTTS',
    '../vits': 'VITS',
    '../grad_tts': 'GradTTS',
    '../fastspeech_2': 'FastSpeech 2',
    '../glow_tts': 'GlowTTS',
    '../mary_tts': 'MaryTTS',
    # '../_data': 'Common Voice',
}


# calculate_average_scores(folder_to_model_names, cer_wer_file, 'average_precision_cer_scores.csv')
top_worst_10()