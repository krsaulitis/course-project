import pandas as pd
from scipy import stats
import os


def read_csv_and_label_model(file_path, model_name, metrics, headers=True):
    if headers:
        data = pd.read_csv(file_path)
        data.rename(columns=metrics, inplace=True)
        metrics = list(metrics.values())
        data = data[metrics]
    else:
        data = pd.read_csv(file_path, header=None, names=list(metrics.values()) + ['model'])

    data['Model'] = model_name
    return data


# Function to perform a t-test between two groups for a given metric
def perform_t_test(data, model1, model2, metric):
    group1_scores = data[data['Model'] == model1][metric]
    group2_scores = data[data['Model'] == model2][metric]

    # Diagnostics
    # print(f"Group 1 ({model1}) size: {len(group1_scores)}, Unique values: {group1_scores.nunique()}")
    # print(f"Group 2 ({model2}) size: {len(group2_scores)}, Unique values: {group2_scores.nunique()}")
    # print(f"Group 1 ({model1}) size after NaN removal: {len(group1_scores)}, Unique values: {group1_scores.nunique()}")
    # print(f"Group 2 ({model2}) size after NaN removal: {len(group2_scores)}, Unique values: {group2_scores.nunique()}")
    #
    # if group1_scores.isnull().any():
    #     print("Warning: NaN values detected in the data 1.")
    # if group2_scores.isnull().any():
    #     print("Warning: NaN values detected in the data 2.")
    # if group1_scores.nunique() == 1 or group2_scores.nunique() == 1:
    #     print("Warning: One of the groups has no variability.")
    #
    # exit(0)

    t_statistic, p_value = stats.ttest_ind(group1_scores, group2_scores, equal_var=False, nan_policy='omit')
    return t_statistic, p_value


nisqa_full_file = 'NISQA_full_results.csv'
nisqa_tts_file = 'NISQA_tts_results.csv'
cer_wer_file = 'cer_wer_fix.csv'

folder_to_model_names = {
    '../comospeech': 'CoMoSpeech',
    # '../mqtts': 'MQTTS',
    '../overflow': 'OverFlow',
    '../your_tts': 'YourTTS',
    '../vits': 'VITS',
    '../grad_tts': 'GradTTS',
    # '../fastspeech_2': 'FastSpeech 2',
    '../glow_tts': 'GlowTTS',
    '../mary_tts': 'MaryTTS',
    '../_data': 'Common Voice',
}

file_metrics = {
    nisqa_full_file: (
        {
            'mos_pred': 'Quality',
            'col_pred': 'Coloration',
            'noi_pred': 'Noisiness',
            'dis_pred': 'Discontinuity',
            'loud_pred': 'Loudness',
        },
        True,
    ),
    nisqa_tts_file: (
        {'mos_pred': 'Naturalness'},
        True,
    ),
    cer_wer_file: (
        {'cer': 'CER', 'wer': 'WER'},
        True,
    ),
}

# Read and combine the data
combined_data = pd.DataFrame()
for folder, model_name in folder_to_model_names.items():
    for file, (metrics, headers) in file_metrics.items():
        file_path = os.path.join(folder, file)
        if os.path.exists(file_path):
            model_data = read_csv_and_label_model(file_path, model_name, metrics, headers=headers)
            combined_data = combined_data.append(model_data, ignore_index=True)

# Checking for missing data
print("Checking for missing data:")
print(combined_data.isnull().sum())

# Print out the number of data points for each model and metric
print("\nData points per model and metric:")
for model in folder_to_model_names.values():
    print(f"\nModel: {model}")
    for metric in set(sum([list(metrics.keys()) for metrics, _ in file_metrics.values()], [])):
        renamed_metric = next((renamed for original, renamed in file_metrics[file][0].items() if original == metric),
                              metric)
        if renamed_metric in combined_data.columns:
            count = combined_data[combined_data['Model'] == model][renamed_metric].count()
            print(f"  Metric '{renamed_metric}': {count} data points")

# exit(0)

# Perform t-test for each model against Common Voice
t_test_results = []
common_voice = 'Common Voice'
for model in folder_to_model_names.values():
    if model != common_voice:
        for file, (metrics, _) in file_metrics.items():  # Iterate over file_metrics
            for original_metric, renamed_metric in metrics.items():  # Iterate over metrics mapping
                if renamed_metric in combined_data.columns:
                    t_stat, p_val = perform_t_test(combined_data, model, common_voice, renamed_metric)
                    t_test_results.append((model, renamed_metric, t_stat, p_val))

# Convert results to a DataFrame for display
t_test_df = pd.DataFrame(t_test_results, columns=['Model', 'Metric', 'T-Statistic', 'P-Value'])
print(t_test_df)

model_order = [
    'CoMoSpeech',
    # 'MQTTS',
    'OverFlow',
    'YourTTS',
    'VITS',
    'GradTTS',
    # 'FastSpeech 2',
    'GlowTTS',
    'MaryTTS',
]
t_test_df['Model'] = pd.Categorical(t_test_df['Model'], categories=model_order, ordered=True)
t_test_df = t_test_df.sort_values('Model')

t_stats_table = t_test_df.pivot(index='Model', columns='Metric', values='T-Statistic')
p_values_table = t_test_df.pivot(index='Model', columns='Metric', values='P-Value')

t_stats_table.to_csv('t_statistics.csv')
p_values_table.to_csv('p_values.csv')

t_test_df.to_csv('t_test_results.csv', index=False)
