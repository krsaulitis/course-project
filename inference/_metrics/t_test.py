import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import os


# Function to read a CSV file and label its data with the model name
def read_csv_and_label_model(file_path, model_name):
    data = pd.read_csv(file_path)
    data['model'] = model_name  # Assign the model name to a new column
    return data

# Function to calculate the z-scores for a given metric within each model group
def calculate_z_scores(data, metric):
    # Group the data by model and calculate the z-score for each group
    data['z_score_' + metric] = data.groupby('model')[metric].transform(lambda x: (x - x.mean()) / x.std())
    return data

# Function to perform a t-test between two groups for a given metric
def perform_t_test(data, model1, model2, metric):
    group1_scores = data[data['model'] == model1][metric]
    group2_scores = data[data['model'] == model2][metric]

    t_statistic, p_value = stats.ttest_ind(group1_scores, group2_scores, equal_var=False)
    return t_statistic, p_value


nisqa_full_file = 'NISQA_full_results.csv'
nisqa_tts_file = 'NISQA_tts_results.csv'

# Replace with the actual names of your CSV files and the directory they are in
# Also, provide a dictionary where the keys are CSV filenames and values are the actual model names
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
    '../common_voice': 'Common Voice',
}
directory = '../'  # Add the path to your directory of CSV files here

# Read the data from each CSV file and assign the correct model name
combined_data = pd.DataFrame()
for file_name, model_name in folder_to_model_names.items():
    file_path = os.path.join(directory, file_name)
    model_data = read_csv_and_label_model(file_path, model_name)
    combined_data = combined_data.append(model_data, ignore_index=True)

# Specify the models and the metric you want to compare
model1_name = 'MaryTTS'  # Replace with your actual model name
model2_name = 'VITS'  # Replace with your actual model name
metric_to_compare = 'mos_pred'  # Replace with the metric you want to compare

# Calculate the z-scores for the metric of interest
combined_data = calculate_z_scores(combined_data, metric_to_compare)

# Perform the t-test
t_stat, p_val = perform_t_test(combined_data, model1_name, model2_name, metric_to_compare)

print(f'T-statistic: {t_stat}')
print(f'P-value: {p_val}')
print(f'Z-score for {model1_name}: {combined_data[combined_data["model"] == model1_name]["z_score_" + metric_to_compare].values[0]}')
print(f'Z-score for {model2_name}: {combined_data[combined_data["model"] == model2_name]["z_score_" + metric_to_compare].values[0]}')


plt.hist(combined_data[combined_data['model'] == model2_name][metric_to_compare], bins=30, alpha=0.7, color='blue')
plt.title(f"Histogram for {model2_name}")
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

plt.boxplot(combined_data[combined_data['model'] == model2_name][metric_to_compare], vert=False)
plt.title(f"Boxplot for {model2_name}")
plt.show()