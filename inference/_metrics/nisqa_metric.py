import pandas as pd

model = 'comospeech'
df_full = pd.read_csv(f'../{model}/NISQA_full_results.csv')
df_tts = pd.read_csv(f'../{model}/NISQA_tts_results.csv')

avg_mos = df_tts['mos_pred'].mean()
avg_quality = df_full['mos_pred'].mean()
avg_coloration = df_full['col_pred'].mean()
avg_noisiness = df_full['noi_pred'].mean()
avg_discontinuity = df_full['dis_pred'].mean()
avg_loudness = df_full['loud_pred'].mean()

print(f'MOS: {avg_mos}')
print(f'Quality: {avg_quality}')
print(f'Coloration: {avg_coloration}')
print(f'Noisiness: {avg_noisiness}')
print(f'Discontinuity: {avg_discontinuity}')
print(f'Loudness: {avg_loudness}')
