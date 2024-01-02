import pandas as pd

df = pd.read_csv('../inference/_data/NISQA_mos_results.csv')

avg_quality_mos = df['mos_pred'].mean()

if 'noi_pred' in df.columns and 'col_pred' in df.columns and 'dis_pred' in df.columns and 'loud_pred' in df.columns:
    avg_noisiness = df['noi_pred'].mean()
    avg_coloration = df['col_pred'].mean()
    avg_discontinuity = df['dis_pred'].mean()
    avg_loudness = df['loud_pred'].mean()
    print(f'Quality MOS: {avg_quality_mos}, Noisiness: {avg_noisiness}, Coloration: {avg_coloration}, Discontinuity: {avg_discontinuity}, Loudness: {avg_loudness}')
else:
    print(f'Quality MOS: {avg_quality_mos}')
