import numpy as np
import pandas as pd
import os

df = pd.read_csv('data/df_final_ready.csv', parse_dates=['Date'])
df['Target_Price'] = np.log(df['Target_Price']).diff()
df = df.dropna()
df.to_csv('data/df_log.csv', index=False)

print('✅ Guardado data/df_log.csv')
