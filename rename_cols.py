# rename_cols.py  –– normaliza columnas en predictions_all_folds.csv
import sys, pandas as pd, pathlib, glob

for csv in glob.glob('models/*/predictions_all_folds.csv'):
    p = pathlib.Path(csv)
    df = pd.read_csv(p)
    if 'y_pred' not in df.columns:
        pred_col = [c for c in df.columns if c not in ('Date','y_true')][0]
        df = df.rename(columns={pred_col:'y_pred'})
    df = df[['Date','y_true','y_pred']]
    df.to_csv(p, index=False)
    print('✓', p)
