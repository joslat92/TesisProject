import pandas as pd
import os
import yaml
import sys

sys.path.append(os.getcwd())
from src.core.metrics import calculate_metrics

def run_wf_metrics(config_path):
    print("--- Stage 24: Calculando Métricas por Bloque Walk-Forward ---")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    wf_file = os.path.join(config['paths']['preds_wf_dir'], "preds_ARIMA_WF.csv")
    if not os.path.exists(wf_file):
        print("[ERROR] No se encuentran predicciones WF. Ejecuta primero el Stage 14.")
        return

    df_wf = pd.read_csv(wf_file)
    monthly_metrics = []

    # Calcular métricas para cada bloque (mes)
    for block in sorted(df_wf['block'].unique()):
        block_df = df_wf[df_wf['block'] == block]
        res = calculate_metrics(block_df['Target_Price'], block_df['y_pred'])
        res['Block'] = block
        # Asignar nombre del mes aproximado
        res['Month'] = pd.to_datetime(block_df['Date']).iloc[0].strftime('%b')
        monthly_metrics.append(res)
    
    df_res = pd.DataFrame(monthly_metrics)
    output_path = os.path.join(config['paths']['reports_data_dir'], "metrics_wf_monthly.csv")
    df_res.to_csv(output_path, index=False)
    print(f"[OK] Métricas mensuales guardadas en: {output_path}")

if __name__ == "__main__":
    run_wf_metrics("config.yaml")
