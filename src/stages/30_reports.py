import pandas as pd
import yaml
import os
import sys

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_path not in sys.path:
    sys.path.append(root_path)

from src.core.viz import plot_oos_results

def run_reports(config_path):
    print("--- Stage 30: Generando Gráficas de Tesis ---")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    preds_dir = config['paths']['preds_oos_dir']
    figs_dir = config['paths']['figs_dir']
    os.makedirs(figs_dir, exist_ok=True)
    
    # Consolidar predicciones para la gráfica
    results = None
    for file in os.listdir(preds_dir):
        if file.endswith(".csv"):
            model_name = file.replace("preds_", "").replace(".csv", "")
            df = pd.read_csv(os.path.join(preds_dir, file))
            if results is None:
                results = df[['Date', 'Target_Price']].rename(columns={'Target_Price': 'y_true'})
            results[model_name] = df['y_pred']
    
    results['Date'] = pd.to_datetime(results['Date'])
    plot_oos_results(results, os.path.join(figs_dir, "comparativa_oos.png"))
    print(f"[OK] Gráfica generada en: {figs_dir}")

if __name__ == "__main__":
    run_reports("config.yaml")
