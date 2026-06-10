import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
import sys

# Configuración de estilo
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_figures():
    cfg = load_config()
    print(">>> [Fase 4] Generando Figuras Finales...")
    
    preds_dir = cfg['paths']['preds_oos_dir']
    figs_dir = cfg['paths']['figures_dir']
    os.makedirs(figs_dir, exist_ok=True)
    
    horizons = cfg['features']['horizons']
    models = cfg['models']['active_models']
    
    # --- 1. FIGURAS DE SERIES DE TIEMPO (Una por Horizonte) ---
    for h in horizons:
        plt.figure()
        
        # Cargar datos de todos los modelos para este h
        data_found = False
        
        # Plotear Ground Truth (solo una vez, tomamos del Baseline)
        base_file = os.path.join(preds_dir, cfg['contract']['naming']['oos'].format(h=h, model="RW"))
        if os.path.exists(base_file):
            df_base = pd.read_csv(base_file)
            df_base['Date'] = pd.to_datetime(df_base['Date'])
            plt.plot(df_base['Date'], df_base['y_true_level'], label='Real Market', color='black', linewidth=1.5, alpha=0.7)
            data_found = True
        
        # Plotear Modelos
        colors = {'RW': 'gray', 'ARIMA': 'blue', 'SARIMAX': 'cyan', 'LSTM': 'red'}
        
        for model in models:
            filename = cfg['contract']['naming']['oos'].format(h=h, model=model)
            filepath = os.path.join(preds_dir, filename)
            
            if not os.path.exists(filepath):
                continue
                
            df = pd.read_csv(filepath)
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Estilo diferente para baseline
            style = '--' if model == 'RW' else '-'
            width = 1 if model == 'RW' else 1.2
            
            plt.plot(df['Date'], df['y_pred_level'], label=f'Pred {model}', 
                     linestyle=style, linewidth=width, color=colors.get(model, 'green'))
            
        if data_found:
            plt.title(f'Forecast vs Real (Horizon h={h}) - OOS 2024')
            plt.xlabel('Date')
            plt.ylabel('Price Level')
            plt.legend()
            plt.tight_layout()
            
            save_path = os.path.join(figs_dir, f"Fig_T{h}_TimeSeries.png")
            plt.savefig(save_path, dpi=300)
            print(f"   -> Guardado: {save_path}")
            plt.close()

    # --- 2. FIGURA RESUMEN MÉTRICAS (Barras) ---
    metrics_path = cfg['paths']['metrics_oos']
    if os.path.exists(metrics_path):
        df_met = pd.read_csv(metrics_path)
        
        # Grafico RMSE
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_met, x='Horizon', y='RMSE', hue='Model', palette='viridis')
        plt.title('RMSE by Horizon and Model (Lower is Better)')
        plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, "Fig_Metrics_RMSE.png"), dpi=300)
        plt.close()
        
        # Grafico MDA
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_met, x='Horizon', y='MDA', hue='Model', palette='magma')
        plt.axhline(0.5, color='red', linestyle='--', label='Random Chance (50%)')
        plt.title('Directional Accuracy (MDA) by Horizon (Higher is Better)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, "Fig_Metrics_MDA.png"), dpi=300)
        print("   -> Guardado: Resúmenes de Métricas (RMSE/MDA)")

    # --- 3. SCATTER PLOTS (LSTM vs Real Returns) ---
    # Solo para el mejor caso (H=20) para demostrar la tesis
    h_best = 20
    model_best = "LSTM"
    file_best = os.path.join(preds_dir, cfg['contract']['naming']['oos'].format(h=h_best, model=model_best))
    
    if os.path.exists(file_best):
        df = pd.read_csv(file_best)
        plt.figure(figsize=(8, 8))
        
        # Scatter
        sns.scatterplot(x=df['y_true_ret'], y=df['y_pred_ret'], alpha=0.6, color='purple')
        
        # Línea de 45 grados y cuadrantes
        limit = max(df['y_true_ret'].abs().max(), df['y_pred_ret'].abs().max()) * 1.1
        plt.plot([-limit, limit], [-limit, limit], color='gray', linestyle='--')
        plt.axhline(0, color='black', linewidth=0.8)
        plt.axvline(0, color='black', linewidth=0.8)
        
        plt.xlim(-limit, limit)
        plt.ylim(-limit, limit)
        plt.title(f'Calibration Scatter: {model_best} (h={h_best})\nMDA indicates points in Q1 and Q3')
        plt.xlabel('Real Returns')
        plt.ylabel('Predicted Returns')
        plt.tight_layout()
        
        save_path = os.path.join(figs_dir, f"Fig_T{h_best}_{model_best}_Scatter.png")
        plt.savefig(save_path, dpi=300)
        print(f"   -> Guardado Scatter: {save_path}")
        plt.close()

if __name__ == "__main__":
    run_figures()
