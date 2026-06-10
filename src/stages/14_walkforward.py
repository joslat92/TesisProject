import pandas as pd
import numpy as np
import yaml
import os
import sys
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Forzar reconocimiento del paquete raíz
sys.path.append(os.getcwd())

def run_walk_forward(config_path):
    print("--- Stage 14: Iniciando Walk-Forward Mensual (2024) ---")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 1. Cargar datos finales
    data_path = os.path.join(config['paths']['data_processed'], "dataset_final.parquet")
    df = pd.read_parquet(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 2. Definir ventanas mensuales para 2024 según config
    wf_year = config['splits']['wf']['year']
    months = pd.date_range(start=f'{wf_year}-01-01', end=f'{wf_year}-12-01', freq='MS')
    
    all_wf_results = []
    
    # El orden (2,1,3) fue el ganador por BIC en el Stage 11
    order = (2, 1, 3) 

    # 3. Ciclo de Re-entrenamiento (Walk-Forward)
    for i, month_start in enumerate(tqdm(months, desc="Bloques WF")):
        # Definir fin de mes
        month_end = month_start + pd.offsets.MonthEnd(1)
        
        # Train: Todo lo anterior al inicio del mes actual
        train_data = df[df['Date'] < month_start]['Target_Price']
        # Test: El mes actual
        test_df = df[(df['Date'] >= month_start) & (df['Date'] <= month_end)].copy()
        
        if len(test_df) == 0:
            continue
            
        try:
            # Re-entrenar modelo con toda la historia disponible hasta t-1
            model = SARIMAX(train_data, order=order).fit(disp=False)
            
            # Pronóstico dinámico para el mes completo
            forecast = model.get_forecast(steps=len(test_df))
            test_df['y_pred'] = forecast.predicted_mean.values
            test_df['block'] = i + 1
            
            all_wf_results.append(test_df)
            
        except Exception as e:
            print(f"Error en bloque {month_start.strftime('%Y-%m')}: {e}")

    # 4. Consolidar y Guardar
    if all_wf_results:
        df_wf = pd.concat(all_wf_results)
        output_dir = config['paths']['preds_wf_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, "preds_ARIMA_WF.csv")
        df_wf[['Date', 'Target_Price', 'y_pred', 'block']].to_csv(output_path, index=False)
        print(f"\n[OK] Walk-Forward completado. Resultados en: {output_path}")
    else:
        print("[ERROR] No se generaron resultados para el Walk-Forward.")

if __name__ == "__main__":
    config_file = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"
    run_walk_forward(config_file)
