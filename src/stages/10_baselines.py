import pandas as pd
import numpy as np
import yaml
import os
import sys

# Agregar src al path para importar el validador
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.core.contract import ContractValidator

def load_config():
    # AGREGAR EL PARÁMETRO encoding="utf-8"
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_baselines():
    cfg = load_config()
    print(">>> [Fase 2] Generando Baselines (RW) bajo Contrato...")
    
    horizons = cfg['features']['horizons']
    model_name = "RW"
    
    # Rutas
    input_pattern = cfg['paths']['features_parquet_pattern']
    output_dir = cfg['paths']['preds_oos_dir']
    
    # Obtener fechas de OOS para filtrar
    oos_start = pd.Timestamp(cfg['data']['splits']['oos_start'])
    oos_end = pd.Timestamp(cfg['data']['splits']['oos_end'])

    for h in horizons:
        # 1. Cargar datos preparados (Feature Set T{h})
        parquet_path = input_pattern.format(h=h)
        df = pd.read_parquet(parquet_path)
        
        # 2. Filtrar OOS
        df_oos = df[(df['Date'] >= oos_start) & (df['Date'] <= oos_end)].copy()
        
        if df_oos.empty:
            print(f"   [WARN] OOS vacío para h={h}. Saltando.")
            continue

        # 3. Generar Predicciones (Random Walk: Retorno = 0)
        # y_pred_ret = 0
        y_pred_ret = np.zeros(len(df_oos))
        
        # Reconstrucción de Nivel: P_hat = P_t * exp(y_pred_ret)
        # Como y_pred_ret es 0, P_hat = P_t * 1 = P_t
        # 'Target_Price' en la fila OOS es P_t (el precio de HOY, conocido antes de predecir t+h)
        # REVISIÓN: En 00_prepare guardamos 'Target_Price' (P_t) y 'Target_Price_h{h}' (P_{t+h})
        # P_t está en la columna configurada como target_col (ej. Target_Price)
        current_price_col = cfg['data']['target_col']
        y_pred_level = df_oos[current_price_col] * np.exp(y_pred_ret)
        
        # 4. Armar DataFrame Contrato
        df_out = pd.DataFrame()
        df_out['Date'] = df_oos['Date']
        df_out['h'] = h
        df_out['model'] = model_name
        
        # Ground Truth
        df_out['y_true_ret'] = df_oos[f'Target_Ret_h{h}']
        df_out['y_true_level'] = df_oos[f'Target_Price_h{h}']
        
        # Predicciones
        df_out['y_pred_ret'] = y_pred_ret
        df_out['y_pred_level'] = y_pred_level
        
        # 5. Guardar con Naming Contract
        # Pattern: preds_T{h}_{model}.csv
        filename = cfg['contract']['naming']['oos'].format(h=h, model=model_name)
        save_path = os.path.join(output_dir, filename)
        df_out.to_csv(save_path, index=False)
        print(f"   -> Generado: {filename}")

    # Validar al final
    print("   Validando artefactos...")
    val = ContractValidator()
    val.validate_all_outputs()

if __name__ == "__main__":
    run_baselines()
