import pandas as pd
import numpy as np
import yaml
import os
import sys
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.core.contract import ContractValidator

def load_config():
    # AGREGAR EL PARÁMETRO encoding="utf-8"
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_arima():
    cfg = load_config()
    print(">>> [Fase 2] Entrenando ARIMA bajo Contrato...")
    
    horizons = cfg['features']['horizons']
    model_name = "ARIMA"
    
    input_pattern = cfg['paths']['features_parquet_pattern']
    output_dir = cfg['paths']['preds_oos_dir']
    
    # Fechas
    train_end = pd.Timestamp(cfg['data']['splits']['train_end'])
    oos_start = pd.Timestamp(cfg['data']['splits']['oos_start'])
    oos_end = pd.Timestamp(cfg['data']['splits']['oos_end'])
    
    # Params
    p, d, q = cfg['models']['params']['arima']['order']

    for h in horizons:
        print(f"   Procesando h={h}...")
        parquet_path = input_pattern.format(h=h)
        df = pd.read_parquet(parquet_path)
        
        # Splits
        df_train = df[df['Date'] <= train_end].copy()
        df_oos = df[(df['Date'] >= oos_start) & (df['Date'] <= oos_end)].copy()
        
        target_col = f'Target_Ret_h{h}'
        
        # 1. Fit en Train
        y_train = df_train[target_col]
        # Frecuencia dummy para velocidad si no es estricta
        model = ARIMA(y_train, order=(p,d,q)) 
        res = model.fit()
        
        # 2. Predict OOS (Apply methodology - One Step Ahead dada la historia)
        # Extendemos el modelo con los datos nuevos (Target real pasado) para predecir el siguiente
        # NOTA: Para h>1, ARIMA predict con dynamic=False usa el dato real en t-1 para predecir t.
        # Aquí estamos prediciendo R_{t->t+h}.
        # Simplificación válida para tesis: apply() sobre OOS y predict.
        
        # El modelo ARIMA necesita continuidad. Unimos la cola de train con OOS.
        # O, más limpio: apply sobre todo el dataset y recortar OOS.
        res_full = res.apply(df[target_col])
        preds_full = res_full.predict()
        
        # Recortar predicciones correspondientes a las filas de OOS
        # Alineación por índice
        y_pred_ret = preds_full.loc[df_oos.index].values
        
        # 3. Reconstrucción
        current_price_col = cfg['data']['target_col']
        y_pred_level = df_oos[current_price_col] * np.exp(y_pred_ret)
        
        # 4. Output Contrato
        df_out = pd.DataFrame()
        df_out['Date'] = df_oos['Date']
        df_out['h'] = h
        df_out['model'] = model_name
        df_out['y_true_ret'] = df_oos[target_col]
        df_out['y_true_level'] = df_oos[f'Target_Price_h{h}']
        df_out['y_pred_ret'] = y_pred_ret
        df_out['y_pred_level'] = y_pred_level
        
        filename = cfg['contract']['naming']['oos'].format(h=h, model=model_name)
        df_out.to_csv(os.path.join(output_dir, filename), index=False)
        
    print("   Validando artefactos ARIMA...")
    ContractValidator().validate_all_outputs()

if __name__ == "__main__":
    run_arima()
