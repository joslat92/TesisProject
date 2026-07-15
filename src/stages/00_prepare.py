import pandas as pd
import numpy as np
import yaml
import os
import sys
import json
import hashlib
import platform
import subprocess
from datetime import datetime

# Cargar configuración SSOT

def load_config():
    # AGREGAR EL PARÁMETRO encoding="utf-8"
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def calculate_accumulated_return(series, h):
    """Calcula R_{t -> t+h} = log(P_{t+h}) - log(P_t)"""
    return series.shift(-h) - series

def get_file_hash(filepath):
    """Genera hash SHA256 para snapshot de auditoría."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_git_commit():
    """Hash del commit HEAD (o 'unknown' si no hay git/repositorio)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"

def run_prepare():
    cfg = load_config()
    print(">>> [Fase 1] Iniciando preparación de datos estricta...")

    # 1. Cargar Data Cruda
    raw_path = cfg['data']['raw_source']
    df = pd.read_csv(raw_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # 2. Validaciones Iniciales (Fail-fast)
    exog_config = cfg['features']['exog']
    required_cols = ['Date', cfg['data']['target_col'], *exog_config.keys()]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Faltan columnas en raw data. Requeridas: {required_cols}")
    if df['Date'].isna().any() or df['Date'].duplicated().any():
        raise ValueError("Date contiene nulos o duplicados.")

    target_col = cfg['data']['target_col']
    if df[target_col].isna().any() or (df[target_col] <= 0).any():
        raise ValueError(f"{target_col} debe ser numérico, positivo y no nulo.")
    if df[list(exog_config)].isna().any().any():
        raise ValueError("Las variables exógenas contienen valores nulos.")

    # 3. Transformaciones Base
    df['logP'] = np.log(df[target_col])
    
    # Retorno diario (1-day) para features de entrada
    df['ret_1d'] = df['logP'].diff()
    
    # 4. Generación de Features Exógenas (Lags)
    # Según config: features -> exog
    feature_cols = ['ret_1d'] # Base features
    
    print("    Generando lags exógenos...")
    for col, lags in exog_config.items():
        for lag in lags:
            feat_name = f"{col}_lag{lag}"
            df[feat_name] = df[col].shift(lag)
            feature_cols.append(feat_name)
    
    # Lags del propio retorno (autoregresivo básico para LSTM/ML)
    for lag in [1, 2, 5]:
        df[f'ret_lag{lag}'] = df['ret_1d'].shift(lag)
        feature_cols.append(f'ret_lag{lag}')

    # 5. Generación de Targets por Horizonte y Guardado
    horizons = cfg['features']['horizons']
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    splits_record = {
        "is_range": [None, str(cfg['data']['splits']['train_end'])],
        "oos_range": [str(cfg['data']['splits']['oos_start']), str(cfg['data']['splits']['oos_end'])],
        "created_at": datetime.now().isoformat()
    }

    print(f"    Generando parquets por horizonte: {horizons}")
    
    for h in horizons:
        df_h = df.copy()
        
        # Target: Retorno Acumulado h-pasos adelante
        # R_{t->t+h}
        target_h_col = f"Target_Ret_h{h}"
        df_h[target_h_col] = calculate_accumulated_return(df_h['logP'], h)
        
        # Target Nivel Futuro (solo para reconstrucción/validación, NO para training)
        df_h[f"Target_Price_h{h}"] = df_h[target_col].shift(-h)
        
        # Limpieza: Filas sin target futuro (los últimos h días)
        # Y filas sin features pasados (los primeros max_lag días)
        df_h = df_h.dropna(subset=[target_h_col] + feature_cols)
        
        # Selección de columnas finales para el parquet
        # Guardamos: Date, Targets, Features, y columnas base necesarias
        cols_to_save = ['Date', target_col, 'logP', target_h_col, f"Target_Price_h{h}"] + feature_cols
        
        # Guardar feature set específico para este horizonte
        output_path = os.path.join(processed_dir, f"features_T{h}.parquet")
        df_h[cols_to_save].to_parquet(output_path, index=False)

    # 6. Guardar definición de Splits (JSON)
    splits_path = cfg['paths']['splits_json']
    with open(splits_path, 'w') as f:
        json.dump(splits_record, f, indent=4)
        
    # 7. Snapshot de Reproducibilidad
    snapshot_path = cfg['paths']['metadata_snapshot']
    os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
    
    # Snapshot de reproducibilidad obligatorio (contrato §10): timestamp,
    # dataset_sha256, git_commit, python_version, pip_freeze_hash
    # (hash de requirements.txt según permite el contrato) y config_sha256.
    req_path = "requirements.txt"
    snapshot_data = {
        "timestamp": [datetime.now().isoformat()],
        "dataset_sha256": [get_file_hash(raw_path)],
        "git_commit": [get_git_commit()],
        "python_version": [platform.python_version()],
        "pip_freeze_hash": [get_file_hash(req_path) if os.path.exists(req_path) else "unknown"],
        "config_sha256": [get_file_hash("config.yaml")],
        "n_rows_raw": [len(df)],
    }
    pd.DataFrame(snapshot_data).to_csv(snapshot_path, index=False)
    
    print(f">>> [Fase 1] Completada. Datos guardados en {processed_dir}")

if __name__ == "__main__":
    run_prepare()
