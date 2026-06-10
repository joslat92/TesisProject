import pandas as pd
import yaml
import os

def load_config(config_path='config.yaml'):
    """Carga la configuración centralizada."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_raw_data(file_path):
    """Carga el CSV original y valida el esquema mínimo[cite: 12, 41]."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Validaciones obligatorias según contrato [cite: 41]
    required_cols = ['Date', 'Target_Price', 'Sentiment_GDELT', 'VIX_Close']
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        raise KeyError(f"Error de esquema: Faltan las columnas {missing}")
        
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def save_canonical_data(df, output_path):
    """Guarda el dataset procesado en formato Parquet."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"[OK] Dataset canónico guardado en: {output_path}")
