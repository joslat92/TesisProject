import pandas as pd
import numpy as np
import yaml
import os

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def run_split():
    cfg = load_config()
    
    # 1. Cargar datos desde la ruta definida en tu YAML
    raw_path = cfg['paths']['data_raw'] #
    print(f">>> Cargando raw data desde: {raw_path}")
    
    # Manejo de error si el archivo no existe
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"No se encontró el archivo en: {raw_path}. Verifica 'paths.data_raw' en config.yaml")

    df = pd.read_csv(raw_path)
    date_col = cfg['schema']['date_col'] # 'Date'
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # 2. Asegurar Log Price (base para los retornos)
    target_col = cfg['schema']['target_col'] # 'Target_Price'
    if 'logP' not in df.columns:
        df['logP'] = np.log(df[target_col])

    # 3. Generar Targets por Horizonte (CRÍTICO: Retornos acumulados)
    # R_{t->t+h} = logP_{t+h} - logP_t
    horizons = cfg['horizons'] # [1, 5, 10, 20]
    
    print(f">>> Generando targets para horizontes: {horizons}")
    for h in horizons:
        col_name = f"Target_Ret_h{h}"
        # Traemos el precio futuro (shift negativo) para comparar con hoy
        df[f'logP_future_h{h}'] = df['logP'].shift(-h)
        df[col_name] = df[f'logP_future_h{h}'] - df['logP']
        
        # Guardamos el precio futuro real para validaciones (opcional)
        df[f'Target_Price_h{h}'] = df[target_col].shift(-h)

    # 4. Limpieza: Eliminar filas que no tienen futuro conocido (los últimos días)
    max_h = max(horizons)
    df_clean = df.dropna(subset=[f"Target_Ret_h{max_h}"]).copy()
    
    # 5. Partición usando tus fechas del YAML
    # Fechas IS
    is_end = pd.Timestamp(cfg['splits']['is']['end'])
    
    # Fechas OOS
    oos_start = pd.Timestamp(cfg['splits']['oos']['start'])
    oos_end = pd.Timestamp(cfg['splits']['oos']['end'])

    # Aplicar filtros
    df_is = df_clean[df_clean[date_col] <= is_end].copy()
    
    df_oos = df_clean[
        (df_clean[date_col] >= oos_start) & 
        (df_clean[date_col] <= oos_end)
    ].copy()

    # WF (Usamos la definición de OOS como base para los bloques, o filtro de año 2024)
    # Tu yaml dice year: 2024
    wf_year = cfg['splits']['wf']['year']
    df_wf = df_clean[df_clean[date_col].dt.year == wf_year].copy()

    # 6. Validaciones de Integridad (Smoke Checks)
    print("\n=== CHEQUEO DE INTEGRIDAD ===")
    print(f"IS Range:  {df_is[date_col].min().date()} -> {df_is[date_col].max().date()} | Filas: {len(df_is)}")
    print(f"OOS Range: {df_oos[date_col].min().date()} -> {df_oos[date_col].max().date()} | Filas: {len(df_oos)}")
    
    if len(df_oos) == 0:
        raise ValueError("¡ERROR CRÍTICO! El DataFrame OOS está vacío. Revisa las fechas en config.yaml vs tu CSV.")
        
    if df_oos[date_col].max() > oos_end:
         print(f"¡ALERTA! Hay datos en OOS posteriores a {oos_end}. Se están filtrando correctamente ahora.")

    # 7. Guardar en las rutas definidas en 'paths'
    # Crear directorios si no existen
    os.makedirs(cfg['paths']['processed_dir'], exist_ok=True)
    
    df_is.to_parquet(cfg['paths']['is_parquet'])
    df_oos.to_parquet(cfg['paths']['oos_parquet'])
    df_wf.to_parquet(cfg['paths']['wf_parquet'])
    
    # Guardar exógenas si se requiere por separado (opcional, basado en tu yaml)
    if 'exog_parquet' in cfg['paths']:
        # Simplemente guardamos todo el dataset limpio como exog base
        df_clean.to_parquet(cfg['paths']['exog_parquet'])

    print(f"\n>>> Archivos guardados en: {cfg['paths']['processed_dir']}")

if __name__ == "__main__":
    run_split()
