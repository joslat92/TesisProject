import pandas as pd
import yaml
import os
import glob
from pathlib import Path


class ContractValidator:
    def __init__(self, config_path="config.yaml"):
        # AQUÍ TAMBIÉN
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        
        self.required_cols = set(self.cfg['contract']['columns_required'])
        self.wf_extra = set(self.cfg['contract']['columns_wf_extra'])
        self.horizons = self.cfg['features']['horizons']
        self.valid_models = self.cfg['models']['active_models']

    def validate_structure(self):
        """Valida que existan las carpetas críticas."""
        dirs = [
            self.cfg['paths']['preds_oos_dir'],
            self.cfg['paths']['preds_wf_dir'],
            self.cfg['paths']['figures_dir'],
            "data/processed",
            "reports/data"
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
        print(">>> [Contract] Estructura de directorios verificada.")

    def validate_prediction_file(self, filepath, context="OOS"):
        """
        Valida un CSV de predicción individual contra el contrato estricto.
        context: 'OOS' o 'WF'
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

        df = pd.read_csv(filepath)
        cols = set(df.columns)

        # 1. Validar Columnas
        missing = self.required_cols - cols
        if missing:
            raise ValueError(f"[Contract Fail] {filepath} le faltan columnas: {missing}")
        
        if context == "WF":
            missing_wf = self.wf_extra - cols
            if missing_wf:
                 raise ValueError(f"[Contract Fail] WF {filepath} le faltan columnas extra: {missing_wf}")

        # 2. Validar Tipos y Nulos
        # No permitimos NaNs en predicciones o targets (deben estar limpios o recortados)
        if df[['y_pred_ret', 'y_pred_level']].isna().any().any():
             raise ValueError(f"[Contract Fail] {filepath} contiene NaNs en predicciones.")
        
        # 3. Validar consistencia de metadatos (h y model dentro del CSV vs nombre archivo)
        # Asume que el nombre del archivo sigue el patrón preds_T{h}_{model}...
        filename = os.path.basename(filepath)
        
        # Validar Horizon en contenido
        h_values = df['h'].unique()
        if len(h_values) != 1:
             raise ValueError(f"[Contract Fail] {filepath} tiene múltiples horizontes mezclados: {h_values}")
        
        current_h = h_values[0]
        if current_h not in self.horizons:
             raise ValueError(f"[Contract Fail] Horizonte {current_h} no permitido en config.")
        
        # Validar Modelo en contenido
        model_values = df['model'].unique()
        if len(model_values) != 1:
             raise ValueError(f"[Contract Fail] {filepath} tiene múltiples modelos mezclados: {model_values}")
             
        # Chequeo "fuzzy" del modelo (ej. LSTM_Plain vs LSTM)
        # Esto depende de cuan estricto quieras ser con los nombres exactos
        
        print(f"✅ Validado: {filename} ({len(df)} filas, h={current_h})")
        return True

    def validate_all_outputs(self):
        """Barre todos los outputs y valida integridad masiva."""
        print("\n--- INICIANDO VALIDACIÓN DE CONTRATO MASIVA ---")
        
        # 1. OOS
        oos_dir = self.cfg['paths']['preds_oos_dir']
        files_oos = glob.glob(os.path.join(oos_dir, "*.csv"))
        if not files_oos:
            print("⚠️ [Warn] No hay archivos OOS para validar.")
        
        for f in files_oos:
            try:
                self.validate_prediction_file(f, context="OOS")
            except Exception as e:
                print(f"❌ ERROR en {os.path.basename(f)}: {str(e)}")
                # Fail-fast: lanzar error para detener pipeline si es necesario
                # raise e 

        # 2. WF
        wf_dir = self.cfg['paths']['preds_wf_dir']
        files_wf = glob.glob(os.path.join(wf_dir, "*.csv"))
        if not files_wf:
             print("⚠️ [Warn] No hay archivos WF para validar.")

        for f in files_wf:
            try:
                self.validate_prediction_file(f, context="WF")
            except Exception as e:
                print(f"❌ ERROR en {os.path.basename(f)}: {str(e)}")

        print("--- VALIDACIÓN COMPLETADA ---\n")

# Uso rápido desde línea de comandos
if __name__ == "__main__":
    val = ContractValidator()
    val.validate_structure()
    val.validate_all_outputs()
