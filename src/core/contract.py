import pandas as pd
import yaml
import os
import sys
import glob
import subprocess
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
            if df['block'].nunique() != 1:
                 raise ValueError(f"[Contract Fail] WF {filepath} mezcla bloques: {df['block'].unique()}")

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
        
        print(f"[OK] Validado: {filename} ({len(df)} filas, h={current_h})")
        return True

    def validate_all_outputs(self, fail_fast=True):
        """
        Barre todos los outputs y valida integridad masiva.
        fail_fast=True (gate del pipeline): cualquier violación de contrato
        levanta RuntimeError con la lista completa de errores.
        """
        print("\n--- INICIANDO VALIDACIÓN DE CONTRATO MASIVA ---")
        errors = []

        # 1. OOS
        oos_dir = self.cfg['paths']['preds_oos_dir']
        files_oos = glob.glob(os.path.join(oos_dir, "*.csv"))
        if not files_oos:
            print("[Warn] No hay archivos OOS para validar.")

        for f in files_oos:
            try:
                self.validate_prediction_file(f, context="OOS")
            except Exception as e:
                errors.append(f"{os.path.basename(f)}: {str(e)}")
                print(f"[ERROR] en {os.path.basename(f)}: {str(e)}")

        # 2. WF
        wf_dir = self.cfg['paths']['preds_wf_dir']
        files_wf = glob.glob(os.path.join(wf_dir, "*.csv"))
        if not files_wf:
             print("[Warn] No hay archivos WF para validar.")

        for f in files_wf:
            try:
                self.validate_prediction_file(f, context="WF")
            except Exception as e:
                errors.append(f"{os.path.basename(f)}: {str(e)}")
                print(f"[ERROR] en {os.path.basename(f)}: {str(e)}")

        # 3. Consistencia de y_true entre modelos del mismo horizonte:
        # en cada fecha común, y_true_ret debe ser IDÉNTICO en todos los
        # archivos (un y_true desalineado = métricas inválidas; ver bug
        # off-by-seq_len del 2026-06-11).
        errors += self._check_truth_consistency(files_oos, "OOS")
        errors += self._check_truth_consistency(files_wf, "WF")

        if errors and fail_fast:
            raise RuntimeError(
                f"[Contract GATE] {len(errors)} archivo(s) violan el contrato:\n"
                + "\n".join(errors)
            )
        print("--- VALIDACIÓN COMPLETADA ---\n")

    def _check_truth_consistency(self, files, context):
        """y_true_ret idéntico entre modelos por horizonte (fechas comunes)."""
        errors = []
        by_h = {}
        for f in files:
            df = pd.read_csv(f, usecols=['Date', 'h', 'y_true_ret'])
            h = df['h'].iloc[0]
            by_h.setdefault(h, []).append((os.path.basename(f), df))
        for h, items in by_h.items():
            ref_name, ref = items[0]
            for name, df in items[1:]:
                m = pd.merge(ref, df, on='Date', suffixes=('_ref', '_other'))
                if m.empty:
                    continue
                diff = (m['y_true_ret_ref'] - m['y_true_ret_other']).abs().max()
                if diff > 1e-10:
                    msg = (f"{name}: y_true_ret difiere de {ref_name} "
                           f"(h={h}, max diff={diff:.2e}) — desalineación")
                    errors.append(msg)
                    print(f"[ERROR] {msg}")
        if not errors:
            print(f">>> [Contract] y_true consistente entre modelos ({context}).")
        return errors

    def run_leakage_gate(self):
        """
        Gate anti-fuga + cordura (contrato §8): ejecuta la suite tests/
        (test_no_leakage.py y test_ytrue_sanity.py — verificación del y_true
        contra la fuente primaria) y detiene el pipeline si algo falla.
        """
        print("--- GATE ANTI-FUGA + CORDURA (tests/) ---")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-q"],
            capture_output=True, text=True
        )
        tail = (result.stdout or "").strip().splitlines()
        for line in tail[-5:]:
            print(f"   {line}")
        if result.returncode != 0:
            raise RuntimeError(
                "[Leakage GATE] El test anti-fuga FALLÓ. Pipeline detenido.\n"
                + (result.stdout or "") + (result.stderr or "")
            )
        print("--- GATE ANTI-FUGA: OK ---\n")

    def run_full_gate(self):
        """Gate completo del pipeline: estructura + contrato + anti-fuga."""
        self.validate_structure()
        self.validate_all_outputs(fail_fast=True)
        self.run_leakage_gate()

# Uso rápido desde línea de comandos
if __name__ == "__main__":
    ContractValidator().run_full_gate()
