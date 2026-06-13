"""
Apéndice B — Reproducción de punta a punta con un solo comando.

Ejecuta la cadena completa desde data/raw/data.csv hasta tablas (reports/data)
y figuras (reports/figs):

    00_prepare -> 10_baselines -> 11_train_arima -> 13_train_sarimax ->
    12_train_lstm -> 14_walkforward -> 15_multiseed_lstm -> 16_robustez_2025 ->
    24_wf_metrics -> 20_evaluate_stats (gate completo) -> 30_make_figures

Uso:
    python run_all.py            # reproducción COMPLETA (horas: WF 144 + multiseed 120)
    python run_all.py --quick    # verificación de mecánica (~15-25 min):
                                 #  - omite walk-forward (14) y multi-semilla (15)
                                 #  - LSTM con epochs=2 (parchea config.yaml y
                                 #    config_robustez2025.yaml, restaura al salir)
                                 # Los números --quick NO son los de la tesis; los
                                 # clásicos (RW/ARIMA/ARIMAX/SARIMAX) sí son exactos
                                 # porque no dependen de los epochs del LSTM.

El gate del pipeline (contrato + anti-fuga + cordura de y_true contra la fuente
primaria) corre dentro de 20_evaluate_stats y detiene todo si falla.
"""
import argparse
import shutil
import subprocess
import sys
import time
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STAGES = [
    ("00_prepare.py", "Preparación de datos (parquets por horizonte)", False),
    ("10_baselines.py", "Baseline RW", False),
    ("11_train_arima.py", "ARIMA iterado", False),
    ("13_train_sarimax.py", "ARIMAX/SARIMAX (exógenas congeladas)", False),
    ("12_train_lstm.py", "Variantes LSTM (OOS 2024)", False),
    ("14_walkforward.py", "Walk-forward 12 bloques x 7 modelos", True),
    ("15_multiseed_lstm.py", "Multi-semilla LSTM (120 entrenamientos)", True),
    ("16_robustez_2025.py", "Bloque de robustez 2025", False),
    ("24_wf_metrics.py", "Métricas walk-forward", False),
    ("20_evaluate_stats.py", "Evaluación consolidada + GATE completo", False),
    ("30_make_figures.py", "Figuras canónicas", False),
]

PATCH_CONFIGS = ["config.yaml", "config_robustez2025.yaml"]
QUICK_EPOCHS = 2

def patch_configs_quick():
    """Baja epochs del LSTM para la verificación rápida; devuelve backups."""
    backups = {}
    for name in PATCH_CONFIGS:
        path = ROOT / name
        backup = ROOT / (name + ".runall.bak")
        shutil.copy2(path, backup)
        backups[path] = backup
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["models"]["params"]["lstm"]["epochs"] = QUICK_EPOCHS
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        print(f"   [quick] {name}: lstm.epochs={QUICK_EPOCHS}")
    return backups

def restore_configs(backups):
    for path, backup in backups.items():
        shutil.move(str(backup), str(path))
    if backups:
        print("   [quick] configs restaurados")

def main():
    ap = argparse.ArgumentParser(description="Reproducción de punta a punta")
    ap.add_argument("--quick", action="store_true",
                    help="verificación rápida: sin WF/multiseed, LSTM epochs=2")
    args = ap.parse_args()

    mode = "QUICK (verificación de mecánica)" if args.quick else "COMPLETO"
    print(f"=== run_all: modo {mode} ===")
    t0 = time.time()
    backups = {}
    try:
        if args.quick:
            backups = patch_configs_quick()

        for filename, desc, skippable in STAGES:
            if args.quick and skippable:
                print(f"\n--- [SKIP --quick] {filename}: {desc}")
                continue
            print(f"\n--- {filename}: {desc}")
            t = time.time()
            result = subprocess.run(
                [sys.executable, str(ROOT / "src" / "stages" / filename)],
                cwd=ROOT,
            )
            if result.returncode != 0:
                print(f"\n[FALLO] {filename} terminó con código {result.returncode}. "
                      f"Cadena detenida.")
                sys.exit(result.returncode)
            print(f"    ({time.time()-t:.0f}s)")
    finally:
        restore_configs(backups)

    print(f"\n=== RUN_ALL_OK ({mode}) en {(time.time()-t0)/60:.1f} min ===")
    print("Tablas: reports/data | Figuras: reports/figs | Preds: outputs/preds")

if __name__ == "__main__":
    main()
