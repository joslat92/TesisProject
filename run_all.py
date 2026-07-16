"""
Apéndice B — Reproducción de punta a punta con un solo comando.

Ejecuta la cadena completa desde la fuente configurada hasta tablas (reports/data)
y figuras (reports/figs):

    00_prepare -> 10_baselines -> 11_train_arima -> 13_train_sarimax ->
    12_train_lstm -> 14_walkforward -> 15_multiseed_lstm -> 16_robustez_2025 ->
    24_wf_metrics -> 20_evaluate_stats (gate completo) -> 30_make_figures ->
    31_regimes

Uso:
    python run_all.py --fresh    # completa y archiva artefactos previos
    python run_all.py            # completa sin archivar (artefactos compatibles)
    python run_all.py --quick    # verificación de mecánica (~15-25 min):
                                 #  - corre en una copia temporal aislada
                                 #  - omite walk-forward (14), sus métricas (24)
                                 #    y multi-semilla (15)
                                 #  - LSTM con epochs=2 dentro de esa copia
                                 # Los números --quick NO son los de la tesis; los
                                 # clásicos (RW/ARIMA/ARIMAX/SARIMAX) sí son exactos
                                 # porque no dependen de los epochs del LSTM.
                                 # Ningún artefacto canónico se sobrescribe.

El gate del pipeline (contrato + anti-fuga + cordura de y_true contra la fuente
primaria) corre dentro de 20_evaluate_stats y detiene todo si falla.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
import yaml
from datetime import datetime, timezone
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
    ("24_wf_metrics.py", "Métricas walk-forward", True),
    ("20_evaluate_stats.py", "Evaluación consolidada + GATE completo", False),
    ("30_make_figures.py", "Figuras canónicas", False),
    ("31_regimes.py", "Regímenes de volatilidad (§9.4)", False),
]

PATCH_CONFIGS = ["config.yaml", "config_robustez2025.yaml"]
QUICK_EPOCHS = 2
QUICK_OUTPUT_DIRS = (
    "outputs/preds/OOS",
    "outputs/preds/WF",
    "reports/data",
    "reports/figs",
    "logs",
)
FRESH_DIRS = ("data/processed", "outputs", "reports")
FRESH_REQUIRED_DIRS = (
    "data/processed",
    "outputs/preds/OOS",
    "outputs/preds/WF",
    "outputs/preds/MULTISEED",
    "outputs/preds/OOS_2025",
    "outputs/preds/WF_2025",
    "reports/data",
    "reports/figs",
)


def configured_data_source(root=ROOT):
    with open(root / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return root / cfg["data"]["raw_source"]


def archive_generated_artifacts(root=ROOT, timestamp=None):
    """Archive generated trees before a clean full run; never delete them."""
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = root / "_run_archive" / stamp
    if archive.exists():
        raise FileExistsError(f"El archivo de corrida ya existe: {archive}")

    moved = []
    for relative in FRESH_DIRS:
        source = root / relative
        destination = archive / relative
        if source.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            moved.append(relative)
    for relative in FRESH_REQUIRED_DIRS:
        (root / relative).mkdir(parents=True, exist_ok=True)
    print(f">>> [fresh] artefactos previos archivados en {archive}")
    return archive, moved


def run_quick_isolated():
    """Ejecuta el smoke test en una copia temporal sin tocar artefactos sellados."""
    ignore = shutil.ignore_patterns(
        ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
        "outputs", "reports", "logs", "*.docx", "*.pdf", "*.zip",
    )
    with tempfile.TemporaryDirectory(prefix="tesis-quick-") as tmp:
        sandbox = Path(tmp) / ROOT.name
        print(f">>> [quick] creando copia aislada en {sandbox}")
        shutil.copytree(ROOT, sandbox, ignore=ignore)
        for relative_dir in QUICK_OUTPUT_DIRS:
            (sandbox / relative_dir).mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [sys.executable, str(sandbox / "run_all.py"),
             "--quick", "--quick-worker"],
            cwd=sandbox,
        )
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    print(">>> [quick] copia temporal eliminada; artefactos canónicos intactos")

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
    ap.add_argument("--quick-worker", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--fresh", action="store_true",
                    help="archiva outputs/reports/procesados/logs antes de la corrida completa")
    args = ap.parse_args()

    if args.quick and args.fresh:
        ap.error("--fresh no se combina con --quick; --quick ya corre aislado")

    source = configured_data_source()
    if not source.exists():
        raise SystemExit(
            f"[FALLO] No existe el dataset configurado: {source}\n"
            "Reconstruyalo siguiendo docs/reconstruccion_datos.md antes de ejecutar el pipeline."
        )

    if args.quick and not args.quick_worker:
        run_quick_isolated()
        return

    if args.fresh:
        archive_generated_artifacts()

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
            env = os.environ.copy()
            if args.quick:
                env["TESIS_QUICK_MODE"] = "1"
            result = subprocess.run(
                [sys.executable, str(ROOT / "src" / "stages" / filename)],
                cwd=ROOT,
                env=env,
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
