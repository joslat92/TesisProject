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
import hashlib
import json
import os
import re
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
    ("05_stationarity.py", "Diagnóstico ADF/KPSS para el apéndice", False),
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


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_canonical_data(root=ROOT):
    """Falla antes de entrenar si la fuente no es el dataset sellado."""
    source = configured_data_source(root)
    if not source.exists():
        raise SystemExit(
            f"[FALLO] No existe el dataset configurado: {source}\n"
            "Construyalo siguiendo docs/construccion_datos.md antes de ejecutar el pipeline."
        )
    manifest_path = root / "data" / "manifests" / "curated_dataset.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed_hash = sha256_file(source)
    expected_hash = manifest["output_sha256"]
    if observed_hash != expected_hash:
        raise SystemExit(
            "[FALLO] El dataset configurado no coincide con el sello canónico.\n"
            f"Esperado: {expected_hash}\nObservado: {observed_hash}"
        )
    with open(source, "r", encoding="utf-8") as handle:
        observed_rows = sum(1 for _ in handle) - 1
    if observed_rows != manifest["rows"]:
        raise SystemExit(
            f"[FALLO] Filas del dataset: {observed_rows}; esperadas: {manifest['rows']}"
        )
    print(f">>> [preflight] dataset canónico verificado ({observed_rows} filas, SHA-256 OK)")
    return source


def collect_test_count(root=ROOT):
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "--collect-only", "-q"],
        cwd=root, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise SystemExit("[FALLO] No se pudo contar la suite final de pruebas")
    match = re.search(r"(\d+) tests? collected", result.stdout)
    if not match:
        raise SystemExit("[FALLO] pytest no informó el número de pruebas recolectadas")
    return int(match.group(1))


def seal_current_run(elapsed_minutes, final_gate_passed, root=ROOT):
    """Registra y sella los artefactos generados por esta misma corrida."""
    summary = root / "logs" / "reproduction_latest_summary.log"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        "# Resumen generado automáticamente por run_all.py\n\n"
        f"RUN_ALL_OK (COMPLETO) en {elapsed_minutes:.3f} min\n"
        f"Suite final: {final_gate_passed} passed\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "data" / "60_seal_reproduction.py"),
            "--verified-run",
            "--minutes", f"{elapsed_minutes:.6f}",
            "--final-gate-passed", str(final_gate_passed),
            "--run-summary", str(summary.relative_to(root)),
        ],
        cwd=root,
    )
    if result.returncode != 0:
        raise SystemExit("[FALLO] No se pudo sellar la corrida completa")


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
        "outputs", "reports", "logs", "tesis-quick-*",
        "*.docx", "*.pdf", "*.zip",
    )
    # Forzar el temporal FUERA de ROOT: en entornos con TMPDIR apuntando al
    # cwd, el comportamiento por defecto copiaría el sandbox dentro de sí.
    with tempfile.TemporaryDirectory(prefix="tesis-quick-", dir=ROOT.parent) as tmp:
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

    validate_canonical_data()

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

    elapsed_minutes = (time.time() - t0) / 60
    if not args.quick:
        final_gate_passed = collect_test_count()
        seal_current_run(elapsed_minutes, final_gate_passed)
    print(f"\n=== RUN_ALL_OK ({mode}) en {elapsed_minutes:.1f} min ===")
    print("Tablas: reports/data | Figuras: reports/figs | Preds: outputs/preds")

if __name__ == "__main__":
    main()
