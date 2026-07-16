"""Seal the rebuilt pipeline outputs with hashes and key reported results."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pandas as pd

from common import sha256_file, utc_now, write_json


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_TREES = (
    "data/processed",
    "outputs/preds",
    "reports/data",
    "reports/figs",
)


def tree_manifest(root: Path) -> dict:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        file_hash = sha256_file(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
        total_bytes += path.stat().st_size
    return {
        "files": len(files),
        "bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
    }


def scalar(df: pd.DataFrame, **filters):
    selected = df
    for column, value in filters.items():
        selected = selected[selected[column] == value]
    if len(selected) != 1:
        raise ValueError(f"Se esperaba una fila para {filters}, se obtuvieron {len(selected)}")
    return selected.iloc[0]


def parse_run_log(path: Path | None) -> dict:
    if path is None:
        return {"verified": False, "reason": "run log not provided"}
    text = path.read_text(encoding="utf-16", errors="replace")
    match = re.search(r"RUN_ALL_OK \(COMPLETO\) en ([0-9.]+) min", text)
    tests = re.findall(r"(\d+) passed in ([0-9.]+)s", text)
    return {
        "verified": match is not None,
        "minutes": float(match.group(1)) if match else None,
        "final_gate_passed": int(tests[-1][0]) if tests else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-log", type=Path)
    args = parser.parse_args()

    curated_manifest = json.loads(
        (ROOT / "data/manifests/curated_dataset.json").read_text(encoding="utf-8")
    )
    dataset_path = ROOT / curated_manifest["output_file"]
    dataset_hash = sha256_file(dataset_path)
    if dataset_hash != curated_manifest["output_sha256"]:
        raise ValueError("El dataset local no coincide con el manifiesto curado")

    metrics = pd.read_csv(ROOT / "reports/data/metrics_OOS.csv")
    metrics_wf = pd.read_csv(ROOT / "reports/data/metrics_WF.csv")
    dm = pd.read_csv(ROOT / "reports/data/tbl_DM_OOS.csv")
    multiseed = pd.read_csv(ROOT / "reports/data/multiseed_dm_T20_LSTM_FULL.csv")
    robust = pd.read_csv(ROOT / "reports/data/metrics_OOS_2025.csv")

    ms_seeds = multiseed[multiseed["Seed"].astype(str) != "ENSEMBLE"]
    ms_ensemble = scalar(multiseed, Seed="ENSEMBLE")
    robust_winners = {
        str(int(h)): group.loc[group["RMSE"].idxmin(), "Model"]
        for h, group in robust.groupby("Horizon")
    }

    payload = {
        "manifest_version": 1,
        "created_at_utc": utc_now(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dataset": {
            "path": curated_manifest["output_file"],
            "sha256": dataset_hash,
            "rows": curated_manifest["rows"],
        },
        "run": parse_run_log(ROOT / args.run_log if args.run_log else None),
        "artifact_trees": {
            relative: tree_manifest(ROOT / relative) for relative in ARTIFACT_TREES
        },
        "report_hashes": {
            path.name: sha256_file(path)
            for path in sorted((ROOT / "reports/data").glob("*.csv"))
        },
        "key_results": {
            "oos_2024_h20_rmse": {
                "RW": float(scalar(metrics, Horizon=20, Model="RW")["RMSE"]),
                "LSTM_FULL": float(
                    scalar(metrics, Horizon=20, Model="LSTM_FULL")["RMSE"]
                ),
            },
            "oos_2024_h20_lstm_full_vs_rw_p_hln": float(
                scalar(dm, Horizon=20, Challenger="LSTM_FULL", Benchmark="RW")["p_HLN"]
            ),
            "walkforward_h20_rmse": {
                "RW": float(scalar(metrics_wf, Horizon=20, Model="RW")["RMSE"]),
                "LSTM_FULL": float(
                    scalar(metrics_wf, Horizon=20, Model="LSTM_FULL")["RMSE"]
                ),
            },
            "multiseed_h20": {
                "median_p_value": float(ms_seeds["p_value"].median()),
                "seeds_p_below_0_05": int((ms_seeds["p_value"] < 0.05).sum()),
                "ensemble_p_value": float(ms_ensemble["p_value"]),
            },
            "robustness_2025_rmse_winner_by_horizon": robust_winners,
        },
    }

    output = ROOT / "data/manifests/reproduction_results.json"
    write_json(output, payload)
    print(f"OK: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
