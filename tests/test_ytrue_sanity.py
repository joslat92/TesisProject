"""
Verificación independiente de cordura del y_true (sellado RC2).

Recomputa A MANO el retorno logarítmico acumulado desde la fuente configurada
—solo pandas/numpy sobre el CSV, sin pasar por ningún módulo del pipeline—
y lo compara contra y_true_ret de los archivos de predicciones de TODOS los
modelos: 10 fechas aleatorias del OOS 2024 y 5 del bloque 2025, por horizonte.

y_manual(t, h) = log(P[fila_t + h]) − log(P[fila_t]), con P = Target_Price
sobre las filas (días hábiles) del CSV ordenado por fecha.

Tolerancia: 1e-10 absoluta en retornos. También verifica y_true_level contra
P[fila_t + h]. Muestreo con semilla fija (reproducible). Forma parte de la
suite del gate del pipeline (ContractValidator.run_leakage_gate).
"""
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [1, 5, 10, 20]
RNG_SEED = 20260612

def _load_raw():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    raw = pd.read_csv(ROOT / cfg["data"]["raw_source"], parse_dates=["Date"])
    raw = raw.sort_values("Date").reset_index(drop=True)
    logp = np.log(raw["Target_Price"].to_numpy())
    pos = {d: i for i, d in enumerate(raw["Date"])}
    return raw, logp, pos

def _check_dir(preds_dir, n_dates):
    raw, logp, pos = _load_raw()
    rng = np.random.default_rng(RNG_SEED)
    checked_files = 0

    for h in HORIZONS:
        files = sorted(glob.glob(str(preds_dir / f"preds_T{h}_*.csv")))
        assert files, f"No hay archivos preds_T{h}_*.csv en {preds_dir}"

        # Universo de fechas del horizonte (todas los archivos comparten fechas;
        # la consistencia entre archivos la cubre el ContractValidator)
        dates = pd.read_csv(files[0], parse_dates=["Date"])["Date"]
        sample = rng.choice(len(dates), size=min(n_dates, len(dates)),
                            replace=False)
        sample_dates = dates.iloc[sample].tolist()

        # Verdad recomputada a mano desde el CSV crudo
        manual = {}
        for t in sample_dates:
            i = pos[t]
            assert i + h < len(logp), f"{t} + {h} días excede el raw"
            manual[t] = (logp[i + h] - logp[i],
                         float(np.exp(logp[i + h])))

        for f in files:
            df = pd.read_csv(f, parse_dates=["Date"]).set_index("Date")
            for t in sample_dates:
                got_ret = df.loc[t, "y_true_ret"]
                exp_ret, exp_level = manual[t]
                assert abs(got_ret - exp_ret) < 1e-10, (
                    f"{os.path.basename(f)} @ {t.date()}: y_true_ret={got_ret} "
                    f"vs recomputado={exp_ret} (diff={abs(got_ret-exp_ret):.2e})"
                )
                got_level = df.loc[t, "y_true_level"]
                assert np.isclose(got_level, exp_level, rtol=1e-9, atol=1e-6), (
                    f"{os.path.basename(f)} @ {t.date()}: y_true_level={got_level} "
                    f"vs P[t+h]={exp_level}"
                )
            checked_files += 1
    return checked_files

def test_ytrue_oos_2024_contra_fuente_primaria():
    preds_dir = ROOT / "outputs" / "preds" / "OOS"
    if not any(preds_dir.glob("preds_T*.csv")):
        pytest.skip("aún no se generó el OOS 2024")
    n = _check_dir(preds_dir, n_dates=10)
    assert n >= 4, "se esperaban archivos para los 4 horizontes"

def test_ytrue_bloque_2025_contra_fuente_primaria():
    preds_dir = ROOT / "outputs" / "preds" / "OOS_2025"
    if not any(preds_dir.glob("preds_T*.csv")):
        pytest.skip("aún no se generó el bloque 2025")
    n = _check_dir(preds_dir, n_dates=5)
    assert n >= 4, "se esperaban archivos para los 4 horizontes"

if __name__ == "__main__":
    print("OOS 2024:", _check_dir(ROOT / "outputs" / "preds" / "OOS", 10), "archivos OK")
    print("2025:", _check_dir(ROOT / "outputs" / "preds" / "OOS_2025", 5), "archivos OK")
    print("OK: y_true verificado contra fuente primaria")
