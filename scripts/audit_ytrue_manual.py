"""
Auditoria independiente de y_true para RC2.1.

Usa solo pandas/numpy y archivos CSV. No importa modulos del proyecto.
Recomputa y_true_ret = log(P[t+h]) - log(P[t]) desde data/data.csv y lo
compara contra todos los archivos de predicciones OOS 2024 y OOS_2025.
"""
from pathlib import Path
import glob
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [1, 5, 10, 20]
PERIODS = {
    "OOS": 10,
    "OOS_2025": 5,
}
SEED = 20260616
TOL = 1e-10


def load_raw():
    raw = pd.read_csv(ROOT / "data" / "data.csv", parse_dates=["Date"])
    raw = raw.sort_values("Date").reset_index(drop=True)
    logp = np.log(raw["Target_Price"].to_numpy())
    price = raw["Target_Price"].to_numpy(dtype=float)
    pos = {date: idx for idx, date in enumerate(raw["Date"])}
    return raw, logp, price, pos


def model_name(path, h):
    name = Path(path).stem
    return re.sub(rf"^preds_T{h}_", "", name)


def main():
    raw, logp, price, pos = load_raw()
    rng = np.random.default_rng(SEED)
    failures = []
    rows = []

    for period, n_dates in PERIODS.items():
        pred_dir = ROOT / "outputs" / "preds" / period
        for h in HORIZONS:
            files = sorted(glob.glob(str(pred_dir / f"preds_T{h}_*.csv")))
            if not files:
                failures.append(f"Sin archivos para {period} h={h}")
                continue

            date_universe = pd.read_csv(files[0], parse_dates=["Date"])["Date"]
            sample_idx = rng.choice(
                len(date_universe), size=min(n_dates, len(date_universe)), replace=False
            )
            sample_dates = date_universe.iloc[sample_idx].sort_values().tolist()

            manual = {}
            for date in sample_dates:
                i = pos[date]
                if i + h >= len(raw):
                    failures.append(f"{period} h={h} {date.date()}: excede raw")
                    continue
                manual[date] = (logp[i + h] - logp[i], price[i + h])

            for path in files:
                df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
                max_ret_diff = 0.0
                max_level_diff = 0.0
                for date, (ret_exp, level_exp) in manual.items():
                    ret_got = float(df.loc[date, "y_true_ret"])
                    level_got = float(df.loc[date, "y_true_level"])
                    ret_diff = abs(ret_got - ret_exp)
                    level_diff = abs(level_got - level_exp)
                    max_ret_diff = max(max_ret_diff, ret_diff)
                    max_level_diff = max(max_level_diff, level_diff)
                    if ret_diff > TOL:
                        failures.append(
                            f"{Path(path).name} {date.date()}: ret diff {ret_diff:.3e}"
                        )
                    if not np.isclose(level_got, level_exp, rtol=1e-9, atol=1e-6):
                        failures.append(
                            f"{Path(path).name} {date.date()}: level diff {level_diff:.3e}"
                        )
                rows.append(
                    {
                        "period": period,
                        "h": h,
                        "model": model_name(path, h),
                        "n_dates": len(manual),
                        "max_abs_ret_diff": max_ret_diff,
                        "max_abs_level_diff": max_level_diff,
                    }
                )

    summary = pd.DataFrame(rows).sort_values(["period", "h", "model"])
    print(summary.to_string(index=False))
    print(f"\nraw_rows={len(raw)} checked_files={len(rows)} seed={SEED} tol={TOL}")
    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("\nOK: y_true_ret/y_true_level coincide contra data/data.csv")


if __name__ == "__main__":
    main()
