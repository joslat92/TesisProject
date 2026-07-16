"""Generate the ADF/KPSS appendix table from the configured canonical data."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.stationarity import stationarity_table


def main() -> None:
    with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    source = ROOT / cfg["data"]["raw_source"]
    target = cfg["data"]["target_col"]
    data = pd.read_csv(source)
    if target not in data.columns:
        raise ValueError(f"Missing configured target column: {target}")

    result = stationarity_table(data[target], target)
    output = ROOT / "reports" / "data" / "stationarity_tests.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    figures = ROOT / "reports" / "figs"
    figures.mkdir(parents=True, exist_ok=True)
    level = pd.to_numeric(data[target], errors="raise").dropna()
    for name, plotter, title in (
        ("Fig_appendix_acf_level.png", plot_acf, "ACF - Precio NASDAQ-100 (nivel)"),
        ("Fig_appendix_pacf_level.png", plot_pacf, "PACF - Precio NASDAQ-100 (nivel)"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        plotter(level, lags=40, ax=ax)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(figures / name, dpi=180)
        plt.close(fig)

    print(
        f">>> [ADF/KPSS] {len(result)} rows and 2 correlograms written "
        "to reports/"
    )


if __name__ == "__main__":
    main()
