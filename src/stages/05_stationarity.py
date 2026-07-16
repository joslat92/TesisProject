"""Generate the ADF/KPSS appendix table from the configured canonical data."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import yaml

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
    print(f">>> [ADF/KPSS] {len(result)} rows written to {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
