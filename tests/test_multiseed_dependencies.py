import importlib.util
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_multiseed_stage():
    path = ROOT / "src" / "stages" / "15_multiseed_lstm.py"
    spec = importlib.util.spec_from_file_location("stage_15_multiseed", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rw_rmse_does_not_require_consolidated_metrics(tmp_path):
    stage = load_multiseed_stage()
    preds_dir = tmp_path / "outputs" / "preds" / "OOS"
    preds_dir.mkdir(parents=True)
    pd.DataFrame({
        "y_true_ret": [0.1, -0.2],
        "y_pred_ret": [0.0, 0.0],
    }).to_csv(preds_dir / "preds_T1_RW.csv", index=False)

    cfg = {
        "paths": {"preds_oos_dir": "outputs/preds/OOS"},
        "contract": {"naming": {"oos": "preds_T{h}_{model}.csv"}},
    }

    result = stage.rw_rmse_from_predictions(cfg, [1], root=str(tmp_path))

    assert result.loc[1] == pytest.approx(0.025 ** 0.5)
    assert not (tmp_path / "reports" / "data" / "metrics_OOS.csv").exists()
