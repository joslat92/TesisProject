from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.core.contract import ContractValidator


REQUIRED = [
    "Date", "h", "model", "y_true_ret", "y_pred_ret",
    "y_true_level", "y_pred_level",
]


def _config(tmp_path):
    cfg = {
        "data": {"splits": {"wf_blocks": 1}},
        "features": {"horizons": [1]},
        "models": {"active_models": ["RW", "ARIMA"]},
        "paths": {
            "preds_oos_dir": str(tmp_path / "OOS"),
            "preds_wf_dir": str(tmp_path / "WF"),
            "figures_dir": str(tmp_path / "figs"),
        },
        "contract": {
            "naming": {
                "oos": "preds_T{h}_{model}.csv",
                "wf": "preds_T{h}_{model}_block{b}.csv",
            },
            "columns_required": REQUIRED,
            "columns_wf_extra": ["block"],
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (tmp_path / "OOS").mkdir()
    (tmp_path / "WF").mkdir()
    return path


def _prediction(path: Path, model="RW", h=1):
    pd.DataFrame({
        "Date": ["2024-01-02", "2024-01-03"],
        "h": [h, h],
        "model": [model, model],
        "y_true_ret": [0.01, -0.02],
        "y_pred_ret": [0.0, 0.0],
        "y_true_level": [101.0, 99.0],
        "y_pred_level": [100.0, 101.0],
    }).to_csv(path, index=False)


def test_contract_rejects_missing_expected_artifact(tmp_path):
    cfg_path = _config(tmp_path)
    _prediction(tmp_path / "OOS" / "preds_T1_RW.csv")

    with pytest.raises(RuntimeError, match="preds_T1_ARIMA.csv"):
        ContractValidator(cfg_path).validate_all_outputs(
            require_complete=True, require_wf=False)


def test_contract_rejects_filename_content_mismatch(tmp_path):
    cfg_path = _config(tmp_path)
    path = tmp_path / "OOS" / "preds_T1_RW.csv"
    _prediction(path, model="ARIMA")

    with pytest.raises(ValueError, match="contradice su contenido"):
        ContractValidator(cfg_path).validate_prediction_file(path)


def test_contract_rejects_nan_in_truth(tmp_path):
    cfg_path = _config(tmp_path)
    path = tmp_path / "OOS" / "preds_T1_RW.csv"
    _prediction(path)
    df = pd.read_csv(path)
    df.loc[0, "y_true_ret"] = float("nan")
    df.to_csv(path, index=False)

    with pytest.raises(ValueError, match="NaN"):
        ContractValidator(cfg_path).validate_prediction_file(path)


def test_contract_accepts_complete_oos_set(tmp_path):
    cfg_path = _config(tmp_path)
    _prediction(tmp_path / "OOS" / "preds_T1_RW.csv", model="RW")
    _prediction(tmp_path / "OOS" / "preds_T1_ARIMA.csv", model="ARIMA")

    ContractValidator(cfg_path).validate_all_outputs(
        require_complete=True, require_wf=False)
