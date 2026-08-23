import importlib.util
from pathlib import Path

import numpy as np

from src.core.dm import diebold_mariano


ROOT = Path(__file__).resolve().parents[1]


def load_evaluation_stage():
    path = ROOT / "src" / "stages" / "20_evaluate_stats.py"
    spec = importlib.util.spec_from_file_location("stage_20_stats", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dm_identical_predictions_are_not_significant():
    y = np.linspace(-0.02, 0.03, 50)
    pred = np.zeros_like(y)
    assert diebold_mariano(y, pred, pred, h=1) == (0.0, 1.0, 0.0, 1.0)


def test_dm_constant_nonzero_loss_difference_is_deterministic():
    y = np.zeros(50)
    benchmark = np.ones(50)
    challenger = np.zeros(50)
    dm, p, dm_hln, p_hln = diebold_mariano(y, benchmark, challenger, h=1)
    assert np.isposinf(dm)
    assert np.isposinf(dm_hln)
    assert p == 0.0
    assert p_hln == 0.0


def test_mincer_zarnowitz_joint_null_for_exact_forecasts_with_noise():
    stage = load_evaluation_stage()
    rng = np.random.default_rng(20260822)
    pred = np.linspace(100.0, 200.0, 500)
    actual = pred + rng.normal(0.0, 0.5, len(pred))
    *_, p_joint = stage.mincer_zarnowitz_test(actual, pred, h=1)
    assert p_joint > 0.05


def test_mincer_zarnowitz_joint_rejects_biased_forecasts():
    stage = load_evaluation_stage()
    rng = np.random.default_rng(20260822)
    pred = np.linspace(100.0, 200.0, 500)
    actual = 20.0 + 0.7 * pred + rng.normal(0.0, 0.5, len(pred))
    *_, p_joint = stage.mincer_zarnowitz_test(actual, pred, h=1)
    assert p_joint < 1e-8
