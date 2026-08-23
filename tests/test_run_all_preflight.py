import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_run_all():
    spec = importlib.util.spec_from_file_location("run_all_module", ROOT / "run_all.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_fixture(tmp_path):
    source = tmp_path / "data" / "curated" / "model_input_ndx.csv"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"Date,Target_Price\n2024-01-01,100\n")
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"data": {"raw_source": "data/curated/model_input_ndx.csv"}}),
        encoding="utf-8",
    )
    manifest_dir = tmp_path / "data" / "manifests"
    manifest_dir.mkdir(parents=True)
    manifest = {
        "output_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "rows": 1,
    }
    (manifest_dir / "curated_dataset.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return source, manifest_dir / "curated_dataset.json"


def test_preflight_accepts_exact_canonical_dataset(tmp_path):
    run_all = load_run_all()
    source, _ = make_fixture(tmp_path)
    assert run_all.validate_canonical_data(tmp_path) == source


def test_preflight_rejects_changed_dataset(tmp_path):
    run_all = load_run_all()
    source, _ = make_fixture(tmp_path)
    source.write_bytes(source.read_bytes() + b"2024-01-02,101\n")
    with pytest.raises(SystemExit, match="no coincide con el sello canónico"):
        run_all.validate_canonical_data(tmp_path)
