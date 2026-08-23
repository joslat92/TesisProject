import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "data"))


def load_sealer():
    path = ROOT / "scripts" / "data" / "60_seal_reproduction.py"
    spec = importlib.util.spec_from_file_location("seal_reproduction", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tree_manifest_is_stable_and_content_sensitive(tmp_path):
    sealer = load_sealer()
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "two.txt").write_text("two", encoding="utf-8")
    (tmp_path / "one.txt").write_text("one", encoding="utf-8")

    first = sealer.tree_manifest(tmp_path)
    second = sealer.tree_manifest(tmp_path)
    assert first == second
    assert first["files"] == 2

    (tmp_path / "one.txt").write_text("changed", encoding="utf-8")
    assert sealer.tree_manifest(tmp_path)["tree_sha256"] != first["tree_sha256"]


def test_text_artifact_hash_is_portable_across_line_endings(tmp_path):
    sealer = load_sealer()
    lf = tmp_path / "lf.csv"
    crlf = tmp_path / "crlf.csv"
    lf.write_bytes(b"a,b\n1,2\n")
    crlf.write_bytes(b"a,b\r\n1,2\r\n")
    assert sealer.artifact_sha256(lf) == sealer.artifact_sha256(crlf)
