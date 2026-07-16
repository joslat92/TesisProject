import run_all


def test_archive_generated_artifacts_preserves_and_recreates_trees(tmp_path):
    for relative in run_all.FRESH_DIRS:
        path = tmp_path / relative
        path.mkdir(parents=True)
        (path / "sentinel.txt").write_text(relative, encoding="utf-8")

    archive, moved = run_all.archive_generated_artifacts(
        root=tmp_path, timestamp="20260716T000000Z"
    )

    assert moved == list(run_all.FRESH_DIRS)
    for relative in run_all.FRESH_DIRS:
        assert (tmp_path / relative).is_dir()
        assert not any((tmp_path / relative).iterdir())
        assert (archive / relative / "sentinel.txt").read_text(encoding="utf-8") == relative
