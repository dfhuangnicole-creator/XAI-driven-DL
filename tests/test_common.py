import os
from pathlib import Path

import pytest

from xai_for_stylistic_analysis.datasets import _normalize_label
from xai_for_stylistic_analysis.vae import common


@pytest.mark.parametrize(
    ("value", "dataset_zero_based", "storage_value"),
    [
        (False, 0, 1),
        (True, 1, 2),
        (0, 0, 1),
        (1, 1, 1),
        (2, 1, 2),
        ("non-translated", 0, 1),
        ("non_translated_e", 0, 1),
        ("translated", 1, 2),
        ("translated_e", 1, 2),
    ],
)
def test_label_normalization_variants(value, dataset_zero_based, storage_value):
    assert _normalize_label(value) == dataset_zero_based
    assert common.normalize_storage_label(value) == storage_value

    # Storage labels use 1/2, while dataset labels use 0/1.
    if storage_value in (1, 2):
        assert common.to_zero_based_label(storage_value) == storage_value - 1

    if isinstance(value, str) or isinstance(value, bool) or value == 0:
        assert common.to_zero_based_label(value) == dataset_zero_based


def test_label_normalization_rejects_unknown_values():
    with pytest.raises(ValueError):
        _normalize_label("maybe")

    with pytest.raises(ValueError):
        common.normalize_storage_label("maybe")


def test_resolve_pipeline_path_prefers_stable_file(monkeypatch, tmp_path):
    stable = tmp_path / "full_stacking_pipeline.pkl"
    stable.write_bytes(b"stable")
    latest = tmp_path / "full_stacking_pipeline_20260101_000000.pkl"
    latest.write_bytes(b"latest")

    monkeypatch.setattr(common, "VAE_OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(common, "PIPELINE_PATH", stable)

    assert common.resolve_pipeline_path() == stable


def test_resolve_pipeline_path_falls_back_to_latest_timestamped_file(monkeypatch, tmp_path):
    older = tmp_path / "full_stacking_pipeline_20260101_000000.pkl"
    newer = tmp_path / "full_stacking_pipeline_20260102_000000.pkl"
    older.write_bytes(b"older")
    newer.write_bytes(b"newer")
    older.touch()
    newer.touch()

    # Make the "newer" file unambiguously newer for filesystems with coarse mtime resolution.
    older_mtime = 1_700_000_000
    newer_mtime = older_mtime + 10
    os.utime(older, (older_mtime, older_mtime))
    os.utime(newer, (newer_mtime, newer_mtime))

    monkeypatch.setattr(common, "VAE_OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(common, "PIPELINE_PATH", tmp_path / "full_stacking_pipeline.pkl")

    assert common.resolve_pipeline_path() == newer


def test_describe_missing_reports_only_absent_paths(tmp_path):
    present = tmp_path / "present.txt"
    missing = tmp_path / "missing.txt"
    present.write_text("ok", encoding="utf-8")

    description = common.describe_missing(
        [
            (present, "Present file"),
            (missing, "Missing file"),
        ]
    )

    assert "Missing file" in description
    assert str(missing) in description
    assert "Present file" not in description
