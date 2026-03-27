import types

import pytest

from xai_for_stylistic_analysis import cli


def test_cli_registry_contains_core_commands():
    expected_module_commands = {
        "fetch-sbert-model",
        "vae-preprocess",
        "vae-train",
        "vae-search",
        "vae-vote",
        "vae-final",
        "vae-logistic",
        "vae-perturb",
        "vae-plot",
        "vae-importance",
        "vae-sbert-search",
        "vae-sbert-vote",
        "vae-sbert-final",
    }
    expected_script_commands = {"tfidf-preprocess", "tfidf-vote"}

    assert expected_module_commands.issubset(cli.MODULE_MAP)
    assert expected_script_commands.issubset(cli.SCRIPT_MAP)


def test_print_check_emits_expected_sections(capsys):
    cli.print_check()

    captured = capsys.readouterr().out
    assert "Project check:" in captured
    assert "Processed TF-IDF data" in captured
    assert "Processed VAE data" in captured
    assert "VAE weights" in captured
    assert "Local SBERT model" in captured


def test_run_module_requires_main(monkeypatch):
    monkeypatch.setattr(cli.importlib, "import_module", lambda _: types.SimpleNamespace())

    with pytest.raises(AttributeError, match="does not expose a main"):
        cli.run_module("dummy.module")
