from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable


VAE_DIR = Path(__file__).resolve().parent
ROOT_DIR = VAE_DIR.parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ASSETS_DIR = ROOT_DIR / "assets"
MODELS_DIR = ASSETS_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
VAE_OUTPUTS_DIR = OUTPUTS_DIR / "vae"
TFIDF_OUTPUTS_DIR = OUTPUTS_DIR / "tfidf"
REPORTS_DIR = OUTPUTS_DIR / "reports"

RAW_INPUT_FILE = RAW_DATA_DIR / "data.xlsx"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "vae" / "processed_final.json"
VAE_MODEL_PATH = VAE_OUTPUTS_DIR / "vae_model.pth"
PIPELINE_PATH = VAE_OUTPUTS_DIR / "full_stacking_pipeline.pkl"
SBERT_MODEL_DIR = MODELS_DIR / "all-mpnet-base-v2"
SBERT_REPO_ID = "sentence-transformers/all-mpnet-base-v2"
SBERT_REQUIRED_FILES = (
    "1_Pooling/config.json",
    "config_sentence_transformers.json",
    "config.json",
    "data_config.json",
    "modules.json",
    "pytorch_model.bin",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "train_script.py",
    "vocab.txt",
)


def vae_path(*parts: str) -> Path:
    return VAE_DIR.joinpath(*parts)


def root_path(*parts: str) -> Path:
    return ROOT_DIR.joinpath(*parts)


def outputs_vae_path(*parts: str) -> Path:
    return VAE_OUTPUTS_DIR.joinpath(*parts)


def ensure_exists(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def latest_matching_file(pattern: str) -> Path | None:
    matches = sorted(VAE_OUTPUTS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def resolve_pipeline_path() -> Path:
    if PIPELINE_PATH.exists():
        return PIPELINE_PATH

    latest = latest_matching_file("full_stacking_pipeline_*.pkl")
    if latest is None:
        raise FileNotFoundError(
            "No stacking pipeline found. Run `uv run xai-style vae-final` first to generate one."
        )
    return latest


def resolve_report_targets(prefix: str, suffix: str) -> tuple[Path, Path]:
    stable = outputs_vae_path(f"{prefix}.{suffix}")
    return stable, outputs_vae_path(prefix)


def sbert_model_ready() -> bool:
    return all((SBERT_MODEL_DIR / rel_path).exists() for rel_path in SBERT_REQUIRED_FILES)


def ensure_sbert_model(force_download: bool = False) -> Path:
    if sbert_model_ready() and not force_download:
        return SBERT_MODEL_DIR

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download the SBERT model. "
            "Run `uv sync` and try again."
        ) from exc

    SBERT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    action = "Refreshing" if force_download else "Downloading"
    print(f"{action} SBERT model from {SBERT_REPO_ID} ...")

    for rel_path in SBERT_REQUIRED_FILES:
        target_path = SBERT_MODEL_DIR / rel_path
        if target_path.exists() and not force_download:
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        cached_path = Path(hf_hub_download(repo_id=SBERT_REPO_ID, filename=rel_path))
        shutil.copy2(cached_path, target_path)
        print(f"  fetched {rel_path}")

    return SBERT_MODEL_DIR


def normalize_storage_label(value: object) -> int:
    if isinstance(value, bool):
        return 2 if value else 1

    if isinstance(value, (int, float)):
        numeric = int(value)
        if numeric in (1, 2):
            return numeric
        if numeric in (0, 1):
            return numeric + 1

    normalized = str(value).strip().lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())

    mapping = {
        "non translated": 1,
        "non translated e": 1,
        "nontranslated": 1,
        "nontranslated e": 1,
        "translated": 2,
        "translated e": 2,
    }
    if normalized in mapping:
        return mapping[normalized]

    if normalized in {"1", "2"}:
        return int(normalized)

    raise ValueError(f"Unsupported label value: {value!r}")


def to_zero_based_label(value: object) -> int:
    storage_value = normalize_storage_label(value)
    return storage_value - 1


def describe_missing(paths: Iterable[tuple[Path, str]]) -> str:
    missing = [f"{description}: {path}" for path, description in paths if not path.exists()]
    return "\n".join(missing)
