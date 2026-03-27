from pathlib import Path

import json
import joblib
import pytest
import torch

from xai_for_stylistic_analysis.datasets import TfidfDataset
from xai_for_stylistic_analysis.vae.VAE import VAE
from xai_for_stylistic_analysis.vae.common import PIPELINE_PATH, PROCESSED_DATA_PATH, ROOT_DIR, VAE_MODEL_PATH
from xai_for_stylistic_analysis.vae.dataset import SBERTVaeDataset


pytestmark = pytest.mark.integration


def test_core_artifacts_exist(project_root):
    expected_paths = [
        project_root / "data" / "processed" / "vae" / "processed_final.json",
        project_root / "data" / "processed" / "tfidf" / "processed_TFIDF_final.json",
        project_root / "data" / "processed" / "tfidf" / "feature_names.json",
        project_root / "outputs" / "vae" / "vae_model.pth",
        project_root / "outputs" / "vae" / "full_stacking_pipeline.pkl",
        project_root / "outputs" / "tfidf" / "full_stacking_pipeline.pkl",
    ]

    missing = [str(path) for path in expected_paths if not path.exists()]
    assert not missing, f"Missing core artifacts: {missing}"


def test_saved_vae_model_can_encode_real_sample():
    dataset = SBERTVaeDataset(PROCESSED_DATA_PATH, test_mode=True, test_size=0.2, random_seed=42)
    emb, label, text = dataset[0]

    model = VAE(input_dim=768, h1=256, h2=128, h3=50)
    state_dict = torch.load(VAE_MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        mu, logvar = model.encode(emb.unsqueeze(0))

    assert mu.shape == (1, 50)
    assert logvar.shape == (1, 50)
    assert label.item() in (0, 1)
    assert isinstance(text, str) and text


def test_saved_vae_pipeline_can_predict_real_latent_features():
    dataset = SBERTVaeDataset(PROCESSED_DATA_PATH, test_mode=True, test_size=0.2, random_seed=42)
    emb, _, _ = dataset[0]

    model = VAE(input_dim=768, h1=256, h2=128, h3=50)
    model.load_state_dict(torch.load(VAE_MODEL_PATH, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        mu, logvar = model.encode(emb.unsqueeze(0))
        std = torch.exp(0.5 * logvar)
        features = torch.cat([mu, std], dim=1).numpy()

    bundle = joblib.load(PIPELINE_PATH)
    X_scaled = bundle["scaler"].transform(features)
    prediction = bundle["stacking_ensemble"].predict(X_scaled)
    probabilities = bundle["stacking_ensemble"].predict_proba(X_scaled)

    assert prediction.shape == (1,)
    assert prediction[0] in (0, 1)
    assert probabilities.shape == (1, 2)


def test_tfidf_processed_data_matches_feature_names():
    tfidf_data_path = ROOT_DIR / "data" / "processed" / "tfidf" / "processed_TFIDF_final.json"
    feature_names_path = ROOT_DIR / "data" / "processed" / "tfidf" / "feature_names.json"

    dataset = TfidfDataset(tfidf_data_path, test_mode=True, test_size=0.2, random_seed=42)
    emb, label, _ = dataset[0]
    feature_names = json.loads(feature_names_path.read_text(encoding="utf-8"))

    assert emb.shape[0] == len(feature_names)
    assert torch.count_nonzero(emb).item() > 0
    assert label.item() in (0, 1)
