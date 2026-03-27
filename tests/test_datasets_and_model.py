import json

import torch

from xai_for_stylistic_analysis.datasets import TfidfDataset
from xai_for_stylistic_analysis.vae.VAE import VAE, vae_loss
from xai_for_stylistic_analysis.vae.dataset import SBERTVaeDataset


def test_tfidf_dataset_splits_and_returns_expected_types(tmp_path):
    samples = [
        {"embedding": [float(i), float(i + 1), float(i + 2)], "label": 1 if i % 2 == 0 else 2, "id": i}
        for i in range(10)
    ]
    json_path = tmp_path / "tfidf.json"
    json_path.write_text(json.dumps(samples), encoding="utf-8")

    train_ds = TfidfDataset(json_path, test_mode=False, test_size=0.2, random_seed=7)
    test_ds = TfidfDataset(json_path, test_mode=True, test_size=0.2, random_seed=7)

    assert len(train_ds) == 8
    assert len(test_ds) == 2

    emb, label, sample_id = train_ds[0]
    assert emb.dtype == torch.float32
    assert emb.shape == (3,)
    assert label.dtype == torch.long
    assert sample_id in range(10)


def test_sbert_vae_dataset_can_return_joined_text_and_index(tmp_path):
    samples = [
        {
            "embedding": [float(i)] * 4,
            "idx": i,
            "label": 1 if i % 2 == 0 else 2,
            "segments": [f"segment-{i}", "tail"],
        }
        for i in range(10)
    ]
    json_path = tmp_path / "vae.json"
    json_path.write_text(json.dumps(samples), encoding="utf-8")

    train_ds = SBERTVaeDataset(json_path, test_mode=False, test_size=0.3, random_seed=11, return_idx=True)
    test_ds = SBERTVaeDataset(json_path, test_mode=True, test_size=0.3, random_seed=11, return_idx=True)

    assert len(train_ds) == 7
    assert len(test_ds) == 3

    emb, label, text, item_idx = train_ds[0]
    assert emb.dtype == torch.float32
    assert emb.shape == (4,)
    assert label.dtype == torch.long
    assert isinstance(text, str)
    assert "segment-" in text
    assert item_idx.dtype == torch.long


def test_vae_forward_pass_and_loss_are_finite():
    model = VAE(input_dim=8, h1=6, h2=4, h3=3, latent_dim=2)
    batch = torch.randn(4, 8)

    recon_x, mu, logvar = model(batch)
    loss, kl_per_dim = vae_loss(recon_x, batch, mu, logvar)

    assert recon_x.shape == batch.shape
    assert mu.shape == (4, 2)
    assert logvar.shape == (4, 2)
    assert kl_per_dim.shape == (4, 2)
    assert torch.isfinite(loss)
