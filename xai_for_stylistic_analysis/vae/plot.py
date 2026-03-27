import argparse

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from .common import (
    PROCESSED_DATA_PATH,
    VAE_MODEL_PATH,
    VAE_OUTPUTS_DIR,
    ensure_exists,
    outputs_vae_path,
    resolve_pipeline_path,
)
from .dataset import SBERTVaeDataset
from .VAE import VAE


def parse_args():
    parser = argparse.ArgumentParser(description="Generate confusion matrix and latent distribution plots.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_exists(PROCESSED_DATA_PATH, "Processed dataset")
    ensure_exists(VAE_MODEL_PATH, "VAE weights")
    VAE_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_path = resolve_pipeline_path()

    vae = VAE(input_dim=768, h1=256, h2=128, h3=50).to(args.device)
    vae.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=args.device))
    vae.eval()

    bundle = joblib.load(pipeline_path)
    clf = bundle["stacking_ensemble"]
    scaler = bundle["scaler"]

    test_ds = SBERTVaeDataset(PROCESSED_DATA_PATH, test_mode=True, test_size=0.2)
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)

    with torch.no_grad():
        for emb, label, _ in test_loader:
            mu, logvar = vae.encode(emb.to(args.device))
            X_mu = mu.cpu().numpy()
            std = torch.exp(0.5 * logvar)
            X_latent_full = torch.cat([mu, std], dim=-1).cpu().numpy()
            y_true = label.numpy()
            break

    X_scaled = scaler.transform(X_latent_full)
    y_pred = clf.predict(X_scaled)
    label_names = ["Non-Translated", "Translated"]

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(outputs_vae_path("confusion_matrix.png"))
    plt.close()

    lda = LDA(n_components=1)
    X_lda = lda.fit_transform(X_mu, y_true)

    plt.figure(figsize=(10, 6))
    for idx, name in enumerate(label_names):
        sns.kdeplot(X_lda[y_true == idx].flatten(), label=name, fill=True, alpha=0.5, linewidth=2)

    plt.axvline(x=0, color="red", linestyle="--", alpha=0.5)
    plt.title("LDA: Discriminant Axis Distribution of Latent Space")
    plt.xlabel("LDA Component 1")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(axis="y", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outputs_vae_path("lda_distribution.png"))
    plt.close()

    print(
        f"Plots saved to: {outputs_vae_path('confusion_matrix.png')} and "
        f"{outputs_vae_path('lda_distribution.png')}"
    )


if __name__ == "__main__":
    main()
