import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .common import PROCESSED_DATA_PATH, VAE_MODEL_PATH, VAE_OUTPUTS_DIR, ensure_exists, outputs_vae_path
from .dataset import SBERTVaeDataset
from .VAE import VAE, vae_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train the VAE on processed SBERT embeddings.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--free-bits", type=float, default=0.1)
    parser.add_argument("--anneal-epochs", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def train_model(
    model,
    train_dataset,
    test_dataset,
    epochs=50,
    batch_size=64,
    device="cuda",
    save_loss_path=None,
    beta=1.0,
    free_bits=0.0,
    use_anneal=True,
    anneal_start=0.0,
    anneal_end=1.0,
    anneal_epochs=10,
):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    model.to(device)

    history = {
        "train_loss": [],
        "test_loss": [],
        "train_kl": [],
        "kl_dim_list": [],
    }

    for epoch in range(epochs):
        model.train()
        kl_anneal = (
            anneal_start + (anneal_end - anneal_start) * min(epoch / max(1, anneal_epochs), 1)
            if use_anneal
            else 1.0
        )

        total_train_loss = 0.0
        total_train_kl = 0.0
        kl_dim_accum = None

        for x_b, _, _ in train_loader:
            x_b = x_b.to(device)

            recon, mu, logvar = model(x_b)
            loss, kl_per_dim = vae_loss(
                recon,
                x_b,
                mu,
                logvar,
                beta=beta,
                kl_anneal=kl_anneal,
                free_bits=free_bits,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_kl += kl_per_dim.sum().item()

            current_kl_dim = kl_per_dim.sum(dim=0).detach().cpu().numpy()
            kl_dim_accum = current_kl_dim if kl_dim_accum is None else kl_dim_accum + current_kl_dim

        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for x_val, _, _ in test_loader:
                x_val = x_val.to(device)
                recon_val, mu_val, logvar_val = model(x_val)
                v_loss, _ = vae_loss(
                    recon_val,
                    x_val,
                    mu_val,
                    logvar_val,
                    beta=beta,
                    kl_anneal=kl_anneal,
                    free_bits=free_bits,
                )
                total_test_loss += v_loss.item()

        avg_train_loss = total_train_loss / len(train_dataset)
        avg_test_loss = total_test_loss / len(test_dataset)
        avg_train_kl = total_train_kl / len(train_dataset)
        avg_kl_dim = kl_dim_accum / len(train_dataset)

        history["train_loss"].append(avg_train_loss)
        history["test_loss"].append(avg_test_loss)
        history["train_kl"].append(avg_train_kl)
        history["kl_dim_list"].append(avg_kl_dim)

        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
            f"Test Loss: {avg_test_loss:.4f} | KL: {avg_train_kl:.4f} | Anneal: {kl_anneal:.3f}"
        )

    if save_loss_path is not None:
        np.save(save_loss_path, np.array(history["train_loss"]))

    return model, history


def plot_results(history, save_path):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss", color="blue")
    plt.plot(history["test_loss"], label="Test Loss", color="orange", linestyle="--")
    plt.title("Training & Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_kl"], label="Avg KL Divergence", color="red")
    plt.title("KL Divergence Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("KL")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)


def main():
    args = parse_args()
    data_path = ensure_exists(PROCESSED_DATA_PATH, "Processed dataset")
    VAE_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset = SBERTVaeDataset(data_path, test_mode=False, test_size=0.2, random_seed=42)
    test_dataset = SBERTVaeDataset(data_path, test_mode=True, test_size=0.2, random_seed=42)

    vae = VAE(input_dim=768, h1=256, h2=128, h3=50)
    loss_path = outputs_vae_path("vae_loss.npy")
    plot_path = outputs_vae_path("vae_loss.png")

    print("\n==== Start Training VAE ====")
    vae, history = train_model(
        vae,
        train_dataset,
        test_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        save_loss_path=loss_path,
        beta=args.beta,
        free_bits=args.free_bits,
        use_anneal=True,
        anneal_epochs=args.anneal_epochs,
    )

    plot_results(history, plot_path)
    torch.save(vae.state_dict(), VAE_MODEL_PATH)

    print("Training complete!")
    print(f"Loss data saved to: {loss_path}")
    print(f"Loss curve saved to: {plot_path}")
    print(f"Model weights saved to: {VAE_MODEL_PATH}")


if __name__ == "__main__":
    main()
