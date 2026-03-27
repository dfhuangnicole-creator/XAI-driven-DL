import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from .common import PROCESSED_DATA_PATH, VAE_MODEL_PATH, VAE_OUTPUTS_DIR, ensure_exists, outputs_vae_path
from .final_cls import get_data_ready


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ensure_exists(PROCESSED_DATA_PATH, "Processed dataset")
    ensure_exists(VAE_MODEL_PATH, "VAE weights")
    VAE_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train_scaled, X_test_scaled, y_train, y_test, _ = get_data_ready(DEVICE)

    model = LogisticRegression(max_iter=100, C=1, random_state=42, verbose=1)
    print("Starting logistic regression baseline training on VAE latent features...")
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    report = classification_report(y_test, y_pred)

    report_filename = outputs_vae_path("log_classification_report.txt")
    with report_filename.open("w", encoding="utf-8") as f:
        f.write("Logistic Regression Classification Report (VAE Latent Features)\n")
        f.write("=" * 50 + "\n")
        f.write(report)

    print(f"\nReport has been saved to: {report_filename}")


if __name__ == "__main__":
    main()
