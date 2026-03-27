import argparse
import datetime

import joblib
import numpy as np
import torch
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader

from .common import PROCESSED_DATA_PATH, VAE_MODEL_PATH, VAE_OUTPUTS_DIR, ensure_exists, outputs_vae_path
from .dataset import SBERTVaeDataset
from .VAE import VAE


def parse_args():
    parser = argparse.ArgumentParser(description="Train the final VAE-based stacking classifier.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def get_data_ready(device):
    vae = VAE(input_dim=768, h1=256, h2=128, h3=50).to(device)
    vae.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=device))
    vae.eval()

    train_ds = SBERTVaeDataset(PROCESSED_DATA_PATH, test_mode=False, test_size=0.2, random_seed=42)
    test_ds = SBERTVaeDataset(PROCESSED_DATA_PATH, test_mode=True, test_size=0.2, random_seed=42)

    def extract(loader):
        X, y = [], []
        with torch.no_grad():
            for emb, lb, _ in loader:
                mu, logvar = vae.encode(emb.to(device))
                std = torch.exp(0.5 * logvar)
                combined_features = torch.cat([mu, std], dim=-1)
                X.append(combined_features.cpu().numpy())
                y.append(lb.numpy())
        return np.vstack(X), np.concatenate(y)

    X_train, y_train = extract(DataLoader(train_ds, batch_size=32))
    X_test, y_test = extract(DataLoader(test_ds, batch_size=32))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def main():
    args = parse_args()
    ensure_exists(PROCESSED_DATA_PATH, "Processed dataset")
    ensure_exists(VAE_MODEL_PATH, "VAE weights")
    VAE_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train_scaled, X_test_scaled, y_train, y_test, scaler = get_data_ready(args.device)

    estimators = [
        ("svm", SVC(C=20, kernel="rbf", gamma="scale", probability=True, class_weight="balanced")),
        ("dt", DecisionTreeClassifier(max_depth=5, criterion="gini", min_samples_leaf=12, random_state=42)),
        ("knn", KNeighborsClassifier(n_neighbors=3, weights="uniform", metric="manhattan")),
    ]

    final_stack_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        passthrough=False,
        cv=5,
        n_jobs=-1,
    )

    final_stack_model.fit(X_train_scaled, y_train)

    y_pred = final_stack_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    print("\n" + "=" * 60)
    print(f"best acc {acc:.4f}")
    print("=" * 60)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stable_report_path = outputs_vae_path("stacking_report.txt")
    timestamped_report_path = outputs_vae_path(f"stacking_report_{timestamp}.txt")
    report_content = f"Accuracy: {acc:.4f}\n\nReport:\n{report}"
    for path in (stable_report_path, timestamped_report_path):
        path.write_text(report_content, encoding="utf-8")

    model_bundle = {
        "stacking_ensemble": final_stack_model,
        "scaler": scaler,
        "base_model_names": [name for name, _ in estimators],
    }

    stable_pipeline_path = outputs_vae_path("full_stacking_pipeline.pkl")
    timestamped_pipeline_path = outputs_vae_path(f"full_stacking_pipeline_{timestamp}.pkl")
    for path in (stable_pipeline_path, timestamped_pipeline_path):
        joblib.dump(model_bundle, path)

    print(f"Saved stable pipeline to: {stable_pipeline_path}")
    print(f"Saved timestamped pipeline to: {timestamped_pipeline_path}")
    print(f"Saved report to: {stable_report_path}")


if __name__ == "__main__":
    main()
