import argparse
import datetime

import joblib
import torch
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .common import PROCESSED_DATA_PATH, VAE_OUTPUTS_DIR, ensure_exists, outputs_vae_path
from .dataset import SBERTVaeDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train the final stacking classifier directly on SBERT embeddings.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def get_data_ready():
    train_dataset = SBERTVaeDataset(PROCESSED_DATA_PATH, test_mode=False, test_size=0.2, random_seed=42)
    test_dataset = SBERTVaeDataset(PROCESSED_DATA_PATH, test_mode=True, test_size=0.2, random_seed=42)

    X_train = train_dataset.embeddings.cpu().numpy()
    y_train = train_dataset.labels.cpu().numpy()
    X_test = test_dataset.embeddings.cpu().numpy()
    y_test = test_dataset.labels.cpu().numpy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def main():
    parse_args()
    ensure_exists(PROCESSED_DATA_PATH, "Processed dataset")
    VAE_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train_scaled, X_test_scaled, y_train, y_test, scaler = get_data_ready()

    estimators = [
        ("svm", SVC(C=0.1, kernel="linear", gamma="scale", probability=True, class_weight="balanced")),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=1000,
                max_depth=20,
                min_samples_split=5,
                bootstrap=False,
                random_state=42,
            ),
        ),
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
    report_content = f"Accuracy: {acc:.4f}\n\nReport:\n{report}"
    stable_report_path = outputs_vae_path("sbert_stacking_report.txt")
    timestamped_report_path = outputs_vae_path(f"sbert_stacking_report_{timestamp}.txt")
    for path in (stable_report_path, timestamped_report_path):
        path.write_text(report_content, encoding="utf-8")

    model_bundle = {
        "stacking_ensemble": final_stack_model,
        "scaler": scaler,
        "base_model_names": [name for name, _ in estimators],
    }
    stable_pipeline_path = outputs_vae_path("full_sbert_stacking_pipeline.pkl")
    timestamped_pipeline_path = outputs_vae_path(f"full_sbert_stacking_pipeline_{timestamp}.pkl")
    for path in (stable_pipeline_path, timestamped_pipeline_path):
        joblib.dump(model_bundle, path)

    print(f"Saved stable SBERT pipeline to: {stable_pipeline_path}")


if __name__ == "__main__":
    main()
