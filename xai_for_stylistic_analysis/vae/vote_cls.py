import argparse
from itertools import combinations

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader
from xgboost import XGBClassifier

from .common import PROCESSED_DATA_PATH, VAE_MODEL_PATH, ensure_exists
from .dataset import SBERTVaeDataset
from .VAE import VAE


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate 3-model stacking combinations on VAE features.")
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

    return X_train_scaled, X_test_scaled, y_train, y_test


def main():
    args = parse_args()
    ensure_exists(PROCESSED_DATA_PATH, "Processed dataset")
    ensure_exists(VAE_MODEL_PATH, "VAE weights")

    X_train_scaled, X_test_scaled, y_train, y_test = get_data_ready(args.device)

    best_models = {
        "SVM": SVC(C=20, kernel="rbf", gamma="scale", probability=True, class_weight="balanced"),
        "RF": RandomForestClassifier(
            n_estimators=1000,
            max_depth=20,
            min_samples_split=10,
            bootstrap=False,
            random_state=42,
        ),
        "XGB": XGBClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=3, weights="uniform", metric="manhattan"),
        "DT": DecisionTreeClassifier(max_depth=5, criterion="gini", min_samples_leaf=12, random_state=42),
    }

    results = []
    for combo in combinations(best_models.keys(), 3):
        estimators = [(name, best_models[name]) for name in combo]
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            cv=5,
            n_jobs=-1,
        )

        stacking_clf.fit(X_train_scaled, y_train)
        y_pred = stacking_clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        results.append({"combo": combo, "accuracy": acc, "clf": stacking_clf})
        print(f"combo {combo} | test acc: {acc:.4f}")

    results.sort(key=lambda item: item["accuracy"], reverse=True)
    best_ensemble = results[0]
    print("\n" + "=" * 60)
    print(f"best combo: {best_ensemble['combo']}")
    print(f"final test acc: {best_ensemble['accuracy']:.4f}")
    print("=" * 60)

    y_pred_best = best_ensemble["clf"].predict(X_test_scaled)
    print(classification_report(y_test, y_pred_best, digits=4))

    meta_model = best_ensemble["clf"].final_estimator_
    print("\nmeta-model coefficients:")
    for name, coef in zip(best_ensemble["combo"], meta_model.coef_[0]):
        print(f"{name}: {coef:.4f}")


if __name__ == "__main__":
    main()
