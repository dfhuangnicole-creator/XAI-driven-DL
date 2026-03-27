import argparse

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader
from xgboost import XGBClassifier

from .common import PROCESSED_DATA_PATH, VAE_MODEL_PATH, ensure_exists
from .dataset import SBERTVaeDataset
from .VAE import VAE


def parse_args():
    parser = argparse.ArgumentParser(description="Search the best single classifier on VAE latent features.")
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

    param_grids = {
        "SVM": {
            "model": SVC(probability=True, cache_size=1000),
            "params": {
                "C": [0.1, 1, 5, 10, 20, 50, 80, 100],
                "kernel": ["rbf", "poly", "linear"],
                "gamma": ["scale", "auto", 0.01, 0.1],
                "degree": [1, 2, 3],
            },
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 80, 100, 300, 500, 1000],
                "max_depth": [5, 10, 20, 30, None],
                "min_samples_split": [2, 5, 10],
                "bootstrap": [True, False],
            },
        },
        "XGBoost": {
            "model": XGBClassifier(eval_metric="logloss", random_state=42),
            "params": {
                "n_estimators": [50, 100, 200, 400, 1000],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "max_depth": [2, 3, 6, 8, 12, 16],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "params": {
                "n_neighbors": [3, 5, 7, 12, 16],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"],
            },
        },
        "DecisionTree": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {
                "max_depth": [3, 5, 10, 20, None],
                "criterion": ["gini", "entropy"],
                "min_samples_leaf": [1, 2, 4, 8, 12],
            },
        },
        "GaussianNB": {
            "model": GaussianNB(),
            "params": {
                "var_smoothing": np.logspace(0, -9, num=10),
            },
        },
    }

    print("\nbegin (CV=5)...\n")
    best_overall_score = 0
    best_model_name = ""

    for name, config in param_grids.items():
        print(f"[{name}] searching...")
        grid_search = GridSearchCV(
            config["model"],
            config["params"],
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1,
        )
        grid_search.fit(X_train_scaled, y_train)

        best_clf = grid_search.best_estimator_
        y_pred = best_clf.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, y_pred)

        print(f"  > best params: {grid_search.best_params_}")
        print(f"  > CV accuracy: {grid_search.best_score_:.4f}")
        print(f"  > test accuracy: {test_acc:.4f}")

        if test_acc > best_overall_score:
            best_overall_score = test_acc
            best_model_name = name

        print("-" * 40)
        print(f"{name} Report:")
        print(classification_report(y_test, y_pred, digits=4))
        print("=" * 60 + "\n")

    print(f"Complete. Best model: {best_model_name}, test acc: {best_overall_score:.4f}")


if __name__ == "__main__":
    main()
