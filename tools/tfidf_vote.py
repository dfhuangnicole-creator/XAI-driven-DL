import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from xai_for_stylistic_analysis.datasets import TfidfDataset


PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "tfidf" / "processed_TFIDF_final.json"
FEATURE_NAMES_PATH = ROOT_DIR / "data" / "processed" / "tfidf" / "feature_names.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TF-IDF stacking combinations.")
    parser.add_argument("--top-k", type=int, default=8, help="How many top weighted terms to print per sample.")
    return parser.parse_args()


def get_data_ready():
    train_ds = TfidfDataset(PROCESSED_DATA_PATH, test_mode=False, test_size=0.2, random_seed=42)
    test_ds = TfidfDataset(PROCESSED_DATA_PATH, test_mode=True, test_size=0.2, random_seed=42)

    def extract(loader):
        X, y = [], []
        for emb, lb, _ in loader:
            X.append(emb.cpu().numpy())
            y.append(lb.numpy())
        return np.vstack(X), np.concatenate(y)

    X_train, y_train = extract(DataLoader(train_ds, batch_size=32))
    X_test, y_test = extract(DataLoader(test_ds, batch_size=32))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_train_scaled, X_test, X_test_scaled, y_train, y_test


def print_sparse_top_terms(X, feature_names, sample_indices, top_k):
    for idx in sample_indices:
        row = X[idx]
        nonzero_indices = np.nonzero(row)[0]
        nonzero_count = len(nonzero_indices)
        total = row.size
        sparsity = 1.0 - nonzero_count / total
        print(f"\nSample #{idx}: {nonzero_count} nonzero out of {total} ({sparsity:.2%} sparse)")
        top_terms = sorted(zip(nonzero_indices, row[nonzero_indices]), key=lambda item: -item[1])
        print(f"Top {top_k} weighted terms:")
        for term_idx, value in top_terms[:top_k]:
            print(f"  {feature_names[term_idx]}: {value:.4f}")

    all_nonzero = np.count_nonzero(X)
    all_total = X.size
    all_sparsity = 1.0 - all_nonzero / all_total
    print(f"\nFull matrix sparsity: {all_sparsity:.4f} ({all_sparsity * 100:.2f}% zeros)")


def main():
    args = parse_args()

    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(f"Processed TF-IDF dataset not found: {PROCESSED_DATA_PATH}")
    if not FEATURE_NAMES_PATH.exists():
        raise FileNotFoundError(f"TF-IDF feature names not found: {FEATURE_NAMES_PATH}")

    with FEATURE_NAMES_PATH.open("r", encoding="utf-8") as f:
        feature_names = json.load(f)

    X_train, X_train_scaled, X_test, X_test_scaled, y_train, y_test = get_data_ready()

    print("Training set samples top-weighted terms & sparsity:")
    print_sparse_top_terms(X_train, feature_names, sample_indices=[0, 1, 2], top_k=args.top_k)
    print("\nTesting set samples top-weighted terms & sparsity:")
    print_sparse_top_terms(X_test, feature_names, sample_indices=[0, 1, 2], top_k=args.top_k)

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
