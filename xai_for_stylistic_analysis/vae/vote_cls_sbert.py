import argparse
from itertools import combinations

import torch
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .common import PROCESSED_DATA_PATH, ensure_exists
from .dataset import SBERTVaeDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate 3-model stacking combinations on SBERT embeddings.")
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

    return X_train_scaled, X_test_scaled, y_train, y_test


def main():
    parse_args()
    ensure_exists(PROCESSED_DATA_PATH, "Processed dataset")

    X_train_scaled, X_test_scaled, y_train, y_test = get_data_ready()

    best_models = {
        "SVM": SVC(C=0.1, kernel="linear", gamma="scale", probability=True, class_weight="balanced"),
        "RF": RandomForestClassifier(
            n_estimators=1000,
            max_depth=20,
            min_samples_split=5,
            bootstrap=False,
            random_state=42,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=16, weights="distance", metric="euclidean"),
        "DT": DecisionTreeClassifier(max_depth=10, criterion="gini", min_samples_leaf=2, random_state=42),
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
