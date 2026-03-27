import argparse
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    parser = argparse.ArgumentParser(description="Compute feature importance for VAE latent dimensions.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def normalize_safe(values):
    values = np.array(values).flatten()[:50]
    if values.max() == values.min():
        return np.zeros(50)
    scaler = MinMaxScaler()
    return scaler.fit_transform(values.reshape(-1, 1)).flatten()


def main():
    args = parse_args()
    ensure_exists(PROCESSED_DATA_PATH, "Processed dataset")
    ensure_exists(VAE_MODEL_PATH, "VAE weights")
    VAE_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_path = resolve_pipeline_path()

    bundle = joblib.load(pipeline_path)
    clf = bundle["stacking_ensemble"]
    scaler = bundle["scaler"]

    vae = VAE(input_dim=768, h1=256, h2=128, h3=50).to(args.device)
    vae.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=args.device))
    vae.eval()

    full_ds = SBERTVaeDataset(PROCESSED_DATA_PATH, test_mode=False, test_size=0.0)
    full_loader = DataLoader(full_ds, batch_size=32, shuffle=False)

    all_mu, all_sentences = [], []
    with torch.no_grad():
        for emb, _, sent in tqdm(full_loader, desc="Building retrieval library"):
            mu, _ = vae.encode(emb.to(args.device))
            all_mu.append(mu.cpu().numpy())
            all_sentences.extend(list(sent))
    all_mu_matrix = np.vstack(all_mu)

    test_ds = SBERTVaeDataset(PROCESSED_DATA_PATH, test_mode=True, test_size=0.2, random_seed=42)
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)

    with torch.no_grad():
        for emb, label, _ in test_loader:
            mu, logvar = vae.encode(emb.to(args.device))
            std = torch.exp(0.5 * logvar)
            X_test = torch.cat([mu, std], dim=-1).cpu().numpy()
            y_test = label.numpy()
            break

    X_test_scaled = scaler.transform(X_test)
    gini_imp = clf.named_estimators_["dt"].feature_importances_
    perm_res = permutation_importance(clf, X_test_scaled, y_test, n_repeats=3, n_jobs=-1)
    perm_imp = perm_res.importances_mean

    background = shap.kmeans(X_test_scaled, 50)
    explainer = shap.KernelExplainer(clf.predict_proba, background)
    shap_results = explainer.shap_values(X_test_scaled[:50], nsamples=200)
    shap_values = shap_results[1] if isinstance(shap_results, list) and len(shap_results) > 1 else shap_results
    shap_imp_full = np.abs(shap_values).mean(axis=0)

    score_df = pd.DataFrame(
        {
            "Feature": [f"Mu_{idx}" for idx in range(50)],
            "Gini": normalize_safe(gini_imp),
            "Permutation": normalize_safe(perm_imp),
            "SHAP": normalize_safe(shap_imp_full),
        }
    )
    score_df["Total_Score"] = (score_df["Gini"] + score_df["Permutation"] + score_df["SHAP"]) / 3
    top_10 = score_df.sort_values(by="Total_Score", ascending=False).head(10)
    all_50 = score_df.sort_values(by="Total_Score", ascending=False).head(50)

    interpretation = []
    for _, row in all_50.iterrows():
        feature_name = row["Feature"]
        idx = int(feature_name.split("_")[1])
        vals = all_mu_matrix[:, idx]
        high_s = all_sentences[int(np.argsort(vals)[-1])]
        low_s = all_sentences[int(np.argsort(vals)[0])]
        interpretation.append(
            {
                "feature": feature_name,
                "score": float(row["Total_Score"]),
                "high": high_s,
                "low": low_s,
            }
        )

    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=top_10.melt(id_vars="Feature", value_vars=["Gini", "Permutation", "SHAP"]),
        y="Feature",
        x="value",
        hue="variable",
        palette="viridis",
    )
    plt.title("Top 10 Important VAE Mu Dimensions")
    plt.tight_layout()
    plt.savefig(outputs_vae_path("mu_importance_ranking.png"))
    plt.close()

    report_path = outputs_vae_path("interpretation_report.json")
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(interpretation, f, ensure_ascii=False, indent=4)

    print(f"Saved ranking plot to: {outputs_vae_path('mu_importance_ranking.png')}")
    print(f"Saved interpretation report to: {report_path}")


if __name__ == "__main__":
    main()
