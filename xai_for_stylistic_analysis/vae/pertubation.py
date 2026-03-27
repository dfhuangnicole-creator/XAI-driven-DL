import argparse
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.metrics.pairwise import cosine_similarity
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
    parser = argparse.ArgumentParser(description="Run perturbation-based XAI on the VAE stacking pipeline.")
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--max-iters", type=int, default=100)
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

    full_ds = SBERTVaeDataset(PROCESSED_DATA_PATH, test_mode=False, test_size=0.0, random_seed=42, return_idx=True)
    full_loader = DataLoader(full_ds, batch_size=256, shuffle=False)
    test_ds = SBERTVaeDataset(PROCESSED_DATA_PATH, test_mode=True, test_size=0.2, random_seed=42, return_idx=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    print("Building SBERT embeddings library...")
    all_embs = []
    all_sentences = []
    with torch.no_grad():
        for batch in tqdm(full_loader):
            embs_batch = batch[0]
            sents_batch = batch[2]
            all_embs.append(embs_batch.numpy())
            all_sentences.extend(list(sents_batch))

    all_embs_matrix = np.vstack(all_embs)

    all_rows = []
    for emb, label, orig_sent, sample_id in tqdm(test_loader, desc="Perturbation experiment"):
        label = label.item()
        orig_text = orig_sent[0]
        current_id = sample_id[0] if isinstance(sample_id, (list, torch.Tensor)) else sample_id
        emb_tensor = emb.to(args.device)

        with torch.no_grad():
            mu, logvar = vae.encode(emb_tensor)
            std_val = torch.exp(0.5 * logvar).cpu().numpy()
            z_orig = mu.cpu().numpy()

        init_feat = np.hstack([z_orig, std_val])
        init_feat_scaled = scaler.transform(init_feat)
        init_pred = int(np.argmax(clf.predict_proba(init_feat_scaled)[0]))
        target_label = 1 - init_pred

        row = {
            "id": int(current_id),
            "speech": label,
            "decision": init_pred,
            "after pertubation": init_pred,
            "ori text": orig_text,
            "matched text": "N/A",
        }

        z_perturbed = z_orig.copy()
        success = False
        matched_iter = 0

        for current_iter in range(args.max_iters):
            best_target_prob = -1
            best_next_z = z_perturbed.copy()

            for _ in range(8):
                direction = np.random.randn(1, 50)
                direction /= np.linalg.norm(direction) + 1e-9
                z_candidate = z_perturbed + direction * args.epsilon

                feat_candidate = scaler.transform(np.hstack([z_candidate, std_val]))
                prob_target = clf.predict_proba(feat_candidate)[0][target_label]

                if prob_target > best_target_prob:
                    best_target_prob = prob_target
                    best_next_z = z_candidate

            z_perturbed = best_next_z
            current_feat_scaled = scaler.transform(np.hstack([z_perturbed, std_val]))
            current_pred = int(np.argmax(clf.predict_proba(current_feat_scaled)[0]))

            if current_pred == target_label:
                success = True
                matched_iter = current_iter + 1
                row["after pertubation"] = current_pred
                break

        if not success:
            continue

        with torch.no_grad():
            z_tensor = torch.from_numpy(z_perturbed).float().to(args.device)
            decoded_emb = vae.decode(z_tensor).cpu().numpy()

        similarities = cosine_similarity(decoded_emb.reshape(1, -1), all_embs_matrix)[0]
        sorted_indices = np.argsort(similarities)[::-1]
        matched_idx = sorted_indices[0]
        for idx in sorted_indices:
            if all_sentences[idx].strip() != orig_text.strip():
                matched_idx = idx
                break

        row["matched text"] = all_sentences[matched_idx]
        row["cosine similarity"] = float(similarities[matched_idx])
        row["iter"] = matched_iter
        all_rows.append(row)

    if not all_rows:
        raise RuntimeError("No perturbation flips were found. Try increasing --max-iters or --epsilon.")

    df = pd.DataFrame(all_rows)
    column_order = [
        "id",
        "speech",
        "decision",
        "after pertubation",
        "cosine similarity",
        "ori text",
        "matched text",
        "iter",
    ]
    df = df[column_order]

    excel_path = outputs_vae_path("perturbation_analysis.xlsx")
    json_path = outputs_vae_path("perturbation_results.json")
    df.to_excel(excel_path, index=False)
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    print(f"Process complete. Total samples: {len(df)}")
    print(f"Saved spreadsheet to: {excel_path}")
    print(f"Saved json to: {json_path}")


if __name__ == "__main__":
    main()
