import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _normalize_label(value):
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):
        numeric = int(value)
        if numeric in (0, 1):
            return numeric
        if numeric in (1, 2):
            return numeric - 1

    normalized = str(value).strip().lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    mapping = {
        "non translated": 0,
        "non translated e": 0,
        "nontranslated": 0,
        "nontranslated e": 0,
        "translated": 1,
        "translated e": 1,
    }
    if normalized in mapping:
        return mapping[normalized]

    raise ValueError(f"Unsupported label value: {value!r}")


class TfidfDataset(Dataset):
    def __init__(self, json_path, test_mode=False, test_size=0.2, random_seed=42):
        json_file = Path(json_path)
        with json_file.open("r", encoding="utf-8") as f:
            samples = json.load(f)

        np.random.seed(random_seed)
        idxs = np.arange(len(samples))
        np.random.shuffle(idxs)
        split = int(len(samples) * (1 - test_size))
        if test_mode:
            idxs = idxs[split:]
        else:
            idxs = idxs[:split]
        self.data = [samples[i] for i in idxs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        emb = torch.tensor(item["embedding"], dtype=torch.float32)
        lb = torch.tensor(_normalize_label(item["label"]), dtype=torch.long)
        return emb, lb, item.get("id", 0)
