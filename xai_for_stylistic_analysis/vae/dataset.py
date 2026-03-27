import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .common import to_zero_based_label

class SBERTVaeDataset(Dataset):
    def __init__(self, json_file, test_mode=False, test_size=0.2, random_seed=42, return_idx=False):
        json_path = Path(json_file)
        with json_path.open('r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        
        self.embeddings = torch.stack([
            torch.tensor(item['embedding'], dtype=torch.float32) 
            for item in self.raw_data
        ])
        self.idx = [item['idx'] for item in self.raw_data]
        self.idx = torch.tensor(self.idx, dtype=torch.long)
        self.labels = torch.tensor(
            [to_zero_based_label(item['label']) for item in self.raw_data],
            dtype=torch.long,
        )
        self.segments = [item['segments'] for item in self.raw_data]
        torch.manual_seed(random_seed)
        indices = torch.randperm(len(self.embeddings))
        test_count = int(test_size * len(self.embeddings))
        self.return_idx = return_idx
        if test_mode:
            self.idx = self.idx[indices[:test_count]]
            self.embeddings = self.embeddings[indices[:test_count]]
            self.labels = self.labels[indices[:test_count]]
            self.segments = [self.segments[i] for i in indices[:test_count]]
        else:
            self.idx = self.idx[indices[test_count:]]
            self.embeddings = self.embeddings[indices[test_count:]]
            self.labels = self.labels[indices[test_count:]]
            self.segments = [self.segments[i] for i in indices[test_count:]]

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        stack = " ".join(self.segments[idx])
        if self.return_idx:
            return self.embeddings[idx], self.labels[idx] , stack, self.idx[idx]
        else:
            return self.embeddings[idx], self.labels[idx] , stack
