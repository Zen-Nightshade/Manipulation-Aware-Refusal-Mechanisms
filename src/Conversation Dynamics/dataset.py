import torch
from torch.utils.data import Dataset
import numpy as np
import pickle


class SnippetDynamicsDataset(Dataset):
    def __init__(self, cache_path):
        with open(cache_path, "rb") as f:
            raw = pickle.load(f)

        self.features = []
        self.labels = []

        for item in raw:
            turns = item["turns"]
            p1 = [t for t in turns if t["speaker"] == "Person1"]
            p2 = [t for t in turns if t["speaker"] == "Person2"]

            if p1:
                p1_embs = np.array([t["embedding"] for t in p1])
                p1_mean = p1_embs.mean(axis=0)
                p1_max  = p1_embs.max(axis=0)
            else:
                p1_mean = p1_max = np.zeros(384, dtype=np.float32)

            if p2:
                p2_embs = np.array([t["embedding"] for t in p2])
                p2_mean = p2_embs.mean(axis=0)
                p2_max  = p2_embs.max(axis=0)
                p2_min  = p2_embs.min(axis=0)
            else:
                p2_mean = p2_max = p2_min = np.zeros(384, dtype=np.float32)

            def t(arr):
                return torch.tensor(arr, dtype=torch.float32)

            self.features.append({
                "content":       t(p1_mean),
                "context":       t(p2_mean),
                # Peak Dynamics (Model 3): spike intensity relative to own/victim baseline
                "trajectory":    t(p1_max - p1_mean),
                "interaction":   t(p1_max - p2_mean),
                # Asymmetry & Convergence (Model 4): sustained relational features
                "asymmetry":     t(p1_mean - p2_mean),
                "convergence":   t(p1_mean * p2_mean),
                # Victim Volatility & Dominance (Model 5): victim-centric destabilisation
                "p2_volatility": t(p2_max - p2_min),
                "dominance":     t(p1_max - p2_max),
                "surface":       t(np.array([len(turns) / 10., len(p1) / 10., len(p2) / 10.], dtype=np.float32)),
            })
            self.labels.append(torch.tensor(item["label"], dtype=torch.float32))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
