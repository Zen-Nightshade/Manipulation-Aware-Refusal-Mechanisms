import torch
import torch.nn as nn


def _head(input_dim, num_labels):
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_labels),
    )


class _Base(nn.Module):
    def predict_proba(self, features):
        return torch.sigmoid(self.forward(features))

    def predict(self, features):
        probs = self.predict_proba(features)
        return (probs > self.thresholds.to(probs.device)).int()

    def set_thresholds(self, thresholds):
        self.thresholds = torch.tensor(thresholds, dtype=torch.float, device=self.thresholds.device)


class StaticModel(_Base):
    """Person1 mean embedding only — content baseline."""
    def __init__(self, emb_dim=384, num_labels=12):
        super().__init__()
        self.classifier = _head(emb_dim, num_labels)
        self.register_buffer("thresholds", torch.full((num_labels,), 0.5))

    def forward(self, f):
        return self.classifier(f["content"])


class ContextModel(_Base):
    """Person1 + Person2 mean embeddings — relational context baseline."""
    def __init__(self, emb_dim=384, num_labels=12):
        super().__init__()
        self.classifier = _head(emb_dim * 2, num_labels)
        self.register_buffer("thresholds", torch.full((num_labels,), 0.5))

    def forward(self, f):
        return self.classifier(torch.cat([f["content"], f["context"]], dim=1))


class DynamicModel(_Base):
    """
    Context + peak-based dynamic features.
    Trajectory = Max(P1) - Mean(P1): manipulator's spike above own baseline.
    Interaction = Max(P1) - Mean(P2): spike relative to victim's average tone.
    """
    def __init__(self, emb_dim=384, num_labels=12, use_surface=False):
        super().__init__()
        self.use_surface = use_surface
        self.classifier = _head(emb_dim * 4 + (3 if use_surface else 0), num_labels)
        self.register_buffer("thresholds", torch.full((num_labels,), 0.5))

    def forward(self, f):
        parts = [f["content"], f["context"], f["trajectory"], f["interaction"]]
        if self.use_surface:
            parts.append(f["surface"])
        return self.classifier(torch.cat(parts, dim=1))


class AsymmetryModel(_Base):
    """
    Context + sustained relational features.
    Asymmetry = Mean(P1) - Mean(P2): persistent tonal dominance gap.
    Convergence = Mean(P1) * Mean(P2): shared semantic alignment; manipulators
    steer victims toward their own framing, so overlap is informative.
    """
    def __init__(self, emb_dim=384, num_labels=12):
        super().__init__()
        self.classifier = _head(emb_dim * 4, num_labels)
        self.register_buffer("thresholds", torch.full((num_labels,), 0.5))

    def forward(self, f):
        return self.classifier(torch.cat([f["content"], f["context"], f["asymmetry"], f["convergence"]], dim=1))


class VolatilityModel(_Base):
    """
    Context + victim-centric destabilisation features.
    P2 Volatility = Max(P2) - Min(P2): range of victim's emotional state.
    Dominance = Max(P1) - Max(P2): peak-to-peak intensity gap between speakers.
    """
    def __init__(self, emb_dim=384, num_labels=12):
        super().__init__()
        self.classifier = _head(emb_dim * 4, num_labels)
        self.register_buffer("thresholds", torch.full((num_labels,), 0.5))

    def forward(self, f):
        return self.classifier(torch.cat([f["content"], f["context"], f["p2_volatility"], f["dominance"]], dim=1))
