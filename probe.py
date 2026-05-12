"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary MLP that classifies feature
vectors as truthful (0) or hallucinated (1).  Called from ``solution.py``
via ``evaluate.run_evaluation``.  All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) must be implemented
and their signatures must not change.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


class ProbeMLP(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class HallucinationProbe(nn.Module):

    def __init__(self):
        super().__init__()

        self._net = None
        self._scaler = StandardScaler()
        self._threshold = 0.5

    def _build_network(self, input_dim):
        self._net = ProbeMLP(input_dim)

    def forward(self, x):
        return self._net(x)

    def fit(self, X, y):

        X = self._scaler.fit_transform(X)

        self._build_network(X.shape[1])

        X_t = torch.from_numpy(X).float()
        y_t = torch.from_numpy(y.astype(np.float32))

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos

        pos_weight = torch.tensor([
            n_neg / max(n_pos, 1)
        ]).float()

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1e-3,
            weight_decay=1e-2,
        )

        self.train()

        for _ in range(100):

            optimizer.zero_grad()

            logits = self(X_t)

            loss = criterion(logits, y_t)

            loss.backward()

            optimizer.step()

        self.eval()

        return self

    def fit_hyperparameters(self, X_val, y_val):

        probs = self.predict_proba(X_val)[:, 1]

        best_t = 0.5
        best_f1 = -1

        for t in np.linspace(0.1, 0.9, 81):

            preds = (probs >= t).astype(int)

            score = f1_score(y_val, preds)

            if score > best_f1:
                best_f1 = score
                best_t = t

        self._threshold = best_t

        return self

    def predict(self, X):
        return (
            self.predict_proba(X)[:, 1] >= self._threshold
        ).astype(int)

    def predict_proba(self, X):

        X = self._scaler.transform(X)

        X_t = torch.from_numpy(X).float()

        with torch.no_grad():
            logits = self(X_t)
            probs = torch.sigmoid(logits).numpy()

        return np.stack([1 - probs, probs], axis=1)
        

