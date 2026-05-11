from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


_PCA_DIM = 24
_C = 0.04
_THRESHOLD = 0.40520593523979187


class HallucinationProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._model: Pipeline | None = None
        self._threshold = _THRESHOLD

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(x.detach().cpu().numpy())[:, 1]
        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
        logits = np.log(probs / (1.0 - probs)).astype(np.float32)
        return torch.from_numpy(logits).to(x.device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        n_components = min(_PCA_DIM, X.shape[0] - 1, X.shape[1])
        self._model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "pca",
                    PCA(
                        n_components=n_components,
                        random_state=42,
                        svd_solver="randomized",
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=_C,
                        solver="liblinear",
                        class_weight=None,
                        max_iter=3000,
                        random_state=42,
                    ),
                ),
            ]
        )
        self._model.fit(X, y)
        self._threshold = _THRESHOLD
        return self

    def fit_hyperparameters(self, X_val: np.ndarray, y_val: np.ndarray) -> "HallucinationProbe":
        self._threshold = _THRESHOLD
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("call fit() first")
        X = np.asarray(X, dtype=np.float32)
        classes = self._model[-1].classes_
        pos_idx = int(np.where(classes == 1)[0][0])
        prob_pos = self._model.predict_proba(X)[:, pos_idx]
        prob_pos = np.clip(prob_pos, 1e-6, 1.0 - 1e-6)
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)
