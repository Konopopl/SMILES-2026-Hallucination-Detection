from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler


# Strong shrinkage: with thousands of features and ~470 samples the probe is
# heavily under-determined.  PCA caps the effective feature count and L2
# (C=0.5) keeps the linear coefficients small.  class_weight='balanced'
# stops the model from collapsing onto the 70 % majority class.
_PCA_DIM = 128
_C = 0.5


class HallucinationProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._scaler = StandardScaler()
        self._pca: PCA | None = None
        self._clf: LogisticRegression | None = None
        self._threshold: float = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._clf is None:
            raise RuntimeError("call fit() first")
        Xn = self._transform(x.detach().cpu().numpy())
        return torch.from_numpy(self._clf.decision_function(Xn)).float()

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X = self._scaler.transform(X)
        if self._pca is not None:
            X = self._pca.transform(X)
        return X

    def _make_clf(self) -> LogisticRegression:
        return LogisticRegression(
            C=_C,
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X_scaled = self._scaler.fit_transform(X)
        n_samples, n_features = X_scaled.shape
        # Cap PCA dim by both the user setting and what the matrix can support.
        n_components = min(_PCA_DIM, n_samples - 1, n_features)
        self._pca = PCA(n_components=n_components, random_state=42)
        X_red = self._pca.fit_transform(X_scaled)

        # Use unbiased CV probabilities to pick the operating threshold; this
        # avoids leaking train labels and is the only signal available when
        # fit_hyperparameters is not called (final probe path).
        cv_probs = cross_val_predict(
            self._make_clf(), X_red, y, cv=5, method="predict_proba", n_jobs=-1
        )[:, 1]
        self._threshold = _best_threshold(y, cv_probs)

        self._clf = self._make_clf()
        self._clf.fit(X_red, y)
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        probs = self.predict_proba(X_val)[:, 1]
        self._threshold = _best_threshold(y_val, probs)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("call fit() first")
        return self._clf.predict_proba(self._transform(X))


def _best_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    candidates = np.unique(np.concatenate([probs, np.linspace(0.05, 0.95, 91)]))
    best_t, best_acc = 0.5, -1.0
    for t in candidates:
        acc = accuracy_score(y_true, (probs >= t).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t
