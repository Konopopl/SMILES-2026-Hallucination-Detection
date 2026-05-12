from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    n_folds: int = 10,
    val_frac: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    y = np.asarray(y, dtype=int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []
    for idx_train, idx_test in skf.split(np.zeros(len(y)), y):
        splits.append((idx_train, None, idx_test))
    return splits
