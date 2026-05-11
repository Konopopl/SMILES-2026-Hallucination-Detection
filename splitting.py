from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    n_folds: int = 5,
    val_frac: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    y = np.asarray(y, dtype=int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []
    for fold, (idx_train_val, idx_test) in enumerate(skf.split(np.zeros(len(y)), y)):
        idx_train, idx_val = train_test_split(
            idx_train_val,
            test_size=val_frac,
            random_state=random_state + fold,
            stratify=y[idx_train_val],
        )
        splits.append((idx_train, idx_val, idx_test))
    return splits
