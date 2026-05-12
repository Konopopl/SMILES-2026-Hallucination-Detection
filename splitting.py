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
    """Use stratified k-fold splits; the probe uses a fixed threshold."""
    _ = df, val_frac

    y = np.asarray(y, dtype=int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    return [
        (idx_train, None, idx_test)
        for idx_train, idx_test in skf.split(np.zeros(len(y)), y)
    ]
