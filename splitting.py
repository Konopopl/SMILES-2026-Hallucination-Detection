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
    """Stratified 5-fold split.

    Each fold rotates one fifth of the data into the *test* slot; the
    remaining 80% is split once more into train (~85%) and validation (~15%),
    both stratified on the label.  Averaging metrics over five folds gives a
    more stable estimate than a single 70/15/15 split.

    The final probe in solution.py is fit on the union of all train+val
    indices, which with 5-fold coverage is the entire dataset.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []
    for fold_idx, (idx_trval, idx_test) in enumerate(
        skf.split(np.zeros(len(y)), y)
    ):
        idx_train, idx_val = train_test_split(
            idx_trval,
            test_size=val_frac,
            random_state=random_state + fold_idx,
            stratify=y[idx_trval],
        )
        splits.append((idx_train, idx_val, idx_test))
    return splits
