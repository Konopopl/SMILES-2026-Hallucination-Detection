# SMILES-2026 Hallucination Detection

## How to run

Install the dependencies and run the standard entry point:

```bash
pip install -r requirements.txt
python solution.py
```

The script writes two files in the repository root:

- `results.json`
- `predictions.csv`

The run uses `Qwen/Qwen2.5-0.5B` from the starter code. A CUDA GPU is useful,
because most of the time is spent extracting hidden states.

## Final approach

I changed three files: `aggregation.py`, `probe.py`, and `splitting.py`.

For each example I feed `prompt + response` into the model and use hidden states
from all layers. The main part of the feature vector is the mean of the last 16
real tokens in every layer. I also add a few scalar summaries:

- normalized sequence length;
- per-layer L2 norms for the last token, mean pool, max pool, tail mean, and
  tail standard deviation;
- L2 norms of changes between neighboring layers.

The probe is intentionally small:

- `StandardScaler`;
- PCA to 18 dimensions;
- logistic regression with `C=0.05`;
- fixed decision threshold `0.438`.

The split is stratified 10-fold cross-validation. I do not create a separate
validation subset inside each fold, since the final threshold is fixed.

## Validation result

The submitted `results.json` gives these averaged 10-fold numbers:

- baseline accuracy: 70.10%;
- probe train accuracy: 76.50%;
- probe held-out accuracy: 76.64%;
- probe held-out F1: 84.99%;
- probe held-out AUROC: 75.25%.

The train and held-out accuracy are close, so I kept this smaller probe instead
of larger models that looked less stable.

## Tried but not kept

I tried final-token features, selected late-layer pools, ridge regression,
linear SVMs, tree models, threshold tuning, and small ensembles. Some of them
matched the final accuracy on a few settings, but none gave a better
cross-validation result than the PCA-18 logistic probe.
