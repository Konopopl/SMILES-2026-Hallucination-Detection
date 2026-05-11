# SMILES-2026 Hallucination Detection Solution

## Reproducibility

The repository uses the original entry point:

```bash
pip install -r requirements.txt
python solution.py
```

This writes `results.json` and `predictions.csv`. A CUDA GPU is recommended
because the script extracts hidden states from `Qwen/Qwen2.5-0.5B` for all
training and competition-test rows.

## Final Approach

Only the student implementation files are changed:

- `aggregation.py`
- `probe.py`
- `splitting.py`

### Aggregation

The best reproducible validation result came from response-local features
rather than the single final token. For every hidden-state layer, the final
aggregation now averages the last 16 non-padding tokens. This gives one
tail-mean vector per layer.

The feature vector contains:

- tail-mean over the last 16 real tokens for all 25 hidden-state entries;
- normalized sequence length;
- layer-wise L2 norms of final-token, mean-pool, max-pool, tail-mean, and
  tail-std vectors;
- layer-wise drift norms between adjacent layers for final-token and tail-mean
  vectors.

This produces a 22574-dimensional feature vector. The scalar norms are a small
side channel that improved stability without adding a large number of fitted
parameters.

### Probe

The probe is a lightweight scikit-learn pipeline wrapped in the required
`HallucinationProbe` API:

- `StandardScaler`;
- PCA to 24 dimensions;
- L2 logistic regression with `C=0.04`;
- fixed decision threshold `0.40520593523979187`.

The low PCA dimension and strong regularization were chosen because the dataset
has only 689 labelled samples. In validation, larger PCA dimensions and more
flexible models often improved train scores but hurt held-out folds.

### Splitting

`splitting.py` uses stratified 5-fold evaluation. Each fold reserves one fifth
of the data as the reported held-out test split, then splits the remaining rows
into train and validation subsets with label stratification.

## Official Local Validation Result

The saved `results.json` was produced with the official evaluation loop using
the implementation above:

- baseline accuracy: 70.10%
- probe train accuracy: 76.12%
- probe validation AUROC: 74.52%
- probe held-out fold accuracy: 75.47%
- probe held-out fold F1: 84.31%
- probe held-out fold AUROC: 74.32%

The train/test gap is small by accuracy, so this version is less overfit than
the higher-capacity experiments.

## Experiments and Discarded Attempts

- Final-token features from all layers reached about 74.16% accuracy but had
  lower AUROC.
- Mean/last/max pooling over selected late layers was stable but weaker than
  the last-16-token tail mean.
- Logistic-regression ensembles and larger PCA dimensions were tested, but the
  best held-out accuracy came from the single compact PCA-24 probe with the
  richer scalar-norm side channel.
- Tree models, ridge classifiers, linear SVMs, and broader feature combinations
  were explored. They either overfit more or did not beat the tail-mean
  logistic baseline.
