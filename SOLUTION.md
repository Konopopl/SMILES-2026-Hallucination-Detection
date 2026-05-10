# SMILES-2026 Hallucination Detection Solution

## Reproducibility

The repository is self-contained and uses the original `solution.py` entry point.
To reproduce the submitted files, run:

```bash
pip install -r requirements.txt
python solution.py
```

This produces:

- `results.json` with the cross-validation metrics from the official evaluation loop.
- `predictions.csv` with labels for `data/test.csv`.

The solution was developed with Python 3, PyTorch, Transformers, NumPy, pandas,
scikit-learn and tqdm as listed in `requirements.txt`. A CUDA GPU is recommended
because `solution.py` extracts Qwen/Qwen2.5-0.5B hidden states for all train and
test samples.

## Final Approach

Only the three allowed implementation files were modified:

- `aggregation.py`
- `probe.py`
- `splitting.py`

### Feature aggregation

The initial baseline used only the final token from the final transformer layer.
The final version uses a small set of late Qwen layers and concatenates two token
pools for each layer:

- mean pooling over all non-padding tokens;
- the last non-padding token representation.

The selected layers are `16`, `18`, `20`, `22`, `23`, and `24`. This keeps the feature vector small
enough for a lightweight probe while retaining both sequence-level and final-token
signals from the model response.

### Probe classifier

The final probe is a scikit-learn logistic regression model wrapped in the required
`HallucinationProbe` API. Features are standardized, reduced with PCA to at most
128 dimensions, then classified with L2-regularized logistic regression (`C=0.3`)
and balanced class weights. The final decision threshold is fixed at `0.07`,
selected from out-of-fold validation probabilities for the accuracy metric.

### Splitting strategy

Instead of a single random split, `splitting.py` uses stratified 5-fold evaluation.
Within each fold, the train+validation part is split again into train and validation
subsets with label stratification. This gives a more stable estimate on the small
689-sample dataset.

## Official Local Validation Result

The saved `results.json` was produced by running `python solution.py` with this
implementation. The averaged official-fold result is:

- baseline accuracy: 70.10%
- probe train accuracy: 76.42%
- probe validation AUROC: 67.45%
- probe held-out fold accuracy: 71.55%
- probe held-out fold AUROC: 68.86%

The final `predictions.csv` contains 100 predictions for the provided unlabeled
competition test file.

## Experiments and Discarded Attempts

- A small PyTorch MLP probe was tried first, but it overfit easily because the
  number of hidden-state features is large relative to the number of labeled
  samples.
- A single final-layer, final-token representation was too weak and unstable; adding middle-to-late layers improved both accuracy and AUROC.
- Additional geometric features such as layer norms, layer drift and sequence
  length were implemented as an optional path, but the final official run keeps
  `USE_GEOMETRIC = False` to stay simple and reproducible.
- Larger feature caches and broader model searches were explored during development,
  but they were not required for the final submitted `solution.py` workflow.