# SMILES-2026 Hallucination Detection Solution

## Reproducibility

I kept the standard project entry point. The submitted files can be regenerated
with:

```bash
pip install -r requirements.txt
python solution.py
```

The run creates `results.json` and `predictions.csv` in the repository root.
The model used by the starter code is `Qwen/Qwen2.5-0.5B`, so using a CUDA GPU
is strongly recommended for the hidden-state extraction step.

## Method

The main changes are in `aggregation.py` and `probe.py`. The split strategy is
the same stratified 5-fold setup used during validation.

### Hidden-state aggregation

Using only the final token was fairly noisy on this dataset. The version I kept
uses information from the end of the answer instead: for every hidden-state
layer, I average the last 16 non-padding tokens.

The final feature vector contains:

- the last-16-token mean for each layer;
- the normalized input length;
- per-layer L2 norms for the final token, full-sequence mean, max pool,
  tail mean, and tail standard deviation;
- adjacent-layer drift norms for the final-token and tail-mean representations.

This gives a feature vector of size 22574. The extra scalar features are small
compared with the pooled hidden states, but they helped the probe make slightly
more stable decisions across folds.

### Probe

The best validation result came from a simple linear probe:

- standardize all features;
- reduce them to 24 PCA components;
- train L2 logistic regression with `C=0.04`;
- predict with a fixed threshold of `0.40520593523979187`.

I kept the probe deliberately small. With only 689 labelled examples, larger
PCA dimensions and more flexible classifiers tended to improve the training
score without improving the held-out folds.

## Validation Results

The submitted `results.json` was produced by the official evaluation code. The
averaged 5-fold numbers are:

- baseline accuracy: 70.10%
- probe train accuracy: 76.12%
- probe validation AUROC: 74.52%
- probe held-out accuracy: 75.47%
- probe held-out F1: 84.31%
- probe held-out AUROC: 74.32%

The train accuracy is close to the held-out accuracy, which was one reason I
preferred this version over higher-capacity probes.

## Other Experiments

- All-layer final-token features reached about 74.16% accuracy, but the AUROC
  was lower.
- A selected late-layer mean/last/max representation was stable but did not
  beat the last-16-token tail mean.
- Ridge, linear SVM, tree models, and small logistic-regression ensembles were
  tried. None of them gave a better held-out accuracy/AUROC tradeoff than the
  compact PCA-24 logistic probe.
- Adding many more pooled vectors usually made the model more sensitive to the
  validation split, so the final version keeps only the tail mean plus compact
  scalar summaries.
