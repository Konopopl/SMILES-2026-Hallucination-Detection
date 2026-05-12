# SMILES-2026 Hallucination Detection

This repository contains my solution for the SMILES-2026 hallucination
detection task. The goal is to predict whether a model response is truthful
(`0`) or hallucinated (`1`) using hidden states from `Qwen/Qwen2.5-0.5B`.

## Run

```bash
pip install -r requirements.txt
python solution.py
```

After the run, the repository root will contain:

- `results.json` with cross-validation metrics;
- `predictions.csv` with labels for `data/test.csv`.

Using a GPU is recommended for the hidden-state extraction step.

## Main files

- `aggregation.py` builds the feature vector from hidden states.
- `probe.py` trains the PCA + logistic regression probe.
- `splitting.py` defines the 10-fold stratified split.
- `SOLUTION.md` describes the approach and validation numbers.
