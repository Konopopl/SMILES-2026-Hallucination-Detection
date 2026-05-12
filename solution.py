from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from aggregation import aggregation_and_feature_extraction
from evaluate import print_summary, run_evaluation, save_predictions, save_results
from model import MAX_LENGTH, get_model_and_tokenizer
from probe import HallucinationProbe
from splitting import split_data


DATA_FILE = "./data/dataset.csv"
TEST_FILE = "./data/test.csv"
OUTPUT_FILE = "results.json"
PREDICTIONS_FILE = "predictions.csv"

BATCH_SIZE = 4
USE_GEOMETRIC = False


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_texts(df: pd.DataFrame) -> list[str]:
    return [f"{row['prompt']}{row['response']}" for _, row in df.iterrows()]


def extract_features(
    texts: list[str],
    model,
    tokenizer,
    device: torch.device,
) -> np.ndarray:
    features = []

    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Extracting", unit="batch"):
        batch = texts[start : start + BATCH_SIZE]
        encoding = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        hidden = torch.stack(outputs.hidden_states, dim=1).float()
        mask = attention_mask.cpu()

        for i in range(hidden.size(0)):
            feat = aggregation_and_feature_extraction(
                hidden[i],
                mask[i],
                use_geometric=USE_GEOMETRIC,
            )
            features.append(feat.cpu().numpy())

    return np.vstack(features)


def main() -> None:
    assert OUTPUT_FILE == "results.json"
    assert PREDICTIONS_FILE == "predictions.csv"

    device = choose_device()
    print(f"Device: {device}")
    print(f"Max length: {MAX_LENGTH}")

    df = pd.read_csv(DATA_FILE)
    y = df["label"].astype(float).astype(int).to_numpy()
    train_texts = make_texts(df)

    print(f"Train rows: {len(df)}")
    print(f"Labels: {dict(df['label'].value_counts().sort_index())}")

    model, tokenizer = get_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    t0 = time.time()
    X = extract_features(train_texts, model, tokenizer, device)
    extract_time = time.time() - t0

    print(f"Feature matrix: {X.shape}")
    print(f"Extraction time: {extract_time:.1f} s")

    splits = split_data(y, df)
    fold_results = run_evaluation(splits, X, y, HallucinationProbe)
    print_summary(fold_results, X.shape[1], len(X), extract_time)
    save_results(fold_results, X.shape[1], len(X), extract_time, OUTPUT_FILE)

    df_test = pd.read_csv(TEST_FILE)
    test_texts = make_texts(df_test)
    X_test = extract_features(test_texts, model, tokenizer, device)

    train_indices = np.unique(
        np.concatenate(
            [
                np.concatenate([idx_train, idx_val]) if idx_val is not None else idx_train
                for idx_train, idx_val, _ in splits
            ]
        )
    )

    final_probe = HallucinationProbe()
    final_probe.fit(X[train_indices], y[train_indices])
    save_predictions(final_probe, X_test, df_test.index, PREDICTIONS_FILE)


if __name__ == "__main__":
    main()
