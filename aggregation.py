from __future__ import annotations

import torch


_TAIL_WINDOW = 16


def _real_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    idx = attention_mask.bool().nonzero(as_tuple=False).flatten()
    if idx.numel() == 0:
        return torch.arange(attention_mask.numel(), device=attention_mask.device)
    return idx


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    idx = _real_indices(attention_mask).to(hidden_states.device)
    n_real = int(idx.numel())
    tail_idx = idx[-min(_TAIL_WINDOW, n_real) :]

    real_tokens = hidden_states[:, idx, :].float()
    tail_mean = hidden_states[:, tail_idx, :].float().mean(dim=1)
    mean_pool = real_tokens.mean(dim=1)
    max_pool = real_tokens.max(dim=1).values
    tail_std = hidden_states[:, tail_idx, :].float().std(dim=1, unbiased=False)
    last_token = hidden_states[:, idx[-1], :].float()

    last_diff = last_token[1:] - last_token[:-1]
    tail_diff = tail_mean[1:] - tail_mean[:-1]

    scalars = [
        torch.tensor([float(n_real) / 512.0], dtype=tail_mean.dtype, device=tail_mean.device),
        torch.linalg.vector_norm(last_token, ord=2, dim=1),
        torch.linalg.vector_norm(mean_pool, ord=2, dim=1),
        torch.linalg.vector_norm(max_pool, ord=2, dim=1),
        torch.linalg.vector_norm(tail_mean, ord=2, dim=1),
        torch.linalg.vector_norm(tail_std, ord=2, dim=1),
        torch.linalg.vector_norm(last_diff, ord=2, dim=1),
        torch.linalg.vector_norm(tail_diff, ord=2, dim=1),
    ]
    return torch.cat([tail_mean.reshape(-1), *scalars], dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    idx = _real_indices(attention_mask).to(hidden_states.device)
    n_real = int(idx.numel())
    tail_idx = idx[-min(_TAIL_WINDOW, n_real) :]
    real_tokens = hidden_states[:, idx, :].float()
    tail_mean = hidden_states[:, tail_idx, :].float().mean(dim=1)
    mean_pool = real_tokens.mean(dim=1)
    last_token = hidden_states[:, idx[-1], :].float()
    return torch.cat(
        [
            torch.linalg.vector_norm(last_token, ord=2, dim=1),
            torch.linalg.vector_norm(mean_pool, ord=2, dim=1),
            torch.linalg.vector_norm(tail_mean, ord=2, dim=1),
            torch.tensor([float(n_real)], dtype=tail_mean.dtype, device=tail_mean.device),
        ],
        dim=0,
    )


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    agg = aggregate(hidden_states, attention_mask)
    if use_geometric:
        return torch.cat([agg, extract_geometric_features(hidden_states, attention_mask)], dim=0)
    return agg
