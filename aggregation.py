from __future__ import annotations

import torch


# We probe a small set of late transformer layers.  Concatenating mean and
# last-token pools across these layers consistently outperforms either pool
# alone on the validation folds.  hidden_states is shape
# (n_layers+1, seq_len, hidden_dim); index 0 is the embedding layer and
# index -1 is the final transformer layer.
_LAYERS = (-4, -2, -1)


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    mask = attention_mask.bool()
    parts: list[torch.Tensor] = []
    for li in _LAYERS:
        real = hidden_states[li][mask]  # (n_real, hidden_dim)
        parts.append(real.mean(dim=0))
        parts.append(real[-1])
    return torch.cat(parts, dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    mask = attention_mask.bool()
    n_layers = hidden_states.shape[0]
    pooled = [hidden_states[li][mask].mean(dim=0) for li in range(n_layers)]
    feats: list[torch.Tensor] = []
    for p in pooled:
        feats.append(p.norm(p=2))
    for i in range(1, n_layers):
        feats.append(
            1.0
            - torch.nn.functional.cosine_similarity(pooled[i - 1], pooled[i], dim=0)
        )
    feats.append(mask.sum().to(pooled[0].dtype))
    return torch.stack(feats)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    agg = aggregate(hidden_states, attention_mask)
    if use_geometric:
        geo = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg, geo], dim=0)
    return agg
