"""Tensor validation and shared MLP construction."""

import torch
from torch import nn


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def _finite_float(value: torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).float()
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains non-finite values")
    return tensor
