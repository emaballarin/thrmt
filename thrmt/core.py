#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
from typing import List

import torch
from torch import Tensor

# ──────────────────────────────────────────────────────────────────────────────
__all__: List[str] = ["batched_outer"]
# ──────────────────────────────────────────────────────────────────────────────


def batched_outer(vec1: Tensor, vec2: Tensor, use_einsum: bool = False) -> Tensor:
    """Outer product over the trailing axis, with arbitrary leading batch dims."""
    if use_einsum:
        return torch.einsum("...i,...j->...ij", vec1, vec2)
    return vec1.unsqueeze(-1) * vec2.unsqueeze(-2)
