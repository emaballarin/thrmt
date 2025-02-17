#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ~~ Imports ~~ ────────────────────────────────────────────────────────────────
from typing import List

import torch as th

# ~~ Exports ~~ ────────────────────────────────────────────────────────────────
__all__: List[str] = [
    "check_dtype",
    "check_size",
    "check_sigma",
]


# ~~ Functions ~~ ──────────────────────────────────────────────────────────────
def check_dtype(
    dtype: th.dtype,
    valid_dtypes: List[th.dtype],
) -> None:
    if dtype not in valid_dtypes:
        raise ValueError(f"Invalid dtype: {dtype}. Must be one of {valid_dtypes}")


def check_size(size: int) -> None:
    if size < 1:
        raise ValueError(f"Invalid size: {size}. Must be > 0.")


def check_sigma(sigma: float) -> None:
    if sigma < 0:
        raise ValueError(f"Invalid sigma: {sigma}. Must be >= 0.")
