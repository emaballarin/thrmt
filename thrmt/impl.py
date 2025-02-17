#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ~~ Imports ~~ ────────────────────────────────────────────────────────────────
from math import sqrt as msqrt
from typing import List
from typing import Optional
from typing import Tuple

import torch as th
from torch import Tensor

# ~~ Exports ~~ ────────────────────────────────────────────────────────────────

__all__: List[str] = [
    "random_coe",  # Circular Orthogonal Ensemble
    "random_cue",  # Circular Unitary (Haar Uniform) Ensemble
    "random_goe",  # Gaussian (Hermite, or Wigner) Orthogonal Ensemble
    "random_gue",  # Gaussian (Hermite, or Wigner) Unitary Ensemble
    "random_jce",  # Jacobi (MANOVA) Complex Ensemble
    "random_jre",  # Jacobi (MANOVA) Real Ensemble
    "random_wce",  # Wishart (Laguerre) Complex Ensemble
    "random_wre",  # Wishart (Laguerre) Real Ensemble
]

# ~~ Functions ~~ ──────────────────────────────────────────────────────────────


def random_cue(
    size: int,
    dtype: th.dtype = th.cdouble,
    batch_shape: Tuple[int, ...] = (),
) -> Tensor:
    a = th.randn(*batch_shape, size, size, dtype=dtype)
    q, r = th.linalg.qr(a)
    d = r.diagonal(dim1=-2, dim2=-1)
    return q @ th.diag_embed(d / th.abs(d))


def random_coe(
    size: int,
    dtype: th.dtype = th.cdouble,
    batch_shape: Tuple[int, ...] = (),
) -> Tensor:
    x = random_cue(size=size, dtype=dtype, batch_shape=batch_shape)
    return x.transpose(-2, -1) @ x


def random_gue(
    size: int,
    sigma: float,
    dtype: th.dtype = th.cdouble,
    batch_shape: Tuple[int, ...] = (),
) -> Tensor:
    x = th.randn(*batch_shape, size, size, dtype=dtype) * sigma
    return (x + x.transpose(-2, -1).conj()) / msqrt(2)


def random_goe(
    size: int,
    sigma: float,
    dtype: th.dtype = th.double,
    batch_shape: Tuple[int, ...] = (),
) -> Tensor:
    x = th.randn(*batch_shape, size, size, dtype=dtype) * sigma
    return (x + x.transpose(-2, -1)) / msqrt(2)


def random_wre(
    size_n: int,
    sigma: float,
    size_m: Optional[int] = None,
    dtype: th.dtype = th.double,
    batch_shape: Tuple[int, ...] = (),
) -> Tensor:
    actual_size_m = size_m or size_n
    x = th.randn(*batch_shape, size_n, actual_size_m, dtype=dtype) * sigma
    return x @ x.transpose(-2, -1)


def random_wce(
    size_n: int,
    sigma: float,
    size_m: Optional[int] = None,
    dtype: th.dtype = th.cdouble,
    batch_shape: Tuple[int, ...] = (),
) -> Tensor:
    actual_size_m = size_m or size_n
    x = th.randn(*batch_shape, size_n, actual_size_m, dtype=dtype) * sigma
    return x @ x.transpose(-2, -1).conj()


def random_jre(
    size_n: int,
    size_m1: Optional[int] = None,
    size_m2: Optional[int] = None,
    dtype: th.dtype = th.double,
    batch_shape: Tuple[int, ...] = (),
) -> Tensor:
    actual_size_m1 = size_m1 or size_n
    actual_size_m2 = size_m2 or size_n
    x = th.randn(*batch_shape, size_n, actual_size_m1, dtype=dtype)
    a1 = x @ x.transpose(-2, -1)
    y = th.randn(*batch_shape, size_n, actual_size_m2, dtype=dtype)
    a2 = a1 + y @ y.transpose(-2, -1)
    return a1 @ th.linalg.inv(a2)


def random_jce(
    size_n: int,
    size_m1: Optional[int] = None,
    size_m2: Optional[int] = None,
    dtype: th.dtype = th.cdouble,
    batch_shape: Tuple[int, ...] = (),
) -> Tensor:
    actual_size_m1 = size_m1 or size_n
    actual_size_m2 = size_m2 or size_n
    x = th.randn(*batch_shape, size_n, actual_size_m1, dtype=dtype)
    a1 = x @ x.transpose(-2, -1).conj()
    y = th.randn(*batch_shape, size_n, actual_size_m2, dtype=dtype)
    a2 = a1 + y @ y.transpose(-2, -1).conj()
    return a1 @ th.linalg.inv(a2)
