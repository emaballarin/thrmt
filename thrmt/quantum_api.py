#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ~~ Imports ~~ ────────────────────────────────────────────────────────────────
from typing import List
from typing import Optional
from typing import Tuple

import torch as th
from torch import Tensor

from .auxiliary import check_dtype
from .auxiliary import check_size
from .quantum_impl import random_rho_bh as _random_rho_bh
from .quantum_impl import random_rho_hs as _random_rho_hs
from .types import complex_dtypes

# ~~ Exports ~~ ────────────────────────────────────────────────────────────────
__all__: List[str] = [
    "random_rho_bh",
    "random_rho_hs",
]


# ~~ Functions ~~ ──────────────────────────────────────────────────────────────
# noinspection DuplicatedCode
def random_rho_hs(
    size: int,
    dtype: th.dtype = th.cdouble,
    device: Optional[th.device] = None,
    batch_shape: Optional[Tuple[int, ...]] = None,
) -> Tensor:
    """
    Generate a random quantum state (or a batch thereof) uniformly w.r.t. the Hilbert-Schmidt measure.

    Parameters
    ----------
    size : int
        The size of the square matrix.
    dtype : torch.dtype
        The data type. Default is torch.double.
    device : torch.device, optional
        The device. Default is None.
    batch_shape : tuple of ints, optional
        The batch shape for generating multiple matrices. Default is None.

    Returns
    -------
    Tensor
        A random quantum state (or a batch thereof) uniformly w.r.t. the Hilbert-Schmidt measure.
    """
    check_size(size)
    check_dtype(dtype, complex_dtypes)
    bs = () if batch_shape is None else batch_shape
    return _random_rho_hs(size=size, dtype=dtype, device=device, batch_shape=bs)


def random_rho_bh(
    size: int,
    dtype: th.dtype = th.cdouble,
    device: Optional[th.device] = None,
    batch_shape: Optional[Tuple[int, ...]] = None,
) -> Tensor:
    """
    Generate a random quantum state (or a batch thereof) uniformly w.r.t. the Bures-Helstrom measure.

    Parameters
    ----------
    size : int
        The size of the square matrix.
    dtype : torch.dtype
        The data type. Default is torch.double.
    device : torch.device, optional
        The device. Default is None.
    batch_shape : tuple of ints, optional
        The batch shape for generating multiple matrices. Default is None.

    Returns
    -------
    Tensor
        A random quantum state (or a batch thereof) uniformly w.r.t. the Bures-Helstrom measure.
    """
    check_size(size)
    check_dtype(dtype, complex_dtypes)
    bs = () if batch_shape is None else batch_shape
    return _random_rho_bh(size=size, dtype=dtype, device=device, batch_shape=bs)
