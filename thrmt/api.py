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
from .auxiliary import check_sigma
from .auxiliary import check_size
from .impl import random_coe as _random_coe
from .impl import random_cue as _random_cue
from .impl import random_goe as _random_goe
from .impl import random_gue as _random_gue
from .impl import random_jce as _random_jce
from .impl import random_jre as _random_jre
from .impl import random_wce as _random_wce
from .impl import random_wre as _random_wre
from .types import complex_dtypes
from .types import real_dtypes


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
    batch_shape: Optional[Tuple[int, ...]] = None,
) -> Tensor:
    """
    Generate a random unitary matrix (or a batch thereof) from the Circular Unitary Ensemble (CUE),
    also known as the Uniform Haar measure on the unitary group.

    Parameters
    ----------
    size : int
        The size of the square matrix.
    dtype : torch.dtype
        The data type. Default is torch.cdouble.
    batch_shape : tuple of ints, optional
        The batch shape for generating multiple matrices. For example, batch_shape=(B,)
        returns a tensor of shape (B, size, size). Default is None (i.e. a single matrix).

    Returns
    -------
    Tensor
        A random unitary matrix of shape (*batch_shape, size, size).

    Raises
    ------
    ValueError
        If size is less than 1 or dtype is invalid.
    """
    check_size(size)
    check_dtype(dtype, complex_dtypes)
    bs = () if batch_shape is None else batch_shape
    return _random_cue(size=size, dtype=dtype, batch_shape=bs)


def random_coe(
    size: int,
    dtype: th.dtype = th.cdouble,
    batch_shape: Optional[Tuple[int, ...]] = None,
) -> Tensor:
    """
    Generate a random matrix (or a batch thereof) from the Circular Orthogonal Ensemble (COE).

    Parameters
    ----------
    size : int
        The size of the square matrix.
    dtype : torch.dtype
        The data type. Default is torch.cdouble.
    batch_shape : tuple of ints, optional
        The batch shape for generating multiple matrices. Default is None.

    Returns
    -------
    Tensor
        A random COE matrix of shape (*batch_shape, size, size).
    """
    check_size(size)
    check_dtype(dtype, complex_dtypes)
    bs = () if batch_shape is None else batch_shape
    return _random_coe(size=size, dtype=dtype, batch_shape=bs)


def random_gue(
    size: int,
    sigma: float,
    dtype: th.dtype = th.cdouble,
    batch_shape: Optional[Tuple[int, ...]] = None,
) -> Tensor:
    """
    Generate a matrix (or a batch thereof) from the Gaussian Unitary Ensemble (GUE) of scale sigma,
    also known as the Hermite (or Wigner) Unitary Ensemble.

    Parameters
    ----------
    size : int
        The size of the square matrix.
    sigma : float
        The scale parameter.
    dtype : torch.dtype
        The data type. Default is torch.cdouble.
    batch_shape : tuple of ints, optional
        The batch shape for generating multiple matrices. Default is None.

    Returns
    -------
    Tensor
        A random GUE matrix of shape (*batch_shape, size, size).
    """
    check_size(size)
    check_dtype(dtype, complex_dtypes)
    check_sigma(sigma)
    bs = () if batch_shape is None else batch_shape
    return _random_gue(size=size, sigma=sigma, dtype=dtype, batch_shape=bs)


def random_goe(
    size: int,
    sigma: float,
    dtype: th.dtype = th.double,
    batch_shape: Optional[Tuple[int, ...]] = None,
) -> Tensor:
    """
    Generate a matrix (or a batch thereof) from the Gaussian Orthogonal Ensemble (GOE) of scale sigma,
    also known as the Hermite (or Wigner) Orthogonal Ensemble.

    Parameters
    ----------
    size : int
        The size of the square matrix.
    sigma : float
        The scale parameter.
    dtype : torch.dtype
        The data type. Default is torch.double.
    batch_shape : tuple of ints, optional
        The batch shape for generating multiple matrices. Default is None.

    Returns
    -------
    Tensor
        A random GOE matrix of shape (*batch_shape, size, size).
    """
    check_size(size)
    check_dtype(dtype, real_dtypes)
    check_sigma(sigma)
    bs = () if batch_shape is None else batch_shape
    return _random_goe(size=size, sigma=sigma, dtype=dtype, batch_shape=bs)


def random_wre(
    size_n: int,
    sigma: float,
    size_m: Optional[int] = None,
    dtype: th.dtype = th.double,
    batch_shape: Optional[Tuple[int, ...]] = None,
) -> Tensor:
    """
    Generate a matrix (or a batch thereof) from the Wishart Real Ensemble (WRE), also known as the Laguerre Ensemble.

    Parameters
    ----------
    size_n : int
        The number of rows.
    sigma : float
        The scale parameter.
    size_m : int, optional
        The number of columns. If None, defaults to size_n.
    dtype : torch.dtype
        The data type. Default is torch.double.
    batch_shape : tuple of ints, optional
        The batch shape for generating multiple matrices. Default is None.

    Returns
    -------
    Tensor
        A random WRE matrix of shape (*batch_shape, size_n, size_m).
    """
    check_size(size_n)
    check_sigma(sigma)
    if size_m is not None:
        check_size(size_m)
    bs = () if batch_shape is None else batch_shape
    return _random_wre(
        size_n=size_n, sigma=sigma, size_m=size_m, dtype=dtype, batch_shape=bs
    )


def random_wce(
    size_n: int,
    sigma: float,
    size_m: Optional[int] = None,
    dtype: th.dtype = th.cdouble,
    batch_shape: Optional[Tuple[int, ...]] = None,
) -> Tensor:
    """
    Generate a matrix (or a batch thereof) from the Wishart Complex Ensemble (WCE), also known as the Laguerre Ensemble.

    Parameters
    ----------
    size_n : int
        The number of rows.
    sigma : float
        The scale parameter.
    size_m : int, optional
        The number of columns. If None, defaults to size_n.
    dtype : torch.dtype
        The data type. Default is torch.cdouble.
    batch_shape : tuple of ints, optional
        The batch shape for generating multiple matrices. Default is None.

    Returns
    -------
    Tensor
        A random WCE matrix of shape (*batch_shape, size_n, size_m).
    """
    check_size(size_n)
    check_sigma(sigma)
    if size_m is not None:
        check_size(size_m)
    bs = () if batch_shape is None else batch_shape
    return _random_wce(
        size_n=size_n, sigma=sigma, size_m=size_m, dtype=dtype, batch_shape=bs
    )


# noinspection DuplicatedCode
def random_jre(
    size_n: int,
    size_m1: Optional[int] = None,
    size_m2: Optional[int] = None,
    dtype: th.dtype = th.double,
    batch_shape: Optional[Tuple[int, ...]] = None,
) -> Tensor:
    """
    Generate a matrix (or a batch thereof) from the Jacobi Real Ensemble (JRE), also known as the MANOVA Ensemble.

    Parameters
    ----------
    size_n : int
        The number of rows.
    size_m1 : int, optional
        The number of columns of the first matrix. If None, defaults to size_n.
    size_m2 : int, optional
        The number of columns of the second matrix. If None, defaults to size_n.
    dtype : torch.dtype
        The data type. Default is torch.double.
    batch_shape : tuple of ints, optional
        The batch shape for generating multiple matrices. Default is None.

    Returns
    -------
    Tensor
        A random JRE matrix of shape (*batch_shape, size_n, size_n).
    """
    check_size(size_n)
    if size_m1 is not None:
        check_size(size_m1)
    if size_m2 is not None:
        check_size(size_m2)
    check_dtype(dtype, real_dtypes)
    bs = () if batch_shape is None else batch_shape
    return _random_jre(
        size_n=size_n, size_m1=size_m1, size_m2=size_m2, dtype=dtype, batch_shape=bs
    )


# noinspection DuplicatedCode
def random_jce(
    size_n: int,
    size_m1: Optional[int] = None,
    size_m2: Optional[int] = None,
    dtype: th.dtype = th.cdouble,
    batch_shape: Optional[Tuple[int, ...]] = None,
) -> Tensor:
    """
    Generate a matrix (or a batch thereof) from the Jacobi Complex Ensemble (JCE), also known as the MANOVA Ensemble.

    Parameters
    ----------
    size_n : int
        The number of rows.
    size_m1 : int, optional
        The number of columns of the first matrix. If None, defaults to size_n.
    size_m2 : int, optional
        The number of columns of the second matrix. If None, defaults to size_n.
    dtype : torch.dtype
        The data type. Default is torch.cdouble.
    batch_shape : tuple of ints, optional
        The batch shape for generating multiple matrices. Default is None.

    Returns
    -------
    Tensor
        A random JCE matrix of shape (*batch_shape, size_n, size_n).
    """
    check_size(size_n)
    if size_m1 is not None:
        check_size(size_m1)
    if size_m2 is not None:
        check_size(size_m2)
    check_dtype(dtype, complex_dtypes)
    bs = () if batch_shape is None else batch_shape
    return _random_jce(
        size_n=size_n, size_m1=size_m1, size_m2=size_m2, dtype=dtype, batch_shape=bs
    )
