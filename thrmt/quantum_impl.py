#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ~~ Imports ~~ ────────────────────────────────────────────────────────────────
from typing import List
from typing import Optional
from typing import Tuple

import torch as th
from torch import Tensor

from .impl import random_cue as _random_cue
from .impl import random_gce as _random_gce

# ~~ Exports ~~ ────────────────────────────────────────────────────────────────
__all__: List[str] = [
    "random_rho_hs",
    "random_rho_bh",
]


# ~~ Functions ~~ ──────────────────────────────────────────────────────────────


def random_rho_hs(
    size: int,
    dtype: th.dtype = th.cdouble,
    device: Optional[th.device] = None,
    batch_shape: Tuple[int, ...] = (),
):
    x: Tensor = _random_gce(
        size=size, dtype=dtype, device=device, batch_shape=batch_shape
    )
    aad: Tensor = x @ x.transpose(-2, -1).conj()
    return aad / th.trace(aad)


def random_rho_bh(
    size: int,
    dtype: th.dtype = th.cdouble,
    device: Optional[th.device] = None,
    batch_shape: Tuple[int, ...] = (),
):
    u: Tensor = _random_cue(
        size=size, dtype=dtype, device=device, batch_shape=batch_shape
    )
    a: Tensor = _random_gce(
        size=size, dtype=dtype, device=device, batch_shape=batch_shape
    )
    beye: Tensor = th.diag_embed(
        th.ones(*batch_shape, size, dtype=dtype, device=device)
    )
    x: Tensor = (beye + u) @ a
    aad: Tensor = x @ x.transpose(-2, -1).conj()
    return aad / th.trace(aad)
