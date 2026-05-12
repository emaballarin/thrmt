#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ~~ Imports ~~ ────────────────────────────────────────────────────────────────
from typing import List

import torch as th

# ~~ Exports ~~ ────────────────────────────────────────────────────────────────
__all__: List[str] = [
    "complex_dtypes",
    "real_dtypes",
    "c2r_map",
    "r2c_map",
]

# ~~ dtypes ~~ ─────────────────────────────────────────────────────────────────

# Restricted to dtypes that work end-to-end through the linalg ops the
# library actually calls (qr, inv, eigvals, matmul). float8 variants,
# float16/bfloat16 and complex32 are rejected by the linalg layer on
# most builds; listing them here would mis-signal support. ``th.cfloat``
# / ``th.cdouble`` / ``th.float`` / ``th.double`` are identity aliases
# of the canonical entries below, so they pass ``check_dtype`` against
# these lists without needing separate entries.
complex_dtypes: List[th.dtype] = [
    th.complex64,
    th.complex128,
]

real_dtypes: List[th.dtype] = [
    th.float32,
    th.float64,
]

# ~~ dtype-maps ~~ ─────────────────────────────────────────────────────────────
c2r_map: dict[th.dtype, th.dtype] = {
    th.complex64: th.float32,
    th.complex128: th.float64,
}

r2c_map: dict[th.dtype, th.dtype] = {
    th.float32: th.complex64,
    th.float64: th.complex128,
}
