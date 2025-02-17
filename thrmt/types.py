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
]

# ~~ dtypes ~~ ─────────────────────────────────────────────────────────────────

complex_dtypes: List[th.dtype] = [
    th.cdouble,
    th.cfloat,
    th.chalf,
    th.complex128,
    th.complex32,
    th.complex64,
]

real_dtypes: List[th.dtype] = [
    th.bfloat16,
    th.float16,
    th.float32,
    th.float64,
    th.float8_e4m3fn,
    th.float8_e4m3fnuz,
    th.float8_e5m2,
    th.float8_e5m2fnuz,
    th.half,
    th.float,
    th.double,
]
