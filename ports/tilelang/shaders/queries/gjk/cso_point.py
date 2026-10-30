"""Counterpart of queries/gjk/cso_point.rs; local CSO values use [field,axis]."""

import tilelang.language as T


@T.macro
def from_points(a, b, Result):
    for d in T.serial(3):
        Result[0, d] = a[d] - b[d]
        Result[1, d] = a[d]
        Result[2, d] = b[d]


@T.macro
def origin(Result):
    for field in T.serial(3):
        for d in T.serial(3):
            Result[field, d] = 0.0
