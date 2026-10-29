"""Counterpart of shapes/triangle.rs; local AABB."""

import tilelang.language as T


@T.macro
def aabb(a, b, c, mins, maxs):
    for d in T.serial(3):
        mins[d] = T.min(T.min(a[d], b[d]), c[d])
        maxs[d] = T.max(T.max(a[d], b[d]), c[d])
