"""Counterpart of utils/basis.rs."""

import tilelang.language as T


@T.macro
def orthonormal_basis3(v, first, second):
    sign = T.if_then_else(v[2] >= T.float32(0), T.float32(1), T.float32(-1))
    a = T.float32(-1) / (sign + v[2])
    b = v[0] * v[1] * a
    first[0] = T.float32(1) + sign * v[0] * v[0] * a
    first[1] = sign * b
    first[2] = -sign * v[0]
    second[0] = b
    second[1] = sign + v[1] * v[1] * a
    second[2] = -v[1]
