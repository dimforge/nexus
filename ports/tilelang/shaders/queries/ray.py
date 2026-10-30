"""Counterpart of queries/ray.rs::Ray::point_at; directions need not be normalized."""

import tilelang.language as T


@T.macro
def point_at(origin, direction, parameter, output):
    for d in T.serial(3):
        output[d] = origin[d] + parameter * direction[d]
