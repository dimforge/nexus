"""Native shapes/mod.rs reexports: Parry support-map counterparts."""

import tilelang.language as T
from kernel_support import dot


@T.macro
def ball_local_support_point(radius, direction, Output):
    inverse = 1.0 / T.sqrt(dot(direction, direction))
    for d in T.serial(3):
        Output[d] = (direction[d] * inverse) * radius


@T.macro
def cuboid_local_support_point(extents, direction, Output):
    for d in T.serial(3):
        bits = T.reinterpret(extents[d], "uint32") & T.uint32(0x7FFFFFFF)
        sign = T.reinterpret(direction[d], "uint32") & T.uint32(0x80000000)
        Output[d] = T.reinterpret(bits | sign, "float32")
