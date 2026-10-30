"""Counterpart of shapes/convex_polyhedron.rs; local vertex-range AABB."""

import tilelang.language as T
from kernel_support import f32


def aabb(layout):
    @T.macro
    def one(Vertices, first, end, mins, maxs):
        for d in T.serial(3):
            mins[d] = f32(
                Vertices, first * layout["vertex_words"] + layout["vertex_vector"] + d
            )
            maxs[d] = mins[d]
        for i in T.serial(first, end):
            for d in T.serial(3):
                v = f32(
                    Vertices, i * layout["vertex_words"] + layout["vertex_vector"] + d
                )
                mins[d] = T.min(mins[d], v)
                maxs[d] = T.max(maxs[d], v)

    return one


def local_support_point(l):
    @T.macro
    def one(Vertices, first, end, direction, Output):
        best = T.alloc_local((1,), "float32")
        best[0] = T.float32(-3.4028234663852886e38)
        for d in T.serial(3):
            Output[d] = 0.0
        for i in T.serial(first, end):
            v = T.alloc_local((3,), "float32")
            for d in T.serial(3):
                v[d] = f32(Vertices, i * l["vertex_words"] + l["vertex_vector"] + d)
            value = (v[0] * direction[0] + v[1] * direction[1]) + v[2] * direction[2]
            if value > best[0]:
                best[0] = value
                for d in T.serial(3):
                    Output[d] = v[d]

    return one
