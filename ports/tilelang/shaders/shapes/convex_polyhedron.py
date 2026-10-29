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
