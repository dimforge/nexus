"""Counterpart of shapes/convex_polyhedron.rs; local vertex-range AABB."""

import tilelang.language as T
from kernel_support import cross, dot, f32


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
        best[0] = -T.float32(l["query_constants"]["max_flt"])
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


def support_face(l):
    @T.macro
    def one(
        Vertices, Indices, first_vertex, first_face, end_face, direction, Output, Count
    ):
        best = T.alloc_local((3,), "int32")
        best_dot = T.alloc_local((1,), "float32")
        ids = T.alloc_local((3,), "int32")
        a = T.alloc_local((3,), "float32")
        ab = T.alloc_local((3,), "float32")
        ac = T.alloc_local((3,), "float32")
        normal = T.alloc_local((3,), "float32")
        best_dot[0] = -T.float32(l["query_constants"]["max_flt"])
        for d in T.serial(3):
            best[d] = 0
        for face in T.serial((end_face - first_face) // 3):
            for v in T.serial(3):
                ids[v] = first_vertex + T.cast(
                    Indices[first_face + face * 3 + v], "int32"
                )
            for d in T.serial(3):
                a[d] = f32(
                    Vertices, ids[0] * l["vertex_words"] + l["vertex_vector"] + d
                )
                ab[d] = (
                    f32(Vertices, ids[1] * l["vertex_words"] + l["vertex_vector"] + d)
                    - a[d]
                )
                ac[d] = (
                    f32(Vertices, ids[2] * l["vertex_words"] + l["vertex_vector"] + d)
                    - a[d]
                )
            cross(ab, ac, normal)
            length = T.sqrt(dot(normal, normal))
            if length != 0.0:
                for d in T.serial(3):
                    normal[d] = normal[d] / length
                value = dot(normal, direction)
                if value > best_dot[0]:
                    best_dot[0] = value
                    for d in T.serial(3):
                        best[d] = ids[d]
        for v in T.serial(3):
            for d in T.serial(3):
                Output[v, d] = f32(
                    Vertices, best[v] * l["vertex_words"] + l["vertex_vector"] + d
                )
        Count[0] = 3

    return one
