# ruff: noqa: SIM117
"""Local simplex diagnostic ABI launch adapters."""

import tilelang.language as T
from kernel_support import f32
from shaders.queries.gjk.voronoi_simplex3 import (
    add_point,
    init,
    project_origin_and_reduce,
)


def make_kernel(l, c, target):
    sw = l["simplex_words"]
    cw = l["cso_words"]
    sf = l["simplex"]
    n = c["count"]
    words = c["storage"] * sw
    groups = c["groups"]
    kind = c["kind"]

    @T.macro
    def one(State, gid):
        if gid < n:
            for i in T.serial(gid, n, groups * 64):
                vertices = T.alloc_local((4, 3, 3), "float32")
                projection = T.alloc_local((3,), "float32")
                previous = T.alloc_local((3,), "float32")
                dims = T.alloc_local((2,), "uint32")
                indices = T.alloc_local((4,), "uint32")
                point = T.alloc_local((3, 3), "float32")
                result = T.alloc_local((3,), "float32")
                for v in T.serial(4):
                    for field in T.serial(3):
                        for d in T.serial(3):
                            vertices[v, field, d] = f32(
                                State, i * sw + sf["vertices"] + v * cw + field * 3 + d
                            )
                    indices[v] = State[i * sw + sf["previous_vertices"] + v]
                for d in T.serial(3):
                    projection[d] = f32(State, i * sw + sf["projection"] + d)
                    previous[d] = f32(State, i * sw + sf["previous_projection"] + d)
                dims[0] = State[i * sw + sf["dim"]]
                dims[1] = State[i * sw + sf["previous_dim"]]
                for field in T.serial(3):
                    for d in T.serial(3):
                        point[field, d] = f32(
                            State, i * sw + sf["new_point"] + field * 3 + d
                        )
                if kind == "simplex_init":
                    init(point, vertices, projection, dims, indices, previous)
                elif kind == "simplex_add":
                    add_point(point, vertices, projection, dims, indices, previous)
                else:
                    project_origin_and_reduce(
                        vertices, projection, dims, indices, previous, result
                    )
                    for d in T.serial(3):
                        State[i * sw + sf["projected"] + d] = T.reinterpret(
                            result[d], "uint32"
                        )
                for v in T.serial(4):
                    for field in T.serial(3):
                        for d in T.serial(3):
                            State[i * sw + sf["vertices"] + v * cw + field * 3 + d] = (
                                T.reinterpret(vertices[v, field, d], "uint32")
                            )
                    State[i * sw + sf["previous_vertices"] + v] = indices[v]
                for d in T.serial(3):
                    State[i * sw + sf["projection"] + d] = T.reinterpret(
                        projection[d], "uint32"
                    )
                    State[i * sw + sf["previous_projection"] + d] = T.reinterpret(
                        previous[d], "uint32"
                    )
                State[i * sw + sf["dim"]] = dims[0]
                State[i * sw + sf["previous_dim"]] = dims[1]

    if target == "c":

        @T.prim_func
        def kernel(State: T.Tensor((words,), "uint32")):
            with T.Kernel(1):
                for gid in T.serial(groups * 64):
                    one(State, gid)
    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(State: A.Tensor((words,), "uint32")):
            with A.Kernel(groups) as group:
                with A.SimtVF(threads=64):
                    one(State, group * 64 + A.get_thread_binding())
    else:

        @T.prim_func
        def kernel(State: T.Tensor((words,), "uint32")):
            with T.Kernel(groups, threads=64) as group:
                one(State, group * 64 + T.get_thread_binding())

    return kernel
