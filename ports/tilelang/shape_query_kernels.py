# ruff: noqa: SIM117
"""Native ABI adapter for four public Shape point-projection methods."""

import tilelang.language as T
from kernel_support import f32
from shaders.shapes.shape import (
    project_local_point,
    project_local_point_on_boundary,
    project_point,
    project_point_on_boundary,
)


def query(l, c):
    local = project_local_point(l)
    boundary = project_local_point_on_boundary(l)
    world = project_point(l)
    world_boundary = project_point_on_boundary(l)

    @T.macro
    def one(Shapes, Poses, Queries, Count, Results, i):
        if i < T.cast(Count[0], "int32"):
            base = i * l["query_words"]
            index = T.cast(Queries[base + l["query"]["shape_index"]], "int32")
            point = T.alloc_local((3,), "float32")
            q = T.alloc_local((4,), "float32")
            translation = T.alloc_local((3,), "float32")
            output = T.alloc_local((3,), "float32")
            inside = T.alloc_local((1,), "bool")
            inside[0] = False
            for d in T.serial(3):
                point[d] = f32(Queries, base + l["query"]["point"] + d)
                translation[d] = f32(
                    Poses, i * l["pose_words"] + l["pose_translation"] + d
                )
            for d in T.serial(4):
                q[d] = f32(Poses, i * l["pose_words"] + l["pose_rotation"] + d)
            if c["kind"] == "shape_project_local":
                local(Shapes, index, point, output)
            elif c["kind"] == "shape_project_local_boundary":
                boundary(Shapes, index, point, output, inside)
            elif c["kind"] == "shape_project_world":
                world(Shapes, index, q, translation, point, output, inside)
            else:
                world_boundary(Shapes, index, q, translation, point, output, inside)
            for d in T.serial(3):
                Results[i * l["result_words"] + l["result"]["point"] + d] = (
                    T.reinterpret(output[d], "uint32")
                )
            Results[i * l["result_words"] + l["result"]["inside"]] = T.cast(
                inside[0], "uint32"
            )

    return one


def make_kernel(l, c, target):
    one = query(l, c)
    sizes, groups = c["buffer_words"], c["groups"]
    if target == "c":

        @T.prim_func
        def kernel(
            Shapes: T.Tensor((sizes["Shapes"],), "uint32"),
            Poses: T.Tensor((sizes["Poses"],), "uint32"),
            Queries: T.Tensor((sizes["Queries"],), "uint32"),
            Count: T.Tensor((sizes["Count"],), "uint32"),
            Results: T.Tensor((sizes["Results"],), "uint32"),
        ):
            with T.Kernel(1):
                for i in T.serial(groups * 64):
                    one(Shapes, Poses, Queries, Count, Results, i)
    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            Shapes: A.Tensor((sizes["Shapes"],), "uint32"),
            Poses: A.Tensor((sizes["Poses"],), "uint32"),
            Queries: A.Tensor((sizes["Queries"],), "uint32"),
            Count: A.Tensor((sizes["Count"],), "uint32"),
            Results: A.Tensor((sizes["Results"],), "uint32"),
        ):
            with A.Kernel(groups) as block:
                with A.SimtVF(threads=64):
                    one(
                        Shapes,
                        Poses,
                        Queries,
                        Count,
                        Results,
                        block * 64 + A.get_thread_binding(),
                    )
    else:

        @T.prim_func
        def kernel(
            Shapes: T.Tensor((sizes["Shapes"],), "uint32"),
            Poses: T.Tensor((sizes["Poses"],), "uint32"),
            Queries: T.Tensor((sizes["Queries"],), "uint32"),
            Count: T.Tensor((sizes["Count"],), "uint32"),
            Results: T.Tensor((sizes["Results"],), "uint32"),
        ):
            with T.Kernel(groups, threads=64) as block:
                one(
                    Shapes,
                    Poses,
                    Queries,
                    Count,
                    Results,
                    block * 64 + T.get_thread_binding(),
                )

    return kernel
