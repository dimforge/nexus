# ruff: noqa: SIM117
"""Native-word launch wrappers for public geometry helpers."""

import tilelang.language as T
from kernel_support import f32
from shaders.queries.ray import point_at
from shaders.shapes.trimesh import project_local_point, pseudo_normal


def query(l, c):
    project = project_local_point(l)
    normal = pseudo_normal(l)
    q, r = l["query"], l["result"]

    @T.macro
    def one(Queries, Vertices, Indices, Count, Results, i):
        if i < T.cast(Count[0], "int32"):
            base = i * l["query_words"]
            out = i * l["result_words"]
            pt = T.alloc_local((3,), "float32")
            direction = T.alloc_local((3,), "float32")
            result = T.alloc_local((3,), "float32")
            inside = T.alloc_local((1,), "bool")
            found = T.alloc_local((1,), "bool")
            inside[0], found[0] = False, True
            for d in T.serial(3):
                pt[d] = f32(Queries, base + q["point"] + d)
                direction[d] = f32(Queries, base + q["direction"] + d)
            if c["kind"] == "mesh_projection":
                project(
                    c["index_root"],
                    c["vertex_root"],
                    c["nodes"],
                    c["vertices_count"],
                    Indices,
                    Vertices,
                    pt,
                    f32(Queries, base + q["parameter"]),
                    result,
                    inside,
                    found,
                )
            elif c["kind"] == "mesh_pseudo_normal":
                normal(
                    c["index_root"],
                    c["vertex_root"],
                    c["nodes"],
                    c["vertices_count"],
                    T.cast(Queries[base + q["triangle"]], "int32"),
                    Queries[base + q["feature"]],
                    Queries[base + q["feature_id"]],
                    Indices,
                    Vertices,
                    result,
                )
            else:
                point_at(pt, direction, f32(Queries, base + q["parameter"]), result)
            for d in T.serial(3):
                Results[out + r["point"] + d] = T.reinterpret(result[d], "uint32")
            Results[out + r["found"]] = T.cast(found[0], "uint32")
            Results[out + r["inside"]] = T.cast(inside[0], "uint32")

    return one


def make_kernel(l, c, target):
    one = query(l, c)
    sizes, groups = c["buffer_words"], c["groups"]
    if target == "c":

        @T.prim_func
        def kernel(
            Queries: T.Tensor((sizes["Queries"],), "uint32"),
            Vertices: T.Tensor((sizes["Vertices"],), "uint32"),
            Indices: T.Tensor((sizes["Indices"],), "uint32"),
            Count: T.Tensor((sizes["Count"],), "uint32"),
            Results: T.Tensor((sizes["Results"],), "uint32"),
        ):
            with T.Kernel(1):
                for i in T.serial(groups * 64):
                    one(Queries, Vertices, Indices, Count, Results, i)
    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            Queries: A.Tensor((sizes["Queries"],), "uint32"),
            Vertices: A.Tensor((sizes["Vertices"],), "uint32"),
            Indices: A.Tensor((sizes["Indices"],), "uint32"),
            Count: A.Tensor((sizes["Count"],), "uint32"),
            Results: A.Tensor((sizes["Results"],), "uint32"),
        ):
            with A.Kernel(groups) as block:
                with A.SimtVF(threads=64):
                    one(
                        Queries,
                        Vertices,
                        Indices,
                        Count,
                        Results,
                        block * 64 + A.get_thread_binding(),
                    )
    else:

        @T.prim_func
        def kernel(
            Queries: T.Tensor((sizes["Queries"],), "uint32"),
            Vertices: T.Tensor((sizes["Vertices"],), "uint32"),
            Indices: T.Tensor((sizes["Indices"],), "uint32"),
            Count: T.Tensor((sizes["Count"],), "uint32"),
            Results: T.Tensor((sizes["Results"],), "uint32"),
        ):
            with T.Kernel(groups, threads=64) as block:
                one(
                    Queries,
                    Vertices,
                    Indices,
                    Count,
                    Results,
                    block * 64 + T.get_thread_binding(),
                )

    return kernel
