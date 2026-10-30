"""Counterpart of shapes/trimesh.rs; BVH nodes and triangle extraction."""

import tilelang.language as T
from kernel_support import f32


@T.macro
def bvh_node_idx(index_root, node, Indices, Output):
    for d in T.serial(3):
        Output[d] = Indices[index_root + node * 3 + d]


def bvh_node_aabb(l):
    @T.macro
    def one(vertex_root, node, Vertices, Min, Max):
        for d in T.serial(3):
            Min[d] = f32(
                Vertices,
                (vertex_root + node * 2) * l["vertex_words"] + l["vertex_vector"] + d,
            )
            Max[d] = f32(
                Vertices,
                (vertex_root + node * 2 + 1) * l["vertex_words"]
                + l["vertex_vector"]
                + d,
            )

    return one


@T.macro
def triangle_vids(index_root, vertex_root, nodes, part, Indices, Output):
    for v in T.serial(3):
        Output[v] = (
            vertex_root
            + nodes * 2
            + T.cast(Indices[index_root + nodes * 3 + part * 3 + v], "int32")
        )


def triangle(l):
    @T.macro
    def one(index_root, vertex_root, nodes, part, Indices, Vertices, Output):
        ids = T.alloc_local((3,), "int32")
        triangle_vids(index_root, vertex_root, nodes, part, Indices, ids)
        for v in T.serial(3):
            for d in T.serial(3):
                Output[v, d] = f32(
                    Vertices, ids[v] * l["vertex_words"] + l["vertex_vector"] + d
                )

    return one
