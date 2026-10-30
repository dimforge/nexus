"""Counterpart of shapes/polyline.rs; BVH nodes and segment extraction."""

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


def segment(l):
    @T.macro
    def one(index_root, vertex_root, nodes, part, Indices, Vertices, Output):
        for v in T.serial(2):
            vid = (
                vertex_root
                + nodes * 2
                + T.cast(Indices[index_root + nodes * 3 + part * 2 + v], "int32")
            )
            for d in T.serial(3):
                Output[v, d] = f32(
                    Vertices, vid * l["vertex_words"] + l["vertex_vector"] + d
                )

    return one
