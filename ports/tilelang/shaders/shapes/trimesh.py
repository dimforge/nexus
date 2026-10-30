"""Counterpart of shapes/trimesh.rs; BVH nodes and triangle extraction."""

import tilelang.language as T
from kernel_support import f32

from .triangle import project_local_point_and_get_location as project_triangle


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


@T.macro
def triangle_project_local_point_and_get_location(a, b, c, pt, values, flags):
    # The mesh helper shares the primitive projection's point/feature arithmetic.
    # Its nearly-degenerate fallback is the native solid query; inside is later
    # determined from the mesh pseudo-normal rather than the primitive flag.
    project_triangle(a, b, c, pt, True, values, flags)


def pseudo_normal(l):
    from kernel_support import cross

    read_triangle = triangle(l)

    @T.macro
    def one(
        index_root,
        vertex_root,
        nodes,
        vertices_count,
        tri_id,
        feature,
        feature_id,
        Indices,
        Vertices,
        Output,
    ):
        for d in T.serial(3):
            Output[d] = 0.0
        # Signed branch temporaries avoid unsigned interval underflow in the
        # compiler allocation analysis; the native feature IDs stay uint32.
        feature_signed = T.cast(feature, "int32")
        id_signed = T.cast(feature_id, "int32")
        if feature_signed == 0:
            ids = T.alloc_local((3,), "int32")
            triangle_vids(index_root, vertex_root, nodes, tri_id, Indices, ids)
            selected = T.if_then_else(
                id_signed == 0,
                0,
                T.if_then_else(id_signed == 1, 1, 2),
            )
            for d in T.serial(3):
                Output[d] = f32(
                    Vertices,
                    (ids[selected] + vertices_count) * l["vertex_words"]
                    + l["vertex_vector"]
                    + d,
                )
        elif feature_signed == 1:
            start = (
                vertex_root
                + nodes * 2
                + vertices_count * 2
                + tri_id * 3
                + T.cast(feature_id, "int32")
            )
            for d in T.serial(3):
                Output[d] = f32(
                    Vertices, start * l["vertex_words"] + l["vertex_vector"] + d
                )
        elif feature_signed == 2 or feature_signed == 3:
            points = T.alloc_local((3, 3), "float32")
            a = T.alloc_local((3,), "float32")
            b = T.alloc_local((3,), "float32")
            read_triangle(
                index_root, vertex_root, nodes, tri_id, Indices, Vertices, points
            )
            for d in T.serial(3):
                a[d] = points[1, d] - points[0, d]
                b[d] = points[2, d] - points[0, d]
            cross(a, b, Output)

    return one


def project_local_point(l):
    from kernel_support import dot

    read_bounds = bvh_node_aabb(l)
    read_triangle = triangle(l)
    normal = pseudo_normal(l)

    @T.macro
    def one(
        index_root,
        vertex_root,
        nodes,
        vertices_count,
        Indices,
        Vertices,
        pt,
        max_distance,
        output,
        inside,
        found,
    ):
        current = T.alloc_local((1,), "int32")
        best = T.alloc_local((1,), "float32")
        best_triangle = T.alloc_local((1,), "int32")
        best_feature = T.alloc_local((2,), "uint32")
        current[0], best[0], best_triangle[0] = 0, max_distance, 0
        found[0] = False
        best_feature[0], best_feature[1] = T.uint32(3), T.uint32(0)
        for d in T.serial(3):
            output[d] = pt[d]
        for step in T.serial(nodes):
            if current[0] < nodes:
                node = T.alloc_local((3,), "uint32")
                bvh_node_idx(index_root, current[0], Indices, node)
                if node[0] == T.uint32(0xFFFFFFFF):
                    points = T.alloc_local((3, 3), "float32")
                    a = T.alloc_local((3,), "float32")
                    b = T.alloc_local((3,), "float32")
                    c = T.alloc_local((3,), "float32")
                    values = T.alloc_local((6,), "float32")
                    flags = T.alloc_local((3,), "uint32")
                    delta = T.alloc_local((3,), "float32")
                    read_triangle(
                        index_root,
                        vertex_root,
                        nodes,
                        T.cast(node[2], "int32"),
                        Indices,
                        Vertices,
                        points,
                    )
                    for d in T.serial(3):
                        a[d], b[d], c[d] = points[0, d], points[1, d], points[2, d]
                    triangle_project_local_point_and_get_location(
                        a, b, c, pt, values, flags
                    )
                    for d in T.serial(3):
                        delta[d] = values[d] - pt[d]
                    distance = T.sqrt(dot(delta, delta))
                    if distance <= best[0]:
                        found[0] = True
                        best[0] = distance
                        best_triangle[0] = T.cast(node[2], "int32")
                        best_feature[0], best_feature[1] = flags[0], flags[1]
                        for d in T.serial(3):
                            output[d] = values[d]
                    current[0] = T.cast(node[1], "int32")
                else:
                    mins = T.alloc_local((3,), "float32")
                    maxs = T.alloc_local((3,), "float32")
                    delta = T.alloc_local((3,), "float32")
                    read_bounds(vertex_root, current[0], Vertices, mins, maxs)
                    for d in T.serial(3):
                        delta[d] = T.max(mins[d] - pt[d], 0.0) - T.max(
                            pt[d] - maxs[d], 0.0
                        )
                    distance = T.sqrt(dot(delta, delta))
                    current[0] = T.cast(
                        T.if_then_else(distance <= best[0], node[0], node[1]), "int32"
                    )
        inside[0] = False
        if found[0]:
            pn = T.alloc_local((3,), "float32")
            delta = T.alloc_local((3,), "float32")
            normal(
                index_root,
                vertex_root,
                nodes,
                vertices_count,
                best_triangle[0],
                best_feature[0],
                best_feature[1],
                Indices,
                Vertices,
                pn,
            )
            for d in T.serial(3):
                delta[d] = pt[d] - output[d]
            inside[0] = dot(pn, delta) <= 0.0
        else:
            for d in T.serial(3):
                output[d] = 0.0

    return one
