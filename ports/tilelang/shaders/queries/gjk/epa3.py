"""Counterpart of queries/gjk/epa3.rs with native bounded capacities."""

import tilelang.language as T
from kernel_support import cross, dot

from ...shapes.triangle import project_local_point_and_get_location as triangle
from ...utils.basis import orthonormal_basis3
from ..projection import barycentric_coordinates
from .gjk import cso_point_from_shapes


@T.macro
def none(Output, Valid):
    Valid[0] = False
    for field in T.serial(3):
        for d in T.serial(3):
            Output[field, d] = 0.0


@T.macro
def face_id_new(negative, Valid):
    Valid[0] = not (negative > T.float32(1e-6))


@T.macro
def heap_best_index(HDist, Counts, Index):
    Index[0] = 0
    for i in T.serial(Counts[3]):
        if HDist[i] > HDist[Index[0]]:
            Index[0] = i


@T.macro
def heap_peek(HIds, HDist, Counts, Id, Distance):
    index = T.alloc_local((1,), "int32")
    heap_best_index(HDist, Counts, index)
    Id[0] = HIds[index[0]]
    Distance[0] = HDist[index[0]]


@T.macro
def heap_pop(HIds, HDist, Counts, Id, Distance):
    index = T.alloc_local((1,), "int32")
    heap_best_index(HDist, Counts, index)
    Id[0] = HIds[index[0]]
    Distance[0] = HDist[index[0]]
    if Counts[3] != 0:
        Counts[3] = Counts[3] - 1
        HIds[index[0]] = HIds[Counts[3]]
        HDist[index[0]] = HDist[Counts[3]]


@T.macro
def heap_push(HIds, HDist, Counts, id, negative, Success):
    Success[0] = Counts[3] != 64
    if Counts[3] != 64:
        HIds[Counts[3]] = id
        HDist[Counts[3]] = negative
        Counts[3] = Counts[3] + 1


@T.macro
def ccw_face_normal(a, b, c, Normal):
    ab = T.alloc_local((3,), "float32")
    ac = T.alloc_local((3,), "float32")
    n = T.alloc_local((3,), "float32")
    for d in T.serial(3):
        ab[d] = b[d] - a[d]
        ac[d] = c[d] - a[d]
    cross(ab, ac, n)
    length = T.sqrt(dot(n, n))
    for d in T.serial(3):
        Normal[d] = T.if_then_else(length > T.float32(1e-7), n[d] / length, 0.0)


@T.macro
def face_new_with_proj(
    Vertices, Points, Adj, Bary, index, FPoints, FAdj, FNormal, FBary, FDeleted
):
    a = T.alloc_local((3,), "float32")
    b = T.alloc_local((3,), "float32")
    c = T.alloc_local((3,), "float32")
    normal = T.alloc_local((3,), "float32")
    for d in T.serial(3):
        a[d] = Vertices[Points[0], 0, d]
        b[d] = Vertices[Points[1], 0, d]
        c[d] = Vertices[Points[2], 0, d]
    ccw_face_normal(a, b, c, normal)
    for d in T.serial(3):
        FPoints[index, d] = Points[d]
        FAdj[index, d] = Adj[d]
        FNormal[index, d] = normal[d]
        FBary[index, d] = Bary[d]
    FDeleted[index] = False


@T.macro
def face_new(
    Vertices, Points, Adj, index, FPoints, FAdj, FNormal, FBary, FDeleted, Inside
):
    a = T.alloc_local((3,), "float32")
    b = T.alloc_local((3,), "float32")
    c = T.alloc_local((3,), "float32")
    zero = T.alloc_local((3,), "float32")
    values = T.alloc_local((6,), "float32")
    flags = T.alloc_local((3,), "uint32")
    bary = T.alloc_local((3,), "float32")
    for d in T.serial(3):
        a[d] = Vertices[Points[0], 0, d]
        b[d] = Vertices[Points[1], 0, d]
        c[d] = Vertices[Points[2], 0, d]
        zero[d] = 0.0
        bary[d] = 0.0
    triangle(a, b, c, zero, True, values, flags)
    kind = T.cast(flags[0], "int32")
    if (kind == 0) | (kind == 1):
        length = (values[0] * values[0] + values[1] * values[1]) + values[2] * values[2]
        eps = T.float32(1e-7) * 100.0
        Inside[0] = (flags[2] != 0) | (length < eps * eps)
        barycentric_coordinates(values, flags, bary)
    elif kind == 2:
        Inside[0] = True
        for d in T.serial(3):
            bary[d] = values[d + 3]
    else:
        Inside[0] = False
    face_new_with_proj(
        Vertices, Points, Adj, bary, index, FPoints, FAdj, FNormal, FBary, FDeleted
    )


@T.macro
def face_closest_points(Vertices, FPoints, FNormal, FBary, index, Output, Valid):
    for d in T.serial(3):
        Output[0, d] = (
            Vertices[FPoints[index, 0], 1, d] * FBary[index, 0]
            + Vertices[FPoints[index, 1], 1, d] * FBary[index, 1]
        ) + Vertices[FPoints[index, 2], 1, d] * FBary[index, 2]
        Output[1, d] = (
            Vertices[FPoints[index, 0], 2, d] * FBary[index, 0]
            + Vertices[FPoints[index, 1], 2, d] * FBary[index, 1]
        ) + Vertices[FPoints[index, 2], 2, d] * FBary[index, 2]
        Output[2, d] = FNormal[index, d]
    Valid[0] = True


@T.macro
def next_ccw_pt_id(FPoints, face, id, Result):
    Result[0] = T.if_then_else(
        FPoints[face, 0] == id, 1, T.if_then_else(FPoints[face, 1] == id, 2, 0)
    )


@T.macro
def can_be_seen_by(Vertices, FPoints, FNormal, face, point, opposite, Seen):
    a = T.alloc_local((3,), "float32")
    b = T.alloc_local((3,), "float32")
    c = T.alloc_local((3,), "float32")
    n = T.alloc_local((3,), "float32")
    crossed = T.alloc_local((3,), "float32")
    for d in T.serial(3):
        a[d] = Vertices[point, 0, d] - Vertices[FPoints[face, opposite], 0, d]
        b[d] = (
            Vertices[FPoints[face, (opposite + 2) % 3], 0, d]
            - Vertices[FPoints[face, (opposite + 1) % 3], 0, d]
        )
        c[d] = Vertices[point, 0, d] - Vertices[FPoints[face, (opposite + 1) % 3], 0, d]
        n[d] = FNormal[face, d]
    cross(b, c, crossed)
    area = dot(crossed, crossed)
    eps = T.float32(1e-7) * 100.0
    square = eps * eps
    Seen[0] = (
        (dot(a, n) >= -T.float32(1e-6)) | (area <= square) | (area <= area * square)
    )


@T.macro
def compute_silhouette(
    Vertices,
    FPoints,
    FAdj,
    FNormal,
    FDeleted,
    Silhouette,
    Counts,
    point,
    id,
    opposite,
    Success,
):
    stack = T.alloc_local((32, 2), "int32")
    length = T.alloc_local((1,), "int32")
    seen = T.alloc_local((1,), "bool")
    opp1 = T.alloc_local((1,), "int32")
    opp2 = T.alloc_local((1,), "int32")
    length[0] = 1
    stack[0, 0] = id
    stack[0, 1] = opposite
    Success[0] = True
    for iteration in T.serial(32):
        if (length[0] > 0) & Success[0]:
            length[0] = length[0] - 1
            face = stack[length[0], 0]
            opp = stack[length[0], 1]
            if not FDeleted[face]:
                can_be_seen_by(Vertices, FPoints, FNormal, face, point, opp, seen)
                if not seen[0]:
                    if Counts[2] < 32:
                        Silhouette[Counts[2], 0] = face
                        Silhouette[Counts[2], 1] = opp
                        Counts[2] = Counts[2] + 1
                    else:
                        Success[0] = False
                else:
                    FDeleted[face] = True
                    ip1 = (opp + 2) % 3
                    ip2 = opp
                    adj1 = FAdj[face, ip1]
                    adj2 = FAdj[face, ip2]
                    next_ccw_pt_id(FPoints, adj1, FPoints[face, ip1], opp1)
                    next_ccw_pt_id(FPoints, adj2, FPoints[face, ip2], opp2)
                    stack[length[0], 0] = adj2
                    stack[length[0], 1] = opp2[0]
                    length[0] = length[0] + 1
                    if length[0] < 32:
                        stack[length[0], 0] = adj1
                        stack[length[0], 1] = opp1[0]
                        length[0] = length[0] + 1
                    else:
                        Success[0] = False


@T.macro
# Separate macro keeps the eager Python frontend below its nesting limit.
# This is the native closest_points silhouette expansion loop, in the same order.
def expand_silhouette(
    vertices,
    fpoints,
    fadj,
    fnormal,
    fbary,
    deleted,
    silhouette,
    hids,
    hdist,
    counts,
    done,
    current,
    face,
    point_id,
    Output,
    Valid,
):
    pts = T.alloc_local((3,), "int32")
    adj = T.alloc_local((3,), "int32")
    inside = T.alloc_local((1,), "bool")
    success = T.alloc_local((1,), "bool")
    first_new = counts[1]
    if counts[2] == 0:
        done[0] = 2
    for edge in T.serial(counts[2]):
        if done[0] == 0:
            edge_face = silhouette[edge, 0]
            edge_opp = silhouette[edge, 1]
            if not deleted[edge_face]:
                new_face = counts[1]
                if new_face == 64:
                    done[0] = 2
                else:
                    pts[0] = fpoints[edge_face, (edge_opp + 2) % 3]
                    pts[1] = fpoints[edge_face, (edge_opp + 1) % 3]
                    pts[2] = point_id
                    adj[0] = edge_face
                    adj[1] = new_face + 1
                    adj[2] = new_face - 1
                    face_new(
                        vertices,
                        pts,
                        adj,
                        new_face,
                        fpoints,
                        fadj,
                        fnormal,
                        fbary,
                        deleted,
                        inside,
                    )
                    fadj[edge_face, (edge_opp + 1) % 3] = new_face
                    counts[1] = counts[1] + 1
                    if inside[0]:
                        vertex = fpoints[new_face, 0]
                        distance = (
                            fnormal[new_face, 0] * vertices[vertex, 0, 0]
                            + fnormal[new_face, 1] * vertices[vertex, 0, 1]
                        ) + fnormal[new_face, 2] * vertices[vertex, 0, 2]
                        if distance < current - T.float32(1e-7) * 100.0:
                            face_closest_points(
                                vertices, fpoints, fnormal, fbary, face, Output, Valid
                            )
                            done[0] = 1
                        else:
                            face_id_new(-distance, success)
                            if success[0]:
                                heap_push(
                                    hids, hdist, counts, new_face, -distance, success
                                )
                            if not success[0]:
                                done[0] = 2
    if done[0] == 0:
        if first_new == counts[1]:
            done[0] = 2
        else:
            fadj[first_new, 2] = counts[1] - 1
            fadj[counts[1] - 1, 1] = first_new
            counts[2] = 0


def closest_points(l):
    support = cso_point_from_shapes(l)

    @T.macro
    def one(
        Shapes,
        ia,
        ib,
        ShapeVertices,
        Q,
        Translation,
        SimplexVertices,
        SimplexDim,
        Output,
        Valid,
    ):
        vertices = T.alloc_local((32, 3, 3), "float32")
        fpoints = T.alloc_local((64, 3), "int32")
        fadj = T.alloc_local((64, 3), "int32")
        fnormal = T.alloc_local((64, 3), "float32")
        fbary = T.alloc_local((64, 3), "float32")
        deleted = T.alloc_local((64,), "bool")
        silhouette = T.alloc_local((32, 2), "int32")
        hids = T.alloc_local((64,), "int32")
        hdist = T.alloc_local((64,), "float32")
        counts = T.alloc_local((4,), "int32")
        pts = T.alloc_local((3,), "int32")
        adj = T.alloc_local((3,), "int32")
        inside = T.alloc_local((1,), "bool")
        success = T.alloc_local((1,), "bool")
        done = T.alloc_local((1,), "int32")
        dp1 = T.alloc_local((3,), "float32")
        dp2 = T.alloc_local((3,), "float32")
        dp3 = T.alloc_local((3,), "float32")
        temp = T.alloc_local((3,), "float32")
        normal = T.alloc_local((3,), "float32")
        newpoint = T.alloc_local((3, 3), "float32")
        none(Output, Valid)
        done[0] = 0
        for k in T.serial(4):
            counts[k] = 0
        dimension = T.cast(SimplexDim[0], "int32")
        for i in T.serial(dimension + 1):
            for field in T.serial(3):
                for d in T.serial(3):
                    vertices[i, field, d] = SimplexVertices[i, field, d]
        counts[0] = dimension + 1
        if dimension == 0:
            Valid[0] = True
            Output[2, 1] = 1.0
            done[0] = 1
        elif dimension == 3:
            for d in T.serial(3):
                dp1[d] = vertices[1, 0, d] - vertices[0, 0, d]
                dp2[d] = vertices[2, 0, d] - vertices[0, 0, d]
                dp3[d] = vertices[3, 0, d] - vertices[0, 0, d]
            cross(dp1, dp2, temp)
            if dot(temp, dp3) > 0.0:
                for field in T.serial(3):
                    for d in T.serial(3):
                        value = vertices[1, field, d]
                        vertices[1, field, d] = vertices[2, field, d]
                        vertices[2, field, d] = value
            counts[1] = 4
            for face in T.serial(4):
                if done[0] == 0:
                    pts[0] = T.if_then_else(face == 1, 1, 0)
                    pts[1] = T.if_then_else(
                        face == 0, 1, T.if_then_else(face == 2, 2, 3)
                    )
                    pts[2] = T.if_then_else(
                        face < 2, 2, T.if_then_else(face == 2, 3, 1)
                    )
                    adj[0] = T.if_then_else(
                        face < 2, 3, T.if_then_else(face == 2, 0, 2)
                    )
                    adj[1] = T.if_then_else(face == 1, 2, 1)
                    adj[2] = T.if_then_else(
                        face == 0, 2, T.if_then_else(face == 2, 3, 0)
                    )
                    face_new(
                        vertices,
                        pts,
                        adj,
                        face,
                        fpoints,
                        fadj,
                        fnormal,
                        fbary,
                        deleted,
                        inside,
                    )
                    if inside[0]:
                        distance = (
                            fnormal[face, 0] * vertices[face, 0, 0]
                            + fnormal[face, 1] * vertices[face, 0, 1]
                        ) + fnormal[face, 2] * vertices[face, 0, 2]
                        face_id_new(-distance, success)
                        if success[0]:
                            heap_push(hids, hdist, counts, face, -distance, success)
                        else:
                            done[0] = 2
            if counts[3] == 0:
                done[0] = 2
        else:
            if dimension == 1:
                for d in T.serial(3):
                    dp1[d] = vertices[1, 0, d] - vertices[0, 0, d]
                orthonormal_basis3(dp1, dp2, dp3)
                support(Shapes, ia, ib, ShapeVertices, Q, Translation, dp2, newpoint)
                for field in T.serial(3):
                    for d in T.serial(3):
                        vertices[counts[0], field, d] = newpoint[field, d]
                counts[0] = counts[0] + 1
            for face in T.serial(2):
                pts[0] = 0
                pts[1] = T.if_then_else(face == 0, 1, 2)
                pts[2] = T.if_then_else(face == 0, 2, 1)
                for d in T.serial(3):
                    adj[d] = 1 - face
                face_new(
                    vertices,
                    pts,
                    adj,
                    face,
                    fpoints,
                    fadj,
                    fnormal,
                    fbary,
                    deleted,
                    inside,
                )
                heap_push(hids, hdist, counts, face, 0.0, success)
            counts[1] = 2
        best = T.alloc_local((1,), "int32")
        maxdist = T.alloc_local((1,), "float32")
        old_dist = T.alloc_local((1,), "float32")
        face_id = T.alloc_local((1,), "int32")
        negdist = T.alloc_local((1,), "float32")
        peek_dist = T.alloc_local((1,), "float32")
        original_pts = T.alloc_local((3,), "int32")
        original_adj = T.alloc_local((3,), "int32")
        opposite = T.alloc_local((1,), "int32")
        best[0] = 0
        maxdist[0] = T.float32(l["query_constants"]["max_flt"])
        old_dist[0] = 0.0
        if done[0] == 0:
            if counts[3] == 0:
                done[0] = 2
            else:
                heap_peek(hids, hdist, counts, best, peek_dist)
        for iteration in T.serial(100):
            if done[0] == 0:
                if counts[3] == 0:
                    face_closest_points(
                        vertices, fpoints, fnormal, fbary, best[0], Output, Valid
                    )
                    done[0] = 1
                else:
                    heap_pop(hids, hdist, counts, face_id, negdist)
                    face = face_id[0]
                    if not deleted[face]:
                        for d in T.serial(3):
                            normal[d] = fnormal[face, d]
                            original_pts[d] = fpoints[face, d]
                            original_adj[d] = fadj[face, d]
                        support(
                            Shapes,
                            ia,
                            ib,
                            ShapeVertices,
                            Q,
                            Translation,
                            normal,
                            newpoint,
                        )
                        point_id = counts[0]
                        if counts[0] != 32:
                            for field in T.serial(3):
                                for d in T.serial(3):
                                    vertices[counts[0], field, d] = newpoint[field, d]
                            counts[0] = counts[0] + 1
                        else:
                            done[0] = 2
                        candidate = (
                            newpoint[0, 0] * normal[0] + newpoint[0, 1] * normal[1]
                        ) + newpoint[0, 2] * normal[2]
                        if done[0] == 0:
                            if candidate < maxdist[0]:
                                best[0] = face
                                maxdist[0] = candidate
                            current = -negdist[0]
                            delta = current - old_dist[0]
                            if (maxdist[0] - current < T.float32(1e-7) * 100.0) | (
                                (
                                    T.if_then_else(delta < 0.0, -delta, delta)
                                    < T.float32(1e-7)
                                )
                                & (candidate < maxdist[0])
                            ):
                                face_closest_points(
                                    vertices,
                                    fpoints,
                                    fnormal,
                                    fbary,
                                    best[0],
                                    Output,
                                    Valid,
                                )
                                done[0] = 1
                            else:
                                old_dist[0] = current
                                deleted[face] = True
                                for edge in T.serial(3):
                                    if done[0] == 0:
                                        next_ccw_pt_id(
                                            fpoints,
                                            original_adj[edge],
                                            original_pts[edge],
                                            opposite,
                                        )
                                        compute_silhouette(
                                            vertices,
                                            fpoints,
                                            fadj,
                                            fnormal,
                                            deleted,
                                            silhouette,
                                            counts,
                                            point_id,
                                            original_adj[edge],
                                            opposite[0],
                                            success,
                                        )
                                        if not success[0]:
                                            done[0] = 2
                                expand_silhouette(
                                    vertices,
                                    fpoints,
                                    fadj,
                                    fnormal,
                                    fbary,
                                    deleted,
                                    silhouette,
                                    hids,
                                    hdist,
                                    counts,
                                    done,
                                    current,
                                    face,
                                    point_id,
                                    Output,
                                    Valid,
                                )
        if done[0] == 0:
            face_closest_points(
                vertices, fpoints, fnormal, fbary, best[0], Output, Valid
            )

    return one
