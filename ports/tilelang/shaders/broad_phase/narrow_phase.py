"""Counterpart of broad_phase/narrow_phase.rs; native analytic first pass."""

import tilelang.language as T
from kernel_support import f32, quat_mul, quat_rotate

from ..queries.contact import (
    ball_ball,
    ball_convex,
    combine_coefficient,
    convex_ball,
    cuboid_cuboid,
)
from ..shapes.shape import primitive_local_extent


def gpu_narrow_phase_shape_shape(layout, case):
    ball_on_convex = ball_convex(layout)
    convex_on_ball = convex_ball(layout)
    extent = primitive_local_extent(layout)
    sf, types = layout["shape"], layout["shape_types"]
    cf, mf, pf = layout["contact"], layout["manifold"], layout["manifold_point"]
    mat = layout["material"]

    @T.macro
    def one(Pairs, Plan, Poses, Shapes, Contacts, Parents, Materials, RbdParams, gid):
        total = T.cast(Plan[layout["plan"]["pfm_base"]], "int32")
        prediction = f32(
            RbdParams, layout["rbd_params"]["normalized_prediction_distance"]
        ) * f32(RbdParams, layout["rbd_params"]["length_unit"])
        for t in T.serial(gid, total, case["groups"] * 64):
            c1 = T.cast(
                Pairs[t * layout["pair_words"] + layout["pair_colliders"]], "int32"
            )
            c2 = T.cast(
                Pairs[t * layout["pair_words"] + layout["pair_colliders"] + 1], "int32"
            )
            body1, body2 = Parents[c1], Parents[c2]
            point = T.alloc_local((3,), "float32")
            points = T.alloc_local((4, 4), "float32")
            for pi in T.serial(4):
                for pd in T.serial(4):
                    points[pi, pd] = T.float32(0)
            normal = T.alloc_local((3,), "float32")
            distance = T.alloc_local((1,), "float32")
            contact_len = T.alloc_local((1,), "int32")
            size1, size2 = (
                T.alloc_local((1,), "float32"),
                T.alloc_local((1,), "float32"),
            )
            contact_len[0] = T.int32(0)
            distance[0] = T.float32(0)
            size1[0], size2[0] = T.float32(0), T.float32(0)
            if body1 != body2:
                pose1, pose2 = c1 * layout["pose_words"], c2 * layout["pose_words"]
                q1 = T.alloc_local((4,), "float32")
                q2 = T.alloc_local((4,), "float32")
                q12 = T.alloc_local((4,), "float32")
                neg1 = T.alloc_local((3,), "float32")
                translation2 = T.alloc_local((3,), "float32")
                inv1 = T.alloc_local((3,), "float32")
                rotated2 = T.alloc_local((3,), "float32")
                t12 = T.alloc_local((3,), "float32")
                for d in T.serial(4):
                    q1[d] = f32(Poses, pose1 + layout["pose_rotation"] + d)
                    q2[d] = f32(Poses, pose2 + layout["pose_rotation"] + d)
                for d in T.serial(3):
                    q1[d] = -q1[d]
                    neg1[d] = -f32(Poses, pose1 + layout["pose_translation"] + d)
                    translation2[d] = f32(Poses, pose2 + layout["pose_translation"] + d)
                quat_mul(q1, q2, q12)
                quat_rotate(q1, neg1, inv1)
                quat_rotate(q1, translation2, rotated2)
                for d in T.serial(3):
                    t12[d] = inv1[d] + rotated2[d]
                ty1 = T.cast(Shapes[c1 * layout["shape_words"] + sf["a"] + 3], "int32")
                ty2 = T.cast(Shapes[c2 * layout["shape_words"] + sf["a"] + 3], "int32")
                radius1 = f32(Shapes, c1 * layout["shape_words"] + sf["a"])
                radius2 = f32(Shapes, c2 * layout["shape_words"] + sf["a"])
                extent(Shapes, c1, size1)
                extent(Shapes, c2, size2)
                if ty1 == types["ball"]:
                    if ty2 == types["ball"]:
                        ball_ball(t12, radius1, radius2, point, normal, distance)
                        contact_len[0] = T.int32(1)
                    elif (
                        ty2 == types["cuboid"]
                        or ty2 == types["capsule"]
                        or ty2 == types["cone"]
                        or ty2 == types["cylinder"]
                    ):
                        ball_on_convex(
                            Shapes, c2, q12, t12, radius1, point, normal, distance
                        )
                        contact_len[0] = T.int32(1)
                elif ty2 == types["ball"] and (
                    ty1 == types["cuboid"]
                    or ty1 == types["capsule"]
                    or ty1 == types["cone"]
                    or ty1 == types["cylinder"]
                ):
                    convex_on_ball(Shapes, c1, t12, radius2, point, normal, distance)
                    contact_len[0] = T.int32(1)
                if ty1 == types["cuboid"] and ty2 == types["cuboid"]:
                    he1, he2 = (
                        T.alloc_local((3,), "float32"),
                        T.alloc_local((3,), "float32"),
                    )
                    for d in T.serial(3):
                        he1[d] = f32(Shapes, c1 * layout["shape_words"] + sf["a"] + d)
                        he2[d] = f32(Shapes, c2 * layout["shape_words"] + sf["a"] + d)
                    cuboid_cuboid(
                        q12, t12, he1, he2, prediction, points, normal, contact_len
                    )
                    distance[0] = points[0, 3]
                elif contact_len[0] > 0:
                    for d in T.serial(3):
                        points[0, d] = point[d]
                    points[0, 3] = distance[0]
            out = t * layout["contact_words"]
            manifold = out + cf["contact"]
            if contact_len[0] > T.int32(0) and distance[0] < prediction:
                for w in T.serial(layout["contact_words"]):
                    Contacts[out + w] = T.uint32(0)
                for pi in T.serial(4):
                    for d in T.serial(3):
                        Contacts[
                            manifold
                            + mf["points_a"]
                            + pi * layout["manifold_point_words"]
                            + pf["pt"]
                            + d
                        ] = T.reinterpret(points[pi, d], "uint32")
                    Contacts[
                        manifold
                        + mf["points_a"]
                        + pi * layout["manifold_point_words"]
                        + pf["dist"]
                    ] = T.reinterpret(points[pi, 3], "uint32")
                for d in T.serial(3):
                    Contacts[manifold + mf["normal_a"] + d] = T.reinterpret(
                        normal[d], "uint32"
                    )
                Contacts[manifold + mf["len"]] = T.cast(contact_len[0], "uint32")
                Contacts[out + cf["colliders"]], Contacts[out + cf["colliders"] + 1] = (
                    T.cast(c1, "uint32"),
                    T.cast(c2, "uint32"),
                )
                Contacts[out + cf["bodies"]], Contacts[out + cf["bodies"] + 1] = (
                    body1,
                    body2,
                )
                m1, m2 = c1 * layout["material_words"], c2 * layout["material_words"]
                mu = combine_coefficient(
                    f32(Materials, m1 + mat["friction"]),
                    f32(Materials, m2 + mat["friction"]),
                    Materials[m1 + mat["friction_combine_rule"]],
                    Materials[m2 + mat["friction_combine_rule"]],
                )
                restitution = combine_coefficient(
                    f32(Materials, m1 + mat["restitution"]),
                    f32(Materials, m2 + mat["restitution"]),
                    Materials[m1 + mat["restitution_combine_rule"]],
                    Materials[m2 + mat["restitution_combine_rule"]],
                )
                Contacts[out + cf["friction"]] = T.reinterpret(mu, "uint32")
                Contacts[out + cf["restitution"]] = T.reinterpret(restitution, "uint32")
                Contacts[out + cf["recycle_extent"]] = T.reinterpret(
                    T.max(size1[0], size2[0]), "uint32"
                )
            else:
                Contacts[manifold + mf["len"]] = T.uint32(0)

    return one


def gpu_reset_narrow_phase(count):
    @T.macro
    def one(PfmCount, gid):
        if gid < count:
            PfmCount[gid] = T.uint32(0)

    return one


def gpu_contact_plan(layout):
    p, b = layout["plan"], layout["batch"]

    @T.macro
    def one(
        PairCount,
        PfmCount,
        Plan,
        PfmSortLen,
        ContactsIndirect,
        PfmIndirect,
        MbIndirect,
        BatchIds,
    ):
        # The pinned Metal and Ascend compilers do not lower atomic_load.
        # Add-zero preserves the counter bits and still performs an atomic read.
        pairs = T.min(
            T.atomic_add(PairCount[0], T.uint32(0), return_prev=True),
            BatchIds[b["collision_pairs_capacity"]],
        )
        pfm = T.min(
            T.atomic_add(PfmCount[0], T.uint32(0), return_prev=True),
            BatchIds[b["collision_pairs_capacity"]],
        )
        bound = pairs + pfm
        Plan[p["bound"]], Plan[p["pfm_base"]], Plan[p["pfm_len"]] = bound, pairs, pfm
        PfmSortLen[0] = pfm
        # Rust div_ceil does not overflow for totals near u32::MAX.
        ContactsIndirect[0] = bound // T.uint32(64) + T.cast(
            bound % T.uint32(64) != 0, "uint32"
        )
        PfmIndirect[0] = pfm // T.uint32(64) + T.cast(pfm % T.uint32(64) != 0, "uint32")
        ContactsIndirect[1], ContactsIndirect[2] = T.uint32(1), T.uint32(1)
        PfmIndirect[1], PfmIndirect[2] = T.uint32(1), T.uint32(1)
        MbIndirect[0] = T.if_then_else(
            bound > 0, BatchIds[b["multibodies_batch_capacity"]], T.uint32(0)
        )
        MbIndirect[1], MbIndirect[2] = BatchIds[b["num_batches"]], T.uint32(1)

    return one


@T.macro
def relative_pose(Q1, T1, Q2, T2, Q, T12):
    inverse = T.alloc_local((4,), "float32")
    negative = T.alloc_local((3,), "float32")
    left = T.alloc_local((3,), "float32")
    right = T.alloc_local((3,), "float32")
    for d in T.serial(3):
        inverse[d] = -Q1[d]
        negative[d] = -T1[d]
    inverse[3] = Q1[3]
    quat_mul(inverse, Q2, Q)
    quat_rotate(inverse, negative, left)
    quat_rotate(inverse, T2, right)
    for d in T.serial(3):
        T12[d] = left[d] + right[d]


def emit_pfm_pair(l):
    f = l["pfm_pair"]

    @T.macro
    def one(
        PfmPairs,
        PfmCount,
        capacity,
        S1,
        S2,
        Q,
        Translation,
        R1,
        R2,
        c1,
        c2,
        pair,
        subshape,
        extent,
    ):
        slot = T.cast(T.atomic_add(PfmCount[0], T.uint32(1), return_prev=True), "int32")
        if slot < capacity:
            base = slot * l["pfm_pair_words"]
            for w in T.serial(l["pfm_pair_words"]):
                PfmPairs[base + w] = T.uint32(0)
            for w in T.serial(l["shape_words"]):
                PfmPairs[base + f["shape1"] + w] = S1[w]
                PfmPairs[base + f["shape2"] + w] = S2[w]
            for d in T.serial(4):
                PfmPairs[base + f["pose12"] + l["pose_rotation"] + d] = T.reinterpret(
                    Q[d], "uint32"
                )
            for d in T.serial(3):
                PfmPairs[base + f["pose12"] + l["pose_translation"] + d] = (
                    T.reinterpret(Translation[d], "uint32")
                )
            values = T.alloc_local((3,), "float32")
            values[0] = R1
            values[1] = R2
            values[2] = extent
            PfmPairs[base + f["thickness1"]] = T.reinterpret(values[0], "uint32")
            PfmPairs[base + f["thickness2"]] = T.reinterpret(values[1], "uint32")
            PfmPairs[base + f["colliders"]] = T.cast(c1, "uint32")
            PfmPairs[base + f["colliders"] + 1] = T.cast(c2, "uint32")
            PfmPairs[base + f["pair_index"]] = T.cast(pair, "uint32")
            PfmPairs[base + f["subshape"]] = T.cast(subshape, "uint32")
            PfmPairs[base + f["recycle_extent"]] = T.reinterpret(values[2], "uint32")

    return one


def composite_convex(l, polyline):
    from ..bounding_volumes.aabb import intersects
    from ..shapes.shape import compute_aabb_at, pfm_subshape

    bounds = compute_aabb_at(l)
    subshape = pfm_subshape(l)
    emit = emit_pfm_pair(l)
    from ..shapes import polyline as line_helpers
    from ..shapes import trimesh as mesh_helpers

    helpers = line_helpers if polyline else mesh_helpers
    node_idx = helpers.bvh_node_idx
    node_aabb = helpers.bvh_node_aabb(l)
    part_points = line_helpers.segment(l) if polyline else mesh_helpers.triangle(l)
    sf = l["shape"]
    types = l["shape_types"]

    @T.macro
    def one(
        Shapes,
        mesh,
        convex,
        Vertices,
        Indices,
        Q,
        Translation,
        prediction,
        c1,
        c2,
        pair,
        PfmPairs,
        PfmCount,
        capacity,
    ):
        sub2 = T.alloc_local((l["shape_words"],), "uint32")
        radius = T.alloc_local((1,), "float32")
        valid = T.alloc_local((1,), "bool")
        subshape(Shapes, convex, sub2, radius, valid)
        if valid[0]:
            mins = T.alloc_local((3,), "float32")
            maxs = T.alloc_local((3,), "float32")
            rootmin = T.alloc_local((3,), "float32")
            rootmax = T.alloc_local((3,), "float32")
            thickness = (
                T.reinterpret(T.uint32(l["polyline_thickness_bits"]), "float32")
                if polyline
                else T.float32(0)
            )
            bounds(Shapes, Vertices, convex, Q, Translation, mins, maxs)
            for d in T.serial(3):
                mins[d] = mins[d] - (prediction + thickness)
                maxs[d] = maxs[d] + (prediction + thickness)
                rootmin[d] = f32(Shapes, mesh * l["shape_words"] + sf["b"] + d)
                rootmax[d] = f32(Shapes, mesh * l["shape_words"] + sf["c"] + d)
            if intersects(mins, maxs, rootmin, rootmax):
                vertex_base = T.cast(Shapes[mesh * l["shape_words"] + sf["a"]], "int32")
                index_base = T.cast(
                    Shapes[mesh * l["shape_words"] + sf["a"] + 1], "int32"
                )
                node_count = T.cast(
                    Shapes[mesh * l["shape_words"] + sf["a"] + 2], "int32"
                )
                current = T.alloc_local((1,), "int32")
                current[0] = 0
                for step in T.serial(node_count):
                    if current[0] < node_count:
                        index_record = T.alloc_local((3,), "uint32")
                        node_idx(index_base, current[0], Indices, index_record)
                        entry = T.reinterpret(index_record[0], "int32")
                        exit_id = T.if_then_else(
                            index_record[1] < T.cast(node_count, "uint32"),
                            T.cast(index_record[1], "int32"),
                            node_count,
                        )
                        part = T.cast(index_record[2], "int32")
                        if entry == T.int32(-1):
                            sub1 = T.alloc_local((l["shape_words"],), "uint32")
                            for w in T.serial(l["shape_words"]):
                                sub1[w] = T.uint32(0)
                            sub1[sf["a"] + 3] = T.uint32(
                                types["capsule"] if polyline else types["triangle"]
                            )
                            coords = T.alloc_local((3, 3), "float32")
                            part_points(
                                index_base,
                                vertex_base,
                                node_count,
                                part,
                                Indices,
                                Vertices,
                                coords,
                            )
                            for v in T.serial(2 if polyline else 3):
                                for d in T.serial(3):
                                    sub1[
                                        (
                                            sf["a"]
                                            if v == 0
                                            else sf["b"]
                                            if v == 1
                                            else sf["c"]
                                        )
                                        + d
                                    ] = T.reinterpret(coords[v, d], "uint32")
                            emit(
                                PfmPairs,
                                PfmCount,
                                capacity,
                                sub1,
                                sub2,
                                Q,
                                Translation,
                                thickness,
                                radius[0],
                                c1,
                                c2,
                                pair,
                                part + 1,
                                T.float32(-1),
                            )
                            current[0] = exit_id
                        else:
                            low = T.alloc_local((3,), "float32")
                            high = T.alloc_local((3,), "float32")
                            node_aabb(vertex_base, current[0], Vertices, low, high)
                            current[0] = T.if_then_else(
                                intersects(mins, maxs, low, high),
                                T.cast(entry, "int32"),
                                exit_id,
                            )

    return one


def trimesh_convex(l):
    return composite_convex(l, False)


def polyline_convex(l):
    return composite_convex(l, True)


def gpu_narrow_phase_shape_shape_deferred(l, c):
    from ..shapes.shape import local_extent, pfm_subshape

    subshape = pfm_subshape(l)
    extent = local_extent(l)
    emit = emit_pfm_pair(l)
    trimesh = trimesh_convex(l)
    polyline = polyline_convex(l)
    sf = l["shape"]
    types = l["shape_types"]

    @T.macro
    def one(
        Pairs,
        PairCount,
        Poses,
        Shapes,
        PfmPairs,
        PfmCount,
        Vertices,
        Indices,
        BatchIds,
        RbdParams,
        gid,
    ):
        capacity = T.cast(BatchIds[l["batch"]["collision_pairs_capacity"]], "int32")
        total = T.cast(
            T.min(
                T.atomic_add(PairCount[0], T.uint32(0), return_prev=True),
                T.cast(capacity, "uint32"),
            ),
            "int32",
        )
        prediction = f32(
            RbdParams, l["rbd_params"]["normalized_prediction_distance"]
        ) * f32(RbdParams, l["rbd_params"]["length_unit"])
        if gid < total:
            for t in T.serial(gid, total, c["groups"] * 64):
                c1 = T.cast(Pairs[t * l["pair_words"] + l["pair_colliders"]], "int32")
                c2 = T.cast(
                    Pairs[t * l["pair_words"] + l["pair_colliders"] + 1], "int32"
                )
                ty1 = T.cast(Shapes[c1 * l["shape_words"] + sf["a"] + 3], "int32")
                ty2 = T.cast(Shapes[c2 * l["shape_words"] + sf["a"] + 3], "int32")
                checked = (
                    (
                        (ty1 == types["ball"])
                        & (
                            (ty2 == types["ball"])
                            | (ty2 == types["cuboid"])
                            | (ty2 == types["capsule"])
                            | (ty2 == types["cone"])
                            | (ty2 == types["cylinder"])
                        )
                    )
                    | (
                        (ty2 == types["ball"])
                        & (
                            (ty1 == types["cuboid"])
                            | (ty1 == types["capsule"])
                            | (ty1 == types["cone"])
                            | (ty1 == types["cylinder"])
                        )
                    )
                    | ((ty1 == types["cuboid"]) & (ty2 == types["cuboid"]))
                )
                if not checked:
                    q1 = T.alloc_local((4,), "float32")
                    q2 = T.alloc_local((4,), "float32")
                    q = T.alloc_local((4,), "float32")
                    t1 = T.alloc_local((3,), "float32")
                    t2 = T.alloc_local((3,), "float32")
                    shift = T.alloc_local((3,), "float32")
                    for d in T.serial(4):
                        q1[d] = f32(
                            Poses, c1 * l["pose_words"] + l["pose_rotation"] + d
                        )
                        q2[d] = f32(
                            Poses, c2 * l["pose_words"] + l["pose_rotation"] + d
                        )
                    for d in T.serial(3):
                        t1[d] = f32(
                            Poses, c1 * l["pose_words"] + l["pose_translation"] + d
                        )
                        t2[d] = f32(
                            Poses, c2 * l["pose_words"] + l["pose_translation"] + d
                        )
                    relative_pose(q1, t1, q2, t2, q, shift)
                    s1 = T.alloc_local((l["shape_words"],), "uint32")
                    s2 = T.alloc_local((l["shape_words"],), "uint32")
                    r1 = T.alloc_local((1,), "float32")
                    r2 = T.alloc_local((1,), "float32")
                    valid1 = T.alloc_local((1,), "bool")
                    valid2 = T.alloc_local((1,), "bool")
                    subshape(Shapes, c1, s1, r1, valid1)
                    subshape(Shapes, c2, s2, r2, valid2)
                    if valid1[0] & valid2[0]:
                        e1 = T.alloc_local((1,), "float32")
                        e2 = T.alloc_local((1,), "float32")
                        extent(Shapes, Vertices, c1, e1)
                        extent(Shapes, Vertices, c2, e2)
                        emit(
                            PfmPairs,
                            PfmCount,
                            capacity,
                            s1,
                            s2,
                            q,
                            shift,
                            r1[0],
                            r2[0],
                            c1,
                            c2,
                            t,
                            0,
                            T.max(e1[0], e2[0]),
                        )
                    elif (ty1 == types["trimesh"]) | (ty1 == types["polyline"]):
                        if ty1 == types["trimesh"]:
                            trimesh(
                                Shapes,
                                c1,
                                c2,
                                Vertices,
                                Indices,
                                q,
                                shift,
                                prediction,
                                c1,
                                c2,
                                t,
                                PfmPairs,
                                PfmCount,
                                capacity,
                            )
                        else:
                            polyline(
                                Shapes,
                                c1,
                                c2,
                                Vertices,
                                Indices,
                                q,
                                shift,
                                prediction,
                                c1,
                                c2,
                                t,
                                PfmPairs,
                                PfmCount,
                                capacity,
                            )
                    elif (ty2 == types["trimesh"]) | (ty2 == types["polyline"]):
                        qi = T.alloc_local((4,), "float32")
                        negative = T.alloc_local((3,), "float32")
                        ti = T.alloc_local((3,), "float32")
                        for d in T.serial(3):
                            qi[d] = -q[d]
                            negative[d] = -shift[d]
                        qi[3] = q[3]
                        quat_rotate(qi, negative, ti)
                        if ty2 == types["trimesh"]:
                            trimesh(
                                Shapes,
                                c2,
                                c1,
                                Vertices,
                                Indices,
                                qi,
                                ti,
                                prediction,
                                c2,
                                c1,
                                t,
                                PfmPairs,
                                PfmCount,
                                capacity,
                            )
                        else:
                            polyline(
                                Shapes,
                                c2,
                                c1,
                                Vertices,
                                Indices,
                                qi,
                                ti,
                                prediction,
                                c2,
                                c1,
                                t,
                                PfmPairs,
                                PfmCount,
                                capacity,
                            )

    return one


def gpu_narrow_phase_pfm_pfm(l, c):
    from ..queries.contact_pfm_pfm import contact_manifold_pfm_pfm

    query = contact_manifold_pfm_pfm(l)
    cf = l["contact"]
    mf = l["manifold"]
    pf = l["manifold_point"]
    nf = l["pfm_pair"]
    mat = l["material"]

    @T.macro
    def one(
        Contacts,
        PfmPairs,
        Order,
        Plan,
        Vertices,
        Indices,
        Parents,
        Materials,
        RbdParams,
        gid,
    ):
        total = T.cast(Plan[l["plan"]["pfm_len"]], "int32")
        base = T.cast(Plan[l["plan"]["pfm_base"]], "int32")
        prediction = f32(
            RbdParams, l["rbd_params"]["normalized_prediction_distance"]
        ) * f32(RbdParams, l["rbd_params"]["length_unit"])
        if gid < total:
            for t in T.serial(gid, total, c["groups"] * 64):
                entry = T.cast(Order[t], "int32") * l["pfm_pair_words"]
                slot = base + t
                c1 = T.cast(PfmPairs[entry + nf["colliders"]], "int32")
                c2 = T.cast(PfmPairs[entry + nf["colliders"] + 1], "int32")
                body1 = Parents[c1]
                body2 = Parents[c2]
                points = T.alloc_local((4, 4), "float32")
                normal = T.alloc_local((3,), "float32")
                count = T.alloc_local((1,), "int32")
                count[0] = 0
                if body1 != body2:
                    shapes = T.alloc_local((l["shape_words"] * 2,), "uint32")
                    q = T.alloc_local((4,), "float32")
                    shift = T.alloc_local((3,), "float32")
                    for w in T.serial(l["shape_words"]):
                        shapes[w] = PfmPairs[entry + nf["shape1"] + w]
                        shapes[l["shape_words"] + w] = PfmPairs[
                            entry + nf["shape2"] + w
                        ]
                    for d in T.serial(4):
                        q[d] = f32(
                            PfmPairs, entry + nf["pose12"] + l["pose_rotation"] + d
                        )
                    for d in T.serial(3):
                        shift[d] = f32(
                            PfmPairs, entry + nf["pose12"] + l["pose_translation"] + d
                        )
                    query(
                        shapes,
                        0,
                        1,
                        Vertices,
                        Indices,
                        q,
                        shift,
                        f32(PfmPairs, entry + nf["thickness1"]),
                        f32(PfmPairs, entry + nf["thickness2"]),
                        prediction,
                        points,
                        normal,
                        count,
                    )
                out = slot * l["contact_words"]
                manifold = out + cf["contact"]
                if count[0] > 0 and points[0, 3] < prediction:
                    for w in T.serial(l["contact_words"]):
                        Contacts[out + w] = T.uint32(0)
                    for v in T.serial(4):
                        for d in T.serial(3):
                            Contacts[
                                manifold
                                + mf["points_a"]
                                + v * l["manifold_point_words"]
                                + pf["pt"]
                                + d
                            ] = T.reinterpret(points[v, d], "uint32")
                        Contacts[
                            manifold
                            + mf["points_a"]
                            + v * l["manifold_point_words"]
                            + pf["dist"]
                        ] = T.reinterpret(points[v, 3], "uint32")
                    for d in T.serial(3):
                        Contacts[manifold + mf["normal_a"] + d] = T.reinterpret(
                            normal[d], "uint32"
                        )
                    Contacts[manifold + mf["len"]] = T.cast(count[0], "uint32")
                    Contacts[out + cf["colliders"]] = T.cast(c1, "uint32")
                    Contacts[out + cf["colliders"] + 1] = T.cast(c2, "uint32")
                    Contacts[out + cf["bodies"]] = body1
                    Contacts[out + cf["bodies"] + 1] = body2
                    m1 = c1 * l["material_words"]
                    m2 = c2 * l["material_words"]
                    mu = combine_coefficient(
                        f32(Materials, m1 + mat["friction"]),
                        f32(Materials, m2 + mat["friction"]),
                        Materials[m1 + mat["friction_combine_rule"]],
                        Materials[m2 + mat["friction_combine_rule"]],
                    )
                    restitution = combine_coefficient(
                        f32(Materials, m1 + mat["restitution"]),
                        f32(Materials, m2 + mat["restitution"]),
                        Materials[m1 + mat["restitution_combine_rule"]],
                        Materials[m2 + mat["restitution_combine_rule"]],
                    )
                    Contacts[out + cf["friction"]] = T.reinterpret(mu, "uint32")
                    Contacts[out + cf["restitution"]] = T.reinterpret(
                        restitution, "uint32"
                    )
                    Contacts[out + cf["subshape"]] = PfmPairs[entry + nf["subshape"]]
                    Contacts[out + cf["recycle_extent"]] = PfmPairs[
                        entry + nf["recycle_extent"]
                    ]
                else:
                    Contacts[manifold + mf["len"]] = T.uint32(0)

    return one


def pfm_keys(l, c, kind):
    @T.macro
    def one(PfmPairs, Plan, Order, Keys, gid):
        total = T.cast(Plan[l["plan"]["pfm_len"]], "int32")
        if gid < total:
            for t in T.serial(gid, total, c["groups"] * 64):
                entry = T.cast(Order[t], "int32") if kind == "permuted" else t
                Keys[t] = PfmPairs[
                    entry * l["pfm_pair_words"]
                    + l["pfm_pair"]["subshape" if kind == "subshape" else "pair_index"]
                ]

    return one


def gpu_pfm_sort_keys(l, c):
    return pfm_keys(l, c, "pair")


def gpu_pfm_sub_shape_sort_keys(l, c):
    return pfm_keys(l, c, "subshape")


def gpu_pfm_pair_sort_keys_permuted(l, c):
    return pfm_keys(l, c, "permuted")


@T.macro
def pool_dedup(Candidates, Count, Point, epsilon_squared):
    hit = T.alloc_local((1,), "bool")
    hit[0] = False
    for k in T.serial(8):
        if k < Count[0] and not hit[0]:
            x = Candidates[k, 0] - Point[0]
            y = Candidates[k, 1] - Point[1]
            z = Candidates[k, 2] - Point[2]
            if ((x * x + y * y) + z * z) < epsilon_squared:
                if Point[3] < Candidates[k, 3]:
                    for d in T.serial(4):
                        Candidates[k, d] = Point[d]
                hit[0] = True
    if not hit[0] and Count[0] < 8:
        for d in T.serial(4):
            Candidates[Count[0], d] = Point[d]
        Count[0] = Count[0] + 1


def gpu_reduce_contacts(l, c):
    from kernel_support import dot

    from ..queries.polygonal_feature import manifold_reduction

    cf = l["contact"]
    mf = l["manifold"]
    pf = l["manifold_point"]
    cw = l["contact_words"]
    co = cf["contact"]

    @T.macro
    def one(Contacts, Keys, Plan, RbdParams, gid):
        total = T.cast(Plan[l["plan"]["pfm_len"]], "int32")
        base = T.cast(Plan[l["plan"]["pfm_base"]], "int32")
        prediction = f32(
            RbdParams, l["rbd_params"]["normalized_prediction_distance"]
        ) * f32(RbdParams, l["rbd_params"]["length_unit"])
        merge_cos = f32(RbdParams, l["rbd_params"]["contact_merge_cos"])
        if gid < total:
            for t in T.serial(gid, total, c["groups"] * 64):
                leader = T.alloc_local((1,), "bool")
                leader[0] = True
                if t > 0:
                    leader[0] = Keys[t - 1] != Keys[t]
                if leader[0]:
                    n = T.alloc_local((1,), "int32")
                    active = T.alloc_local((1,), "bool")
                    n[0] = 1
                    active[0] = True
                    for j in T.serial(t + 1, total):
                        if active[0]:
                            if Keys[j] == Keys[t]:
                                n[0] = n[0] + 1
                            else:
                                active[0] = False
                    if n[0] > 1:
                        write = T.alloc_local((1,), "int32")
                        write[0] = 0
                        for i in T.serial(n[0]):
                            im = T.alloc_local((cw,), "uint32")
                            for word in T.serial(cw):
                                im[word] = Contacts[(base + t + i) * cw + word]
                            if im[co + mf["len"]] != 0:
                                merged = T.alloc_local((1,), "bool")
                                merged[0] = False
                                for j in T.serial(write[0]):
                                    if not merged[0]:
                                        out = T.alloc_local((cw,), "uint32")
                                        n1 = T.alloc_local((3,), "float32")
                                        n2 = T.alloc_local((3,), "float32")
                                        for word in T.serial(cw):
                                            out[word] = Contacts[
                                                (base + t + j) * cw + word
                                            ]
                                        for d in T.serial(3):
                                            n1[d] = f32(out, co + mf["normal_a"] + d)
                                            n2[d] = f32(im, co + mf["normal_a"] + d)
                                        if dot(n1, n2) >= merge_cos:
                                            na = T.cast(
                                                T.min(out[co + mf["len"]], T.uint32(4)),
                                                "int32",
                                            )
                                            nb = T.cast(
                                                T.min(im[co + mf["len"]], T.uint32(4)),
                                                "int32",
                                            )
                                            candidates = T.alloc_local(
                                                (8, 4), "float32"
                                            )
                                            num = T.alloc_local((1,), "int32")
                                            point = T.alloc_local((4,), "float32")
                                            num[0] = 0
                                            for v in T.serial(8):
                                                for d in T.serial(4):
                                                    candidates[v, d] = 0.0
                                            eps = prediction * T.float32(0.25)
                                            eps_sq = eps * eps
                                            for v in T.serial(na):
                                                for d in T.serial(3):
                                                    point[d] = f32(
                                                        out,
                                                        co
                                                        + mf["points_a"]
                                                        + v * l["manifold_point_words"]
                                                        + pf["pt"]
                                                        + d,
                                                    )
                                                point[3] = f32(
                                                    out,
                                                    co
                                                    + mf["points_a"]
                                                    + v * l["manifold_point_words"]
                                                    + pf["dist"],
                                                )
                                                pool_dedup(
                                                    candidates, num, point, eps_sq
                                                )
                                            for v in T.serial(nb):
                                                for d in T.serial(3):
                                                    point[d] = f32(
                                                        im,
                                                        co
                                                        + mf["points_a"]
                                                        + v * l["manifold_point_words"]
                                                        + pf["pt"]
                                                        + d,
                                                    )
                                                point[3] = f32(
                                                    im,
                                                    co
                                                    + mf["points_a"]
                                                    + v * l["manifold_point_words"]
                                                    + pf["dist"],
                                                )
                                                pool_dedup(
                                                    candidates, num, point, eps_sq
                                                )
                                            deep1 = T.alloc_local((1,), "float32")
                                            deep2 = T.alloc_local((1,), "float32")
                                            deep1[0] = f32(
                                                out, co + mf["points_a"] + pf["dist"]
                                            )
                                            deep2[0] = f32(
                                                im, co + mf["points_a"] + pf["dist"]
                                            )
                                            for v in T.serial(1, na):
                                                value = f32(
                                                    out,
                                                    co
                                                    + mf["points_a"]
                                                    + v * l["manifold_point_words"]
                                                    + pf["dist"],
                                                )
                                                if value < deep1[0]:  # noqa: PLR1730 -- Python min cannot compare TIR expressions
                                                    deep1[0] = value
                                            for v in T.serial(1, nb):
                                                value = f32(
                                                    im,
                                                    co
                                                    + mf["points_a"]
                                                    + v * l["manifold_point_words"]
                                                    + pf["dist"],
                                                )
                                                if value < deep2[0]:  # noqa: PLR1730 -- Python min cannot compare TIR expressions
                                                    deep2[0] = value
                                            normal = T.alloc_local((3,), "float32")
                                            points = T.alloc_local((4, 4), "float32")
                                            count = T.alloc_local((1,), "int32")
                                            for d in T.serial(3):
                                                normal[d] = T.if_then_else(
                                                    deep2[0] < deep1[0], n2[d], n1[d]
                                                )
                                            manifold_reduction(
                                                candidates,
                                                num[0],
                                                normal,
                                                prediction,
                                                points,
                                                count,
                                            )
                                            for v in T.serial(4):
                                                for d in T.serial(3):
                                                    out[
                                                        co
                                                        + mf["points_a"]
                                                        + v * l["manifold_point_words"]
                                                        + pf["pt"]
                                                        + d
                                                    ] = T.reinterpret(
                                                        points[v, d], "uint32"
                                                    )
                                                out[
                                                    co
                                                    + mf["points_a"]
                                                    + v * l["manifold_point_words"]
                                                    + pf["dist"]
                                                ] = T.reinterpret(
                                                    points[v, 3], "uint32"
                                                )
                                            for d in T.serial(3):
                                                out[co + mf["normal_a"] + d] = (
                                                    T.reinterpret(normal[d], "uint32")
                                                )
                                            out[co + mf["len"]] = T.cast(
                                                count[0], "uint32"
                                            )
                                            for word in T.serial(cw):
                                                Contacts[(base + t + j) * cw + word] = (
                                                    out[word]
                                                )
                                            merged[0] = True
                                if not merged[0]:
                                    for word in T.serial(cw):
                                        Contacts[(base + t + write[0]) * cw + word] = (
                                            im[word]
                                        )
                                    write[0] = write[0] + 1
                        for i in T.serial(write[0], n[0]):
                            Contacts[(base + t + i) * cw + co + mf["len"]] = T.uint32(0)

    return one
