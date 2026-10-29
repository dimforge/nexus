"""Counterpart of broad_phase/narrow_phase.rs; native analytic first pass."""

import tilelang.language as T
from kernel_support import f32, quat_rotate, quat_mul
from ..queries.contact import ball_ball, ball_convex, convex_ball, combine_coefficient
from ..shapes.shape import primitive_local_extent
from ..queries.contact import cuboid_cuboid


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
