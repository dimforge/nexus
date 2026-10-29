"""Counterpart of src_rbd_shaders/dynamics/contact_links.rs (3D)."""

import tilelang.language as T
from kernel_support import dot, f32, quat_mul, quat_rotate


def gpu_init_contact_links(layout, case):
    l, c, m = layout["link"], layout["contact"], layout["manifold"]

    @T.macro
    def one(Contacts, Links, BodyIsMultibody, i):
        if i < case["count"]:
            cb = i * layout["contact_words"]
            lb = i * layout["link_words"]
            a = T.cast(Contacts[cb + c["bodies"]], "int32")
            b = T.cast(Contacts[cb + c["bodies"] + 1], "int32")
            # Reduce flags to booleans before combining: keep high bits, avoid u32 range analysis.
            ordinary = (BodyIsMultibody[a] == T.uint32(0)) & (
                BodyIsMultibody[b] == T.uint32(0)
            )
            Links[lb + l["len"]] = T.if_then_else(
                ordinary, Contacts[cb + c["contact"] + m["len"]], T.uint32(0)
            )
            Links[lb + l["solver_body_a"]] = T.cast(a, "uint32")
            Links[lb + l["solver_body_b"]] = T.cast(b, "uint32")
            Links[lb + l["vel_slot_a"]] = T.cast(a, "uint32")
            Links[lb + l["vel_slot_b"]] = T.cast(b, "uint32")
            Links[lb + l["warmstart_collider_a"]] = Contacts[cb + c["colliders"]]
            Links[lb + l["warmstart_collider_b"]] = Contacts[cb + c["colliders"] + 1]
            Links[lb + l["warmstart_subshape"]] = Contacts[cb + c["subshape"]]
            Links[lb + l["previous_constraint"]] = T.uint32(0xFFFFFFFF)
            Links[lb + l["previous_constraint_index"]] = T.uint32(0xFFFFFFFF)
            Links[lb + l["recycled"]] = T.uint32(0)
            # Exact f32 one; the C backend cannot reinterpret an rvalue literal.
            Links[lb + l["mass_scale_a"]] = T.uint32(0x3F800000)
            Links[lb + l["mass_scale_b"]] = T.uint32(0x3F800000)
            Links[lb + l["restitution"]] = Contacts[cb + c["restitution"]]
            Links[lb + l["contact"]] = T.cast(i, "uint32")
            Links[lb + l["_padding"]] = T.uint32(0)

    return one


def find_previous_constraint(layout):
    l = layout["link"]

    @T.macro
    def one(Counts, Ids, OldLinks, Links, i, previous):
        lb = i * layout["link_words"]
        a = T.cast(Links[lb + l["solver_body_a"]], "int32")
        b = T.cast(Links[lb + l["solver_body_b"]], "int32")
        first_a = T.alloc_local((1,), "int32")
        first_b = T.alloc_local((1,), "int32")
        first_a[0] = 0
        first_b[0] = 0
        if a != 0:
            first_a[0] = T.cast(Counts[a - 1], "int32")
        if b != 0:
            first_b[0] = T.cast(Counts[b - 1], "int32")
        last_a = T.cast(Counts[a], "int32")
        last_b = T.cast(Counts[b], "int32")
        len_a = last_a - first_a[0]
        len_b = last_b - first_b[0]
        use_a = (len_b == 0) | ((len_a != 0) & (len_a < len_b))
        first = T.if_then_else(use_a, first_a[0], first_b[0])
        last = T.if_then_else(use_a, last_a, last_b)
        previous[0] = -1
        for entry in T.serial(first, last):
            if previous[0] == -1:
                cid = T.cast(Ids[entry], "int32")
                old = T.cast(cid, "int32") * layout["link_words"]
                if (
                    (
                        OldLinks[old + l["solver_body_a"]]
                        == Links[lb + l["solver_body_a"]]
                    )
                    & (
                        OldLinks[old + l["solver_body_b"]]
                        == Links[lb + l["solver_body_b"]]
                    )
                    & (
                        OldLinks[old + l["warmstart_collider_a"]]
                        == Links[lb + l["warmstart_collider_a"]]
                    )
                    & (
                        OldLinks[old + l["warmstart_collider_b"]]
                        == Links[lb + l["warmstart_collider_b"]]
                    )
                    & (
                        OldLinks[old + l["warmstart_subshape"]]
                        == Links[lb + l["warmstart_subshape"]]
                    )
                ):
                    previous[0] = cid

    return one


def rot_cos(layout):
    q = layout["pose_rotation"]

    @T.macro
    def one(A, a, B, b, result):
        c = (
            f32(A, a + q) * f32(B, b + q)
            + f32(A, a + q + 1) * f32(B, b + q + 1)
            + f32(A, a + q + 2) * f32(B, b + q + 2)
            + f32(A, a + q + 3) * f32(B, b + q + 3)
        )
        result[0] = 2.0 * c * c - 1.0

    return one


def can_recycle(layout):
    s = layout["state"]
    cosine = rot_cos(layout)

    @T.macro
    def relative_pose(A, a, B, b, q, t):
        # Rust uses a.inverse() * b, not inv_mul: preserve its two rotations and add.
        inverse = T.alloc_local((4,), "float32")
        qb = T.alloc_local((4,), "float32")
        negative_a = T.alloc_local((3,), "float32")
        translation_b = T.alloc_local((3,), "float32")
        ta = T.alloc_local((3,), "float32")
        tb = T.alloc_local((3,), "float32")
        for d in T.serial(4):
            inverse[d] = f32(A, a + layout["pose_rotation"] + d) * T.if_then_else(
                d < 3, -1.0, 1.0
            )
            qb[d] = f32(B, b + layout["pose_rotation"] + d)
        for d in T.serial(3):
            negative_a[d] = -f32(A, a + layout["pose_translation"] + d)
            translation_b[d] = f32(B, b + layout["pose_translation"] + d)
        quat_rotate(inverse, negative_a, ta)
        quat_rotate(inverse, translation_b, tb)
        quat_mul(inverse, qb, q)
        for d in T.serial(3):
            t[d] = ta[d] + tb[d]

    @T.macro
    def one(States, old, Poses, a, b, recycled):
        qo = T.alloc_local((4,), "float32")
        qn = T.alloc_local((4,), "float32")
        to = T.alloc_local((3,), "float32")
        tn = T.alloc_local((3,), "float32")
        delta = T.alloc_local((3,), "float32")
        drot = T.alloc_local((4,), "float32")
        cos_a = T.alloc_local((1,), "float32")
        cos_b = T.alloc_local((1,), "float32")
        relative_pose(States, old + s["pose_a"], States, old + s["pose_b"], qo, to)
        relative_pose(Poses, a, Poses, b, qn, tn)
        for d in T.serial(3):
            delta[d] = tn[d] - to[d]
            qo[d] = -qo[d]
        quat_mul(qn, qo, drot)
        trans = T.sqrt(dot(delta, delta))
        chord = 2.0 * T.sqrt(dot(drot, drot)) * f32(States, old + s["max_extent"])
        cosine(States, old + s["pose_a"], Poses, a, cos_a)
        cosine(States, old + s["pose_b"], Poses, b, cos_b)
        drift = f32(States, old + s["max_drift"])
        recycled[0] = T.cast(
            (drift > 0.0)
            & (trans + chord <= drift)
            & (T.min(cos_a[0], cos_b[0]) > 0.98),
            "uint32",
        )

    return one


def gpu_match_contact_links(layout, case):
    l, c, s = layout["link"], layout["contact"], layout["state"]
    match = find_previous_constraint(layout)
    reuse = can_recycle(layout)

    @T.macro
    def one(
        Contacts,
        Links,
        OldCounts,
        OldIds,
        OldLinks,
        OldIndices,
        States,
        ColliderPoses,
        Params,
        i,
    ):
        if i < case["count"]:
            lb = i * layout["link_words"]
            if Links[lb + l["len"]] != 0:
                cb = i * layout["contact_words"]
                a = (
                    T.cast(Contacts[cb + c["colliders"]], "int32")
                    * layout["pose_words"]
                )
                b = (
                    T.cast(Contacts[cb + c["colliders"] + 1], "int32")
                    * layout["pose_words"]
                )
                dst = (case["new_base"] + i) * layout["state_words"]
                previous = T.alloc_local((1,), "int32")
                recycled = T.alloc_local((1,), "uint32")
                index = T.alloc_local((1,), "uint32")
                match(OldCounts, OldIds, OldLinks, Links, i, previous)
                recycled[0] = T.uint32(0)
                index[0] = T.uint32(0xFFFFFFFF)
                if previous[0] != -1:
                    old = (case["old_base"] + T.cast(previous[0], "int32")) * layout[
                        "state_words"
                    ]
                    index[0] = OldIndices[T.cast(previous[0], "int32")]
                    reuse(States, old, ColliderPoses, a, b, recycled)
                    if recycled[0] != 0:
                        for d in T.serial(layout["state_words"]):
                            States[dst + d] = States[old + d]
                if recycled[0] == 0:
                    for d in T.serial(layout["pose_words"]):
                        States[dst + s["pose_a"] + d] = ColliderPoses[a + d]
                        States[dst + s["pose_b"] + d] = ColliderPoses[b + d]
                    States[dst + s["colliders"]] = Contacts[cb + c["colliders"]]
                    States[dst + s["colliders"] + 1] = Contacts[cb + c["colliders"] + 1]
                    States[dst + s["max_extent"]] = Contacts[cb + c["recycle_extent"]]
                    States[dst + s["max_drift"]] = T.uint32(0)
                    if f32(Contacts, cb + c["recycle_extent"]) >= 0.0:
                        States[dst + s["max_drift"]] = T.reinterpret(
                            f32(Params, layout["recycle_params"]["normalized_distance"])
                            * f32(Params, layout["recycle_params"]["length_unit"]),
                            "uint32",
                        )
                Links[lb + l["previous_constraint"]] = T.cast(previous[0], "uint32")
                Links[lb + l["previous_constraint_index"]] = index[0]
                Links[lb + l["recycled"]] = recycled[0]

    return one


def gpu_seed_colors_from_warmstart(layout, case):
    l = layout["link"]

    @T.macro
    def one(Links, OldColors, Colors, Colored, i):
        if i < case["count"]:
            lb = i * layout["link_words"]
            if Links[lb + l["len"]] != 0:
                # Check u32::MAX before casting: signed -1 comparisons may be optimized away.
                previous = Links[lb + l["previous_constraint"]]
                if previous != T.uint32(0xFFFFFFFF):
                    color = OldColors[T.cast(previous, "int32")]
                    if (color > 0) & (color < 64):
                        Colors[i] = color
                        Colored[i] = T.uint32(1)

    return one
