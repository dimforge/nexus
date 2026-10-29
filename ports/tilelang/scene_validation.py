"""ABI and launch preconditions for the connected native analytic scene path."""

from body_validation import BUFFERS as BODY_BUFFERS

BASE_BUFFERS = (
    *BODY_BUFFERS,
    "LocalPoses",
    "ColliderPoses",
    "Shapes",
    "Materials",
    "Parents",
    "Filters",
    "Groups",
    "Vertices",
    "Aabbs",
    "Pairs",
    "PairCount",
    "PfmCount",
    "Plan",
    "PfmSortLen",
    "ContactsIndirect",
    "PfmIndirect",
    "MbIndirect",
    "RawContacts",
    "Contacts",
    "Ids",
    "SortedIds",
    "IsMultibody",
    "BodyGroup",
    "Links",
    "OldLinks",
    "OldCounts",
    "OldIds",
    "OldIndices",
    "OldColors",
    "States",
    "Colors",
    "Rands",
    "Colored",
    "Pending",
    "NumColors",
    "ColoringIndirect",
    "Tiles",
    "OldTiles",
    "SortedLinks",
    "Indices",
    "Buckets",
    "Stats",
    "Indirect",
    "InOrder",
    "Keys",
    "Values",
    "SortCount",
    "BatchIds",
    "RbdParams",
    "Gravity",
    "Selector",
    "CurrentColor",
    "Mode",
    "SortKeys",
    "PongKeys",
    "PongValues",
    "RadixAux",
    "RadixOutAux",
    "RadixBatchIds",
    "RadixCounts",
    "Reduced",
    "SortGrid",
    "ReduceGrid",
    "Config",
    "InitConfig",
)
BUFFERS = (
    tuple(dict.fromkeys(BASE_BUFFERS))
    + tuple(
        f"{prefix}{i}"
        for prefix in ("AdjAux", "AdjStride", "BucketAux", "BucketStride")
        for i in range(4)
    )
    + tuple(f"{prefix}{i}" for prefix in ("Color", "Config") for i in range(8))
)
READONLY = (
    "LocalMass",
    "LocalPoses",
    "Shapes",
    "Materials",
    "Parents",
    "Filters",
    "Groups",
    "Vertices",
    "IsMultibody",
    "BodyGroup",
    "BatchIds",
    "RbdParams",
    "Gravity",
    "Selector",
    "CurrentColor",
    "Mode",
    "RadixAux",
    "RadixOutAux",
    "RadixBatchIds",
    "Config",
    "InitConfig",
    "Rands",
    *(f"{p}{i}" for p in ("AdjStride", "BucketStride") for i in range(4)),
    *(f"{p}{i}" for p in ("Color", "Config") for i in range(8)),
)


def body_case(case, kind):
    return dict(
        case,
        kind=kind,
        count=case["body_count"],
        slots=case["body_storage"],
        storage=case["body_storage"],
        bodies_per_batch=case["colliders_per_batch"],
        capacity_per_batch=case["colliders_per_batch"] + 3,
    )


def buffer_sizes(l, c):
    n, bs, cs, pc = (
        c[k] for k in ("body_count", "body_storage", "contact_storage", "pair_capacity")
    )
    sizes = {}
    for names, records, width in (
        (("Poses", "SolverPoses", "ColliderPoses", "LocalPoses"), bs, l["pose_words"]),
        (("Velocities", "SolverVelocities", "Increments"), bs, l["velocity_words"]),
        (("LocalMass",), bs, l["local_words"]),
        (("WorldMass",), bs, l["world_words"]),
        (("Shapes",), bs, l["shape_words"]),
        (("Materials",), bs, l["material_words"]),
        (("Filters",), bs, 2),
        (("Groups",), bs, l["group_words"]),
        (("Aabbs",), bs, l["aabb_words"]),
        (("Pairs",), pc + 3, l["pair_words"]),
        (("Contacts", "RawContacts"), cs, l["contact_words"]),
        (("Links", "OldLinks", "SortedLinks"), cs, l["link_words"]),
        (("Tiles", "OldTiles"), (cs + 31) // 32, l["tile_words"]),
        (("States",), 2 * cs, l["state_words"]),
        (
            ("Parents", "Counts", "OldCounts", "FirstSlot", "IsMultibody", "BodyGroup"),
            bs,
            1,
        ),
        (("Ids", "SortedIds", "OldIds"), 2 * c["contact_capacity"] + 3, 1),
        (
            (
                "Indices",
                "OldIndices",
                "Colors",
                "OldColors",
                "Rands",
                "Colored",
                "Pending",
                "InOrder",
                "Keys",
                "Values",
                "SortKeys",
                "PongKeys",
                "PongValues",
                "RadixAux",
                "RadixOutAux",
                "RadixBatchIds",
            ),
            cs,
            1,
        ),
    ):
        sizes.update({name: records * width for name in names})
    sizes.update(
        Vertices=l["vertex_words"],
        PairCount=3,
        PfmCount=3,
        Plan=2 * l["plan_words"],
        PfmSortLen=3,
        ContactsIndirect=6,
        PfmIndirect=6,
        MbIndirect=6,
        NumColors=2,
        ColoringIndirect=9,
        Buckets=8 * c["batches"],
        Stats=l["stats_words"],
        Indirect=24,
        SortCount=1,
        BatchIds=l["batch_words"],
        RbdParams=l["rbd_params_words"],
        Gravity=4,
        HubCounts=l["hub_words"],
        Params=6,
        Selector=2,
        CurrentColor=2,
        Mode=2,
        RadixCounts=16 * ((cs + 1023) // 1024) + 3,
        Reduced=1027,
        SortGrid=6,
        ReduceGrid=6,
        Config=l["sort_uniforms_words"],
        InitConfig=l["sort_uniforms_words"],
    )
    for i in range(4):
        sizes.update(
            {
                f"AdjAux{i}": (n + 255) // 256 + 3 if i == 0 else 4,
                f"BucketAux{i}": 4,
                f"AdjStride{i}": 2,
                f"BucketStride{i}": 2,
            }
        )
    for i in range(8):
        sizes.update({f"Color{i}": 2, f"Config{i}": l["sort_uniforms_words"]})
    return sizes


def validate_case(l, c, a):
    import numpy as np
    from body_validation import validate_case as validate_body

    fields = (
        "body_count",
        "body_storage",
        "colliders_per_batch",
        "batches",
        "columns",
        "levels_scene",
        "contact_capacity",
        "contact_storage",
        "pair_capacity",
        "pair_storage",
        "groups",
        "substeps",
        "pgs",
        "color_stride",
        "rounds",
        "frames",
        "iterations",
        "old_base",
        "new_base",
    )
    if c["kind"] != "analytic_scene" or any(
        type(c[k]) is not int or not 0 <= c[k] < 2**20 for k in fields
    ):
        raise ValueError("Invalid scene dimensions")
    n, bs, cs, nb, per = (
        c[k]
        for k in (
            "body_count",
            "body_storage",
            "contact_storage",
            "batches",
            "colliders_per_batch",
        )
    )
    if not (
        1 <= nb <= 3
        and c["color_stride"] == 8
        and c["rounds"] == 16
        and c["groups"] == 1
        and c["substeps"] == 4
        and c["pgs"] == 2
        and 1 <= c["frames"] <= 120
        and c["iterations"] == c["frames"]
    ):
        raise ValueError("Unsupported scene schedule")
    if (
        n != per * nb
        or bs != (per + 3) * nb
        or c["pair_capacity"] != per * per * nb
        or c["pair_storage"] != c["pair_capacity"] + 3
        or c["contact_capacity"] != 2 * c["pair_capacity"]
        or cs != c["contact_capacity"] + 3
        or c["old_base"] != 0
        or c["new_base"] != cs
    ):
        raise ValueError("Invalid scene capacity or persistence ranges")
    if c["scene"] not in (
        "empty",
        "ground",
        "stack",
        "free",
        "friction",
        "boxes",
        "impact",
    ):
        raise ValueError("Unknown scene geometry")
    if per != (0 if c["scene"] == "empty" else 1 + c["columns"] * c["levels_scene"]):
        raise ValueError("Scene dimensions differ from bodies")
    if c["strides"] != [n, (n + 255) // 256, 1, 1] or c["levels"] != (
        1 if n <= 256 else 2
    ):
        raise ValueError("Invalid scene adjacency scan")
    sizes = buffer_sizes(l, c)
    if (
        set(a) != set(BUFFERS)
        or c["buffer_words"] != sizes
        or any(
            size <= 0
            or size >= 2**31
            or a[name].dtype != np.uint32
            or a[name].shape != (size,)
            for name, size in sizes.items()
        )
    ):
        raise ValueError("Invalid scene storage manifest")
    inputs = {name: a[name] for name in BODY_BUFFERS}
    inputs["Params"] = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32).view(
        np.uint32
    )
    validate_body(l, body_case(c, "update_mprops"), inputs)
    b = l["batch"]
    for field, value in {
        "num_batches": nb,
        "bodies_len": per,
        "colliders_len": per,
        "colliders_batch_capacity": per + 3,
        "collision_pairs_capacity": c["pair_capacity"],
        "contacts_capacity": c["contact_capacity"],
        "solver_color_buckets_stride": 8,
        "deterministic": 1,
        "contact_sort_collider_shift": 0,
    }.items():
        if a["BatchIds"][b[field]] != value:
            raise ValueError("Scene batch uniform differs from specialization")
    p = a["RbdParams"].view(np.float32)
    f = l["rbd_params"]
    if (
        not np.isfinite(p).all()
        or p[f["dt"]] <= 0
        or p[f["length_unit"]] <= 0
        or p[f["normalized_prediction_distance"]] < 0
        or p[f["warmstart_coefficient"]] != 1
        or a["RbdParams"][f["num_solver_iterations"]] != c["substeps"]
        or a["RbdParams"][f["num_internal_pgs_iterations"]] != c["pgs"]
        or a["RbdParams"][f["friction_in_bias_pass"]] != 1
        or a["Mode"][0] != 2
    ):
        raise ValueError("Invalid scene solver parameters")
    for prefix in ("contact", "static_contact"):
        if (
            p[f[prefix + "_natural_frequency"]] < 0
            or p[f[prefix + "_damping_ratio"]] <= 0
        ):
            raise ValueError("Invalid scene softness")
    for i in range(8):
        if (
            a[f"Color{i}"][0] != i
            or a[f"Config{i}"][l["sort_uniforms"]["shift"]] != 4 * i
        ):
            raise ValueError("Invalid scene color or radix uniform")
    for name in ("Config", "InitConfig", *(f"Config{i}" for i in range(8))):
        if (
            a[name][l["sort_uniforms"]["max_keys_per_batch"]] != cs
            or a[name][l["sort_uniforms"]["has_aux"]] != 0
        ):
            raise ValueError("Invalid scene radix capacity")
    for i in range(4):
        if (
            a[f"AdjStride{i}"][0] != c["strides"][i]
            or a[f"BucketStride{i}"][0] != ([8 * nb, 1, 1, 1][i])
        ):
            raise ValueError("Invalid device scan uniform")
    if (
        np.any(a["IsMultibody"])
        or np.any(a["Plan"])
        or a["PairCount"][0] != 0
        or a["PfmCount"][0] != 0
    ):
        raise ValueError("Scene requires fresh ordinary-body state")
    if not np.array_equal(a["Parents"], np.arange(bs, dtype=np.uint32)):
        raise ValueError("Scene requires one identity-attached collider per body")
    shapes = a["Shapes"].reshape(-1, l["shape_words"])
    tags = shapes[:n, l["shape"]["a"] + 3]
    if np.any(
        (tags != l["shape_types"]["ball"]) & (tags != l["shape_types"]["cuboid"])
    ):
        raise ValueError("Unsupported scene shape")
    mass = a["LocalMass"].view(np.float32).reshape(-1, l["local_words"])
    for field in ("local_inv_mass", "local_inv_inertia"):
        v = mass[:n, l[field] : l[field] + 3]
        if (
            np.any(v < 0)
            or np.any(v[: min(nb, n)] != 0)
            or np.any(v[min(nb, n) :] <= 0)
        ):
            raise ValueError("Scene static/dynamic classification")
    # Physical acceptance is for these unit-mass, half-meter fixture shapes,
    # not an implicit promise that the same absolute bounds fit arbitrary units.
    if not np.array_equal(
        a["Gravity"].view(np.float32), np.array([0.0, -9.81, 0.0, 0.0], np.float32)
    ):
        raise ValueError("Scene fixture gravity changed")
    if p[f["dt"]] != np.float32(1 / 240) or p[f["length_unit"]] != 1:
        raise ValueError("Scene fixture timestep or units changed")
    for i in range(n):
        off = l["shape"]["a"]
        shape = shapes[i]
        values = shape.view(np.float32)
        fixed = i < nb
        box = fixed or c["scene"] == "boxes"
        if shape[off + 3] != l["shape_types"]["cuboid" if box else "ball"]:
            raise ValueError("Scene fixture shape type changed")
        expected = np.array([20.0, 0.5, 20.0] if fixed else [0.5, 0.5, 0.5], np.float32)
        width = 3 if box else 1
        if not np.array_equal(values[off : off + width], expected[:width]):
            raise ValueError("Scene fixture shape dimensions changed")
        if not fixed:
            expected_inertia = 6.0 if box else 10.0
            if np.any(
                mass[i, l["local_inv_mass"] : l["local_inv_mass"] + 3] != 1
            ) or np.any(
                mass[i, l["local_inv_inertia"] : l["local_inv_inertia"] + 3]
                != expected_inertia
            ):
                raise ValueError("Scene fixture mass changed")
    if np.any(mass[:, l["local_com"] : l["local_com"] + 3] != 0):
        raise ValueError("Scene physical checks require centered mass")
