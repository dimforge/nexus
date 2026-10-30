"""Connected native ball/box frame over resident device storage."""

import body_kernels
import body_uniform_kernels
import bucket_kernels
import canonical_kernels
import collision_kernels
import colored_kernels
import coloring_kernels
import contact_kernels
import general_collision_kernels
import graph_kernels
import joint_kernels
import link_kernels
import memory_kernels
import pose_kernels
import primitive_kernels
from body_validation import BUFFERS as BODY
from contact_replay import BUFFERS as CONTACT
from link_replay import BUFFERS as LINK
from shaders.dynamics.solver import gpu_integrate_linearized


def make_program(l, c, target):
    # This controller mirrors the native scene oracle and calls only mirrored
    # shaders. Repetition belongs to the resident runtime's iteration count;
    # no body state is read back between stages or frames.
    plans = []
    cache = {}
    sizes = c["buffer_words"]
    n = c["body_count"]
    bs = c["body_storage"]
    nb = c["batches"]
    per = c["colliders_per_batch"]
    cap = c["contact_capacity"]
    cs = c["contact_storage"]
    bg = max((bs + 63) // 64, 1)

    def identity(names):
        return {x: x for x in names}

    def add(name, kernel, mapping, blocks=1, threads=64):
        plans.append(
            {
                "name": f"{name}_{len(plans)}",
                "kernel": kernel,
                "buffer_map": mapping,
                "blocks": blocks,
                "threads": threads,
            }
        )

    def program(name, sub, mapping=None):
        for stage in sub:
            plans.append(
                dict(
                    stage,
                    name=f"{name}_{len(plans)}",
                    buffer_map={
                        k: (mapping or {}).get(v, v)
                        for k, v in stage["buffer_map"].items()
                    },
                )
            )

    def local(mapping, **extra):
        return dict(c, buffer_words={k: sizes[v] for k, v in mapping.items()}, **extra)

    def body(kind):
        if kind not in cache:
            case = dict(
                c,
                kind=kind,
                count=n,
                storage=bs,
                slots=bs,
                bodies_per_batch=per,
                capacity_per_batch=per + 3,
                batches=nb,
            )
            cache[kind] = body_kernels.make_kernel(l, case, target)
        add(kind, cache[kind], identity(BODY), bg)

    add(
        "reset_pair_count",
        memory_kernels.reset(sizes["PairCount"], target),
        {"Data": "PairCount"},
        threads=64,
    )
    add(
        "body_coefficients",
        body_uniform_kernels.make_kernel(l, c, target),
        identity(["RbdParams", "Gravity", "Params"]),
        threads=1,
    )
    for kind in ["update_mprops", "cleanup", "init_bodies"]:
        body(kind)
    pm = {
        "BodyPoses": "Poses",
        "LocalPoses": "LocalPoses",
        "WorldPoses": "ColliderPoses",
        "Parents": "Parents",
    }
    add(
        "collider_poses",
        pose_kernels.make_kernel(
            l, dict(c, count=n, storage=bs, body_storage=bs), target
        ),
        pm,
        bg,
    )
    collision = (
        general_collision_kernels if c["kind"] == "general_scene" else collision_kernels
    )
    cm = identity(collision.BUFFERS)
    cm.update(Poses="ColliderPoses", Contacts="RawContacts")
    if c["kind"] == "general_scene":
        cm["Indices"] = "GeometryIndices"
    cc = local(
        cm,
        count=n,
        colliders=per,
        batches=nb,
        capacity=per + 3,
        storage=bs,
        dispatch=per * per * nb,
        groups=max((per * per * nb + 63) // 64, 1),
        contact_groups=1,
        iterations=1,
        vertices=c.get("vertex_count", 1),
    )
    program("collision", collision.make_program(l, cc, target), cm)
    cm = identity(canonical_kernels.BUFFERS)
    cm.update(Contacts="RawContacts", Output="Contacts", Counts="RadixCounts")
    cc = local(
        cm,
        kind="canonical_contacts",
        storage=cs,
        groups=1,
        passes=[4, 1, 2],
        iterations=1,
    )
    program("canonical", canonical_kernels.make_program(l, cc, target), cm)
    lm = identity(LINK)
    lm.update(BodyIsMultibody="IsMultibody", Params="RbdParams")
    lc = local(
        lm,
        count=cap,
        storage=cs,
        bodies=bs,
        ids_storage=sizes["OldIds"],
        old_storage=cs,
        state_storage=cs * 2,
        colliders=bs,
        iterations=1,
    )
    for kind in ["init_links", "match_links"]:
        add(
            kind,
            link_kernels.make_kernel(l, dict(lc, kind=kind), target),
            lm,
            max((cap + 63) // 64, 1),
        )
    gm = identity(["Contacts", "Counts", "Ids", "WorldMass", "IsMultibody", "Plan"])
    for i in range(4):
        gm[f"Aux{i}"] = f"AdjAux{i}"
        gm[f"Stride{i}"] = f"AdjStride{i}"
    gc = local(gm, kind="adjacency_build", count=cap, bodies=n, batches=1, groups=1)
    program("adjacency", graph_kernels.make_program(l, gc, target), gm)
    add(
        "stable_adjacency",
        primitive_kernels.make_kernel({"rows": n, "storage": sizes["Ids"]}, target),
        {"Counts": "Counts", "Ids": "Ids", "Sorted": "SortedIds"},
        max(n, 1),
    )
    from coloring_replay import BUFFERS as COLORING

    cm = identity(COLORING)
    cm["Ids"] = "SortedIds"
    cc = local(cm, kind="topo_bounded", count=cap, groups=1, threads=64, seeded=True)
    reset_case = dict(
        cc, kind="topo_reset", groups=max((cap + 63) // 64, 1), threads=64
    )
    add(
        "topo_reset",
        coloring_kernels.make_kernel(l, reset_case, target),
        cm,
        reset_case["groups"],
    )
    add(
        "seed_colors",
        link_kernels.make_kernel(l, dict(lc, kind="seed_colors"), target),
        lm,
        max((cap + 63) // 64, 1),
    )
    program("coloring", coloring_kernels.make_program(l, cc, target), cm)
    bm = identity(
        [
            "Colors",
            "Links",
            "Plan",
            "Buckets",
            "BatchIds",
            "SortedLinks",
            "Indices",
            "Stats",
            "Indirect",
        ]
    )
    for i in range(4):
        bm[f"Aux{i}"] = f"BucketAux{i}"
        bm[f"Stride{i}"] = f"BucketStride{i}"
    bc = local(
        bm,
        kind="bucket_build",
        bucket_count=c["color_stride"] * nb,
        batches=nb,
        groups=1,
        strides=[c["color_stride"] * nb, 1, 1, 1],
        levels=1,
    )
    program("buckets", bucket_kernels.make_program(l, bc, target), bm)
    pm = identity(CONTACT)
    pm["Links"] = "SortedLinks"
    pc = local(
        pm,
        count=cap,
        storage=cs,
        bodies=bs,
        states=cs * 2,
        tiles=(cs + 31) // 32,
        old_tiles=(cs + 31) // 32,
    )
    add(
        "prepare",
        contact_kernels.make_kernel(l, pc, target),
        pm,
        max((cap + 63) // 64, 1),
    )

    def joint(kind, col=0, bias=0):
        mapping = {name: name for name in joint_kernels.BUFFERS}
        mapping.update(
            Constraints="JointConstraints",
            ColorGroups="JointColors",
            Poses="SolverPoses",
            Velocities="SolverVelocities",
            Params="RbdParams",
            Color=f"Color{col}",
            UseBias="JointBias" if bias else "JointUnbiased",
        )
        if kind not in cache:
            cache[kind] = joint_kernels.make_kernel(
                l, local(mapping, kind=kind, count=c["joint_count"], groups=1), target
            )
        add(kind, cache[kind], mapping)

    if c["kind"] == "joint_scene":
        joint("joint_init")
    body("init_increments")

    def color(kind, col, mode):
        cm = identity(colored_kernels.BUFFERS)
        cm.update(
            Velocities="SolverVelocities",
            Poses="SolverPoses",
            Color=f"Color{col}",
            Mode=mode,
        )
        if kind not in cache:
            cache[kind] = colored_kernels.make_kernel(
                l, local(cm, count=cap, groups=1), target, kind
            )
        add(kind, cache[kind], cm)

    for _ in range(c["substeps"]):
        body("apply_increments")
        if c["kind"] == "joint_scene":
            joint("joint_update")
        for col in range(1, c["color_stride"] - 1):
            color("colored_warmstart", col, "Mode")
        for _ in range(c["pgs"]):
            if c["kind"] == "joint_scene":
                for col in range(c["joint_color_count"]):
                    joint("joint_solve", col, 1)
            for col in range(1, c["color_stride"] - 1):
                color("colored_solve", col, "Mode")
        if "integrate" not in cache:
            cache["integrate"] = gpu_integrate_linearized(
                l, dict(c, active_bodies=n, bodies=bs, tiles=(cs + 31) // 32), target
            )
        add(
            "integrate",
            cache["integrate"],
            {
                "Tiles": "Tiles",
                "Velocities": "SolverVelocities",
                "Poses": "SolverPoses",
                "Params": "Params",
            },
            bg,
        )
        if c["kind"] == "joint_scene":
            for col in range(c["joint_color_count"]):
                joint("joint_solve", col, 0)
        for col in range(1, c["color_stride"] - 1):
            color("colored_solve", col, "Color0")
    body("finalize")
    if c["kind"] in ("joint_scene", "general_scene"):
        from trajectory_kernels import make_kernel as trajectory_kernel

        add(
            "record_frame",
            trajectory_kernel(l, c, target),
            {
                name: name
                for name in ("Poses", "Velocities", "Trajectory", "FrameIndex")
            },
            max((n + 63) // 64, 1),
        )
        add(
            "advance_frame",
            memory_kernels.increment(sizes["FrameIndex"], target),
            {"Data": "FrameIndex"},
        )
    for source, dest in [
        ("Links", "OldLinks"),
        ("Counts", "OldCounts"),
        ("SortedIds", "OldIds"),
        ("Indices", "OldIndices"),
        ("Colors", "OldColors"),
        ("Tiles", "OldTiles"),
    ]:
        add(
            "persist_" + source,
            memory_kernels.copy(sizes[source], target),
            {"Source": source, "Dest": dest},
            max((sizes[source] + 63) // 64, 1),
        )
    add(
        "persist_states",
        memory_kernels.states(sizes["States"], cs * l["state_words"], target),
        {"Data": "States"},
        max((cs * l["state_words"] + 63) // 64, 1),
    )
    return plans
