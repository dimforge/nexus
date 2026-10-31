"""Native scheduling metadata; physics stays in the mirrored shader modules.

Indirect grids never exceed the original compiled capacity. Its grid stride is
therefore safe: either there is at most one live item per lane, or the original
full grid is dispatched. This does not shrink cooperative/reduction launches.
"""

import tilelang.language as T

GRIDS = {"pairs": 0, "contacts": 1, "analytic": 2, "pfm": 3, "prepared": 4}
COLOR_BASE = 5


def grid_words(case):
    return 3 * (COLOR_BASE + case["color_stride"])


def make_grids(layout, case, kind, target="metal"):
    sizes = case["buffer_words"]
    cap = case["contact_capacity"]
    nb = case["batches"]
    stride = case["color_stride"]

    @T.macro
    def publish(PairCount, Plan, Buckets, Dispatches, lane):
        if kind == "pairs":
            if lane == 0:
                Dispatches[0] = T.max(
                    T.uint32(1),
                    T.ceildiv(T.min(PairCount[0], T.uint32(case["pair_capacity"])), 64),
                )
                Dispatches[1], Dispatches[2] = T.uint32(1), T.uint32(1)
        elif kind == "contacts":
            if lane < 3:
                count = T.alloc_local((1,), "uint32")
                count[0] = Plan[layout["plan"]["bound"]]
                if lane == 1:
                    count[0] = Plan[layout["plan"]["pfm_base"]]
                if lane == 2:
                    count[0] = Plan[layout["plan"]["pfm_len"]]
                slot = (1 + lane) * 3
                # Key generation publishes SortCount even for an empty frame.
                Dispatches[slot] = T.max(
                    T.uint32(1), T.ceildiv(T.min(count[0], T.uint32(cap)), 64)
                )
                Dispatches[slot + 1], Dispatches[slot + 2] = (
                    T.uint32(1),
                    T.uint32(1),
                )
        else:
            if lane == 0:
                Dispatches[12] = T.ceildiv(
                    T.min(Buckets[stride * nb - 1], T.uint32(cap)), 64
                )
                Dispatches[13], Dispatches[14] = T.uint32(1), T.uint32(1)
            if lane > 0 and lane < stride - 1:
                start = Buckets[lane * nb - 1]
                end = Buckets[(lane + 1) * nb - 1]
                slot = (COLOR_BASE + lane) * 3
                Dispatches[slot] = T.ceildiv(T.min(end - start, T.uint32(cap)), 64)
                Dispatches[slot + 1], Dispatches[slot + 2] = (
                    T.uint32(1),
                    T.uint32(1),
                )

    if target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            PairCount: T.Tensor((sizes["PairCount"],), "uint32"),
            Plan: T.Tensor((sizes["Plan"],), "uint32"),
            Buckets: T.Tensor((sizes["Buckets"],), "uint32"),
            Dispatches: T.Tensor((sizes["Dispatches"],), "uint32"),
        ):
            with A.Kernel(1):
                with A.SimtVF(threads=64):
                    publish(
                        PairCount, Plan, Buckets, Dispatches, A.get_thread_binding()
                    )
    else:

        @T.prim_func
        def kernel(
            PairCount: T.Tensor((sizes["PairCount"],), "uint32"),
            Plan: T.Tensor((sizes["Plan"],), "uint32"),
            Buckets: T.Tensor((sizes["Buckets"],), "uint32"),
            Dispatches: T.Tensor((sizes["Dispatches"],), "uint32"),
        ):
            with T.Kernel(1, threads=64):
                publish(PairCount, Plan, Buckets, Dispatches, T.get_thread_binding())

    return kernel


def apply_indirect_dispatches(plans, layout, case, target):
    if target not in ("metal", "cuda", "ascend"):
        raise ValueError("Device scheduling requires a GPU/NPU target")
    result = []

    def publish(kind):
        result.append(
            {
                "name": "dispatch_" + kind,
                "kernel": make_grids(layout, case, kind, target),
                "buffer_map": {
                    n: n for n in ("PairCount", "Plan", "Buckets", "Dispatches")
                },
                "blocks": 1,
                "threads": 64,
            }
        )

    for stage in plans:
        op, phase = stage.get("operation", ""), stage.get("phase", "")
        slot = None
        if phase == "collision":
            if op == "deferred_pairs":
                slot = GRIDS["pairs"]
            if op == "analytic_contacts":
                slot = GRIDS["analytic"]
            if op in (
                "deferred_manifolds",
                "reduce_contacts",
                "pfm_subshape_keys",
                "pfm_permuted_keys",
            ):
                slot = GRIDS["pfm"]
        if phase == "canonical" and (op.startswith(("keys_", "permute_"))):
            slot = GRIDS["contacts"]
        if phase in (
            "init_links",
            "match_links",
            "topo_reset",
            "seed_colors",
            "persist_states",
        ):
            slot = GRIDS["contacts"]
        if phase == "coloring" and stage["threads"] == 64:
            slot = GRIDS["contacts"]
        if phase in ("adjacency", "buckets") and op in (
            "adjacency_count",
            "adjacency_scatter",
            "bucket_count",
            "bucket_scatter",
        ):
            slot = GRIDS["contacts"]
        if phase in ("prepare", "scale_impulses"):
            slot = GRIDS["prepared"]
        if phase in ("colored_solve", "colored_warmstart"):
            from colored_kernels import fuse_scene_colors

            if not fuse_scene_colors(case, target):
                slot = COLOR_BASE + int(
                    stage["buffer_map"]["Color"].removeprefix("Color")
                )
        if slot is not None:
            stage = dict(stage, indirect={"buffer": "Dispatches", "word": slot * 3})
            if slot > COLOR_BASE and case.get("color_checks"):
                stage["frame_condition"] = {
                    "gate": "colors",
                    "minimum": slot - COLOR_BASE,
                }
        result.append(stage)
        if stage.get("capacity_gate", {}).get("id") == "pairs":
            publish("pairs")
        if phase == "collision" and op == "contact_plan":
            publish("contacts")
        if phase == "buckets" and op == "bucket_dispatch":
            publish("colors")
    return result


def copy_live_states(layout, case, target="metal"):
    """The ABI places both state generations in one allocation; copy live rows."""
    sizes = case["buffer_words"]
    half = case["contact_storage"] * layout["state_words"]
    groups = max(1, (case["contact_capacity"] + 63) // 64)

    @T.macro
    def copy_rows(States, Plan, row):
        if row < Plan[layout["plan"]["bound"]]:
            for word in T.serial(layout["state_words"]):
                i = row * layout["state_words"] + word
                States[i] = States[half + i]

    if target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            States: T.Tensor((sizes["States"],), "uint32"),
            Plan: T.Tensor((sizes["Plan"],), "uint32"),
        ):
            with A.Kernel(groups) as group:
                with A.SimtVF(threads=64):
                    copy_rows(States, Plan, group * 64 + A.get_thread_binding())
    else:

        @T.prim_func
        def kernel(
            States: T.Tensor((sizes["States"],), "uint32"),
            Plan: T.Tensor((sizes["Plan"],), "uint32"),
        ):
            with T.Kernel(groups, threads=64) as group:
                copy_rows(States, Plan, group * 64 + T.get_thread_binding())

    return kernel
