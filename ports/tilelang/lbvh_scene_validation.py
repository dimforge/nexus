"""Connected scene plus independently reconstructed LBVH state and overlap set."""

import general_scene_validation as general
import joint_scene_validation as joint
import lbvh_program_validation as lbvh
import numpy as np
import scene_validation as analytic
from scene_broadphase import EXTRA, MAPPING


def core_module(c):
    return {"general_scene": general, "joint_scene": joint, "analytic_scene": analytic}[
        c["kind"]
    ]


def base_case(c, buffers):
    return dict(c, buffer_words={n: c["buffer_words"][n] for n in buffers})


def tree_case(c, b):
    return dict(
        c,
        kind="lbvh_program",
        colliders=c["colliders_per_batch"],
        capacity=c.get("body_capacity_per_batch", c["colliders_per_batch"] + 3),
        groups=1,
        storage=c["body_storage"],
        count=c.get("collider_count", c["body_count"]),
        vertices=c.get("vertex_count", 1),
        iterations=1,
        buffer_words={n: v.size for n, v in b.items()},
    )


def tree_arrays(r):
    return {n: r["ColliderPoses" if p == "Poses" else p] for n, p in MAPPING.items()}


def validate_case(l, c, b, *, capacity_variant=False):
    if c.get("broadphase") != "lbvh":
        raise ValueError("LBVH scene requires LBVH scheduling")
    module = core_module(c)
    if set(b) != set(module.BUFFERS) | set(EXTRA):
        raise ValueError("Invalid LBVH scene buffers")
    core = {n: b[n] for n in module.BUFFERS}
    module.validate_case(l, base_case(c, core), core, capacity_variant=capacity_variant)
    validate_tree(l, c, b)


def normalized_tree(l, c, b, restarted=False):
    arrays = tree_arrays(b)
    if c["kind"] == "runtime_scene":
        from native_geometry import lbvh_vertices

        arrays["Vertices"] = lbvh_vertices(l, b)
    if restarted:
        # The resident refit planner clears this live counter before use. Preserve
        # every guard word, all actual input storage and the captured tree.
        arrays["FrontierLen"] = arrays["FrontierLen"].copy()
        arrays["FrontierLen"][: c["batches"]] = 0
    return arrays


def validate_tree(l, c, b, restarted=False):
    arrays = normalized_tree(l, c, b, restarted)
    lbvh.validate_case(l, tree_case(c, arrays), arrays)


def check_outputs(l, c, b, a, e, k):
    validate_case(l, c, b)
    for r in (a, e):
        if set(r) != set(b) or any(
            r[n].dtype != np.uint32 or r[n].shape != v.shape for n, v in b.items()
        ):
            raise ValueError("Invalid LBVH scene outputs")
    module = core_module(c)
    core = tuple(module.BUFFERS)
    if c["kind"] == "analytic_scene":
        from scene_replay import check_outputs as compare
    else:
        compare = module.check_outputs
    errors = compare(
        l, base_case(c, core), *({n: r[n] for n in core} for r in (b, a, e)), k
    )
    check_trees(l, c, b, a, e, k)
    return errors


def check_trees(l, c, b, a, e, k, restarted=False):
    before = normalized_tree(l, c, b, restarted)
    lc = tree_case(c, before)
    for r in (a, e):
        tree = normalized_tree(l, c, r)
        initial = dict(before, Poses=tree["Poses"])
        # Live scratch can retain earlier frames; keep outer guards from upload.
        # Pair membership and refit cut membership are still checked independently.
        for n, limit in [
            ("Pairs", c["pair_capacity"] * l["pair_words"]),
            ("Frontier", lc["capacity"] * lc["batches"]),
        ]:
            initial[n] = before[n].copy()
            initial[n][:limit] = tree[n][:limit]
        lbvh.check_outputs(l, lc, initial, tree, tree, k)
