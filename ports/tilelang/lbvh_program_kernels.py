"""Resident Morton generation, stable radix sorting, tree build/refit and traversal."""

import lbvh_kernels as lbvh
import radix_kernels as radix


def make_program(l, c, target):
    plans = []
    sizes = c["buffer_words"]
    cache = {}

    def stage(kind):
        mapping = {n: n for n in lbvh.BUFFERS}
        if kind == "lbvh_build":
            mapping["Keys"] = "SortKeys"
        sc = dict(c, kind=kind, buffer_words={n: sizes[p] for n, p in mapping.items()})
        blocks, threads, _ = lbvh.dimensions(sc)
        plans.append(
            {
                "name": kind,
                "kernel": lbvh.make_kernel(l, sc, target),
                "buffer_map": mapping,
                "blocks": blocks,
                "threads": threads,
            }
        )

    for kind in ("lbvh_domain", "lbvh_domain_merge", "lbvh_morton"):
        stage(kind)

    def sorting(kind, p):
        mapping = {n: n for n in radix.LOCAL_BUFFERS}
        mapping.update(
            Config=f"Config{p}",
            Aux="RadixAux",
            OutAux="RadixOutAux",
            BatchIds="RadixBatchIds",
            Counts="RadixCounts",
        )
        if p == 0:
            mapping.update(Keys="Keys", Values="Identity")
        elif p % 2:
            mapping.update(Keys="PongKeys", Values="PongValues")
        else:
            mapping.update(Keys="SortKeys", Values="Sorted")
        if p % 2:
            mapping.update(OutKeys="SortKeys", OutValues="Sorted")
        else:
            mapping.update(OutKeys="PongKeys", OutValues="PongValues")
        rc = dict(c, kind=kind, buffer_words={n: sizes[v] for n, v in mapping.items()})
        if kind not in cache:
            cache[kind] = radix.make_kernel(l, rc, target, kind)
        blocks, threads, _ = radix.dimensions(rc, kind)
        plans.append(
            {
                "name": f"{kind}_{p}",
                "kernel": cache[kind],
                "buffer_map": mapping,
                "blocks": blocks,
                "threads": threads,
            }
        )

    sorting("radix_dispatch", 0)
    for p in range(8):
        for kind in radix.STAGES:
            sorting(kind, p)
    for kind in (
        "lbvh_build",
        "lbvh_escapes",
        "lbvh_leaves",
        "lbvh_refit_chunks",
        "lbvh_refit_plan",
        "lbvh_refit_frontier",
        "lbvh_reset",
        "lbvh_pairs",
        "flat_list_dispatch",
    ):
        stage(kind)
    return plans
