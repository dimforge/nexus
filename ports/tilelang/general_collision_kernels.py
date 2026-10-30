"""Resident native narrow phase with stable composite ordering and reduction."""

import broadphase_kernels as broad
import contact_plan_kernels as plan
import deferred_kernels as deferred
import narrow_kernels as narrow
import radix_kernels as radix
import scene_broadphase

EXTRA = (
    "PfmPairs",
    "Indices",
    "PfmIdentity",
    "PfmOrder",
    "PfmKeys",
    "PfmSortedKeys",
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
    *(f"Config{i}" for i in range(8)),
)
BUFFERS = tuple(dict.fromkeys((*broad.BUFFERS, *narrow.BUFFERS, *plan.BUFFERS, *EXTRA)))


def make_program(l, c, target):
    plans = []
    cache = {}
    sizes = c["buffer_words"]

    def case(mapping, **kwargs):
        return dict(c, buffer_words={k: sizes[v] for k, v in mapping.items()}, **kwargs)

    def identity(names):
        return {n: n for n in names}

    def append(stages):
        plans.extend(stages)

    mapping = identity(plan.BUFFERS)
    append(
        plan.make_program(
            l, case(mapping, kind="reset_narrow", count=1, groups=1, threads=64), target
        )
    )
    mapping = identity(broad.BUFFERS)
    append(scene_broadphase.make_program(l, c, target))

    def stage(kind):
        mapping = identity(deferred.BUFFERS)
        mapping.update(Order="PfmOrder", Keys="PfmKeys")
        if kind == "reduce_contacts":
            mapping["Keys"] = "PfmSortedKeys"
        dc = case(mapping, kind=kind, groups=c["contact_groups"])
        plans.append(
            {
                "name": kind,
                "kernel": deferred.make_kernel(l, dc, target, kind),
                "buffer_map": mapping,
                "blocks": c["contact_groups"],
                "threads": 64,
            }
        )

    stage("deferred_pairs")
    mapping = identity(plan.BUFFERS)
    append(
        plan.make_program(
            l, case(mapping, kind="contact_plan", count=1, groups=1, threads=1), target
        )
    )
    mapping = identity(narrow.BUFFERS)
    append(
        narrow.make_program(
            l,
            case(mapping, kind="analytic_contacts", groups=c["contact_groups"]),
            target,
        )
    )

    def sorting(kind, p, first):
        mapping = identity(radix.LOCAL_BUFFERS)
        mapping.update(
            NumKeys="PfmSortLen",
            BatchIds="RadixBatchIds",
            Aux="RadixAux",
            OutAux="RadixOutAux",
            Counts="RadixCounts",
            Config=f"Config{p}",
        )
        if p == 0:
            mapping.update(
                Keys="PfmKeys", Values="PfmIdentity" if first else "PfmOrder"
            )
        elif p % 2:
            mapping.update(Keys="PongKeys", Values="PongValues")
        else:
            mapping.update(Keys="PfmSortedKeys", Values="PfmOrder")
        if p % 2:
            mapping.update(OutKeys="PfmSortedKeys", OutValues="PfmOrder")
        else:
            mapping.update(OutKeys="PongKeys", OutValues="PongValues")
        rc = case(
            mapping, kind=kind, capacity=c["contact_storage"], batches=1, iterations=1
        )
        if kind not in cache:
            cache[kind] = radix.make_kernel(l, rc, target, kind)
        blocks, threads, _ = radix.dimensions(rc, kind)
        plans.append(
            {
                "name": f"{kind}_{len(plans)}",
                "kernel": cache[kind],
                "buffer_map": mapping,
                "blocks": blocks,
                "threads": threads,
            }
        )

    for first, kind in ((True, "pfm_subshape_keys"), (False, "pfm_permuted_keys")):
        stage(kind)
        sorting("radix_dispatch", 0, first)
        for p in range(8):
            for kind in radix.STAGES:
                sorting(kind, p, first)
    stage("deferred_manifolds")
    stage("reduce_contacts")
    return plans
