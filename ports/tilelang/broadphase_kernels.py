"""Target launches and ordered AABB/pair composition, with no host geometry."""

import tilelang.language as T
from shaders.broad_phase import brute_force as B

BUFFERS = (
    "Poses",
    "Shapes",
    "Aabbs",
    "BatchIds",
    "Vertices",
    "Pairs",
    "PairCount",
    "Groups",
    "Filters",
    "RbdParams",
)


def make_kernel(layout, case, target):
    kind = case["kind"]
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = [case["buffer_words"][n] for n in BUFFERS]
    groups = case["groups"]
    bounds = B.gpu_bf_compute_aabbs(layout, case)
    pairs = B.gpu_bf_find_pairs(layout, case)
    serial = B.gpu_bf_find_pairs_serial(layout, case)

    @T.macro
    def one(
        Poses,
        Shapes,
        Aabbs,
        BatchIds,
        Vertices,
        Pairs,
        PairCount,
        Groups,
        Filters,
        RbdParams,
        gid,
    ):
        if kind == "world_aabbs":
            bounds(Poses, Shapes, Aabbs, BatchIds, Vertices, gid)
        elif kind == "brute_pairs_serial":
            serial(Aabbs, Pairs, PairCount, Groups, BatchIds, Filters, RbdParams, gid)
        else:
            pairs(Aabbs, Pairs, PairCount, Groups, BatchIds, Filters, RbdParams, gid)

    if target == "c":

        @T.prim_func
        def kernel(
            Poses: T.Tensor((n0,), "uint32"),
            Shapes: T.Tensor((n1,), "uint32"),
            Aabbs: T.Tensor((n2,), "uint32"),
            BatchIds: T.Tensor((n3,), "uint32"),
            Vertices: T.Tensor((n4,), "uint32"),
            Pairs: T.Tensor((n5,), "uint32"),
            PairCount: T.Tensor((n6,), "uint32"),
            Groups: T.Tensor((n7,), "uint32"),
            Filters: T.Tensor((n8,), "uint32"),
            RbdParams: T.Tensor((n9,), "uint32"),
        ):
            with T.Kernel(1):
                for gid in T.serial(groups * 64):
                    one(
                        Poses,
                        Shapes,
                        Aabbs,
                        BatchIds,
                        Vertices,
                        Pairs,
                        PairCount,
                        Groups,
                        Filters,
                        RbdParams,
                        gid,
                    )
    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            Poses: A.Tensor((n0,), "uint32"),
            Shapes: A.Tensor((n1,), "uint32"),
            Aabbs: A.Tensor((n2,), "uint32"),
            BatchIds: A.Tensor((n3,), "uint32"),
            Vertices: A.Tensor((n4,), "uint32"),
            Pairs: A.Tensor((n5,), "uint32"),
            PairCount: A.Tensor((n6,), "uint32"),
            Groups: A.Tensor((n7,), "uint32"),
            Filters: A.Tensor((n8,), "uint32"),
            RbdParams: A.Tensor((n9,), "uint32"),
        ):
            with A.Kernel(groups) as group:
                with A.SimtVF(threads=64):
                    one(
                        Poses,
                        Shapes,
                        Aabbs,
                        BatchIds,
                        Vertices,
                        Pairs,
                        PairCount,
                        Groups,
                        Filters,
                        RbdParams,
                        group * 64 + A.get_thread_binding(),
                    )
    else:

        @T.prim_func
        def kernel(
            Poses: T.Tensor((n0,), "uint32"),
            Shapes: T.Tensor((n1,), "uint32"),
            Aabbs: T.Tensor((n2,), "uint32"),
            BatchIds: T.Tensor((n3,), "uint32"),
            Vertices: T.Tensor((n4,), "uint32"),
            Pairs: T.Tensor((n5,), "uint32"),
            PairCount: T.Tensor((n6,), "uint32"),
            Groups: T.Tensor((n7,), "uint32"),
            Filters: T.Tensor((n8,), "uint32"),
            RbdParams: T.Tensor((n9,), "uint32"),
        ):
            with T.Kernel(groups, threads=64) as group:
                one(
                    Poses,
                    Shapes,
                    Aabbs,
                    BatchIds,
                    Vertices,
                    Pairs,
                    PairCount,
                    Groups,
                    Filters,
                    RbdParams,
                    group * 64 + T.get_thread_binding(),
                )

    return kernel


def make_program(layout, case, target):
    kinds = (
        ["world_aabbs", "brute_pairs"]
        if case["kind"] == "broadphase_pairs"
        else [case["kind"]]
    )
    plans = []
    for kind in kinds:
        groups = (
            max((case["count"] + 63) // 64, 1)
            if kind == "world_aabbs"
            else case["groups"]
        )
        plans.append(
            dict(
                name=kind,
                kernel=make_kernel(
                    layout, dict(case, kind=kind, groups=groups), target
                ),
                buffer_map={n: n for n in BUFFERS},
                blocks=groups,
                threads=64,
            )
        )
    return plans
