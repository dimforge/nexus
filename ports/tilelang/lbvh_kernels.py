"""Target launch adapters for mirrored LBVH stages."""

# ruff: noqa: SIM117
import tilelang.language as T
from shaders.broad_phase import lbvh as B

BUFFERS = (
    "Pairs",
    "Groups",
    "Filters",
    "Poses",
    "Shapes",
    "Vertices",
    "Partials",
    "Domain",
    "Keys",
    "Sorted",
    "Tree",
    "Frontier",
    "FrontierLen",
    "Rounds",
    "PairCount",
    "Indirect",
    "BatchIds",
)


def dimensions(c):
    kind = c["kind"]
    threads = (
        256
        if kind in ("lbvh_refit_chunks", "lbvh_refit_frontier")
        else 128
        if kind == "lbvh_domain"
        else 1
        if kind in ("lbvh_refit_plan", "flat_list_dispatch")
        else 64
    )
    per = (
        max((c["colliders"] + 255) // 256, 1)
        if kind == "lbvh_refit_chunks"
        else 1
        if kind
        in (
            "lbvh_domain_merge",
            "lbvh_refit_frontier",
            "lbvh_refit_plan",
            "flat_list_dispatch",
            "lbvh_reset",
        )
        else c["groups"]
    )
    batches = (
        1
        if kind in ("lbvh_refit_plan", "flat_list_dispatch", "lbvh_reset")
        else c["batches"]
    )
    return per * batches, threads, per


def make_kernel(l, c, target):
    sizes = c["buffer_words"]
    kind = c["kind"]
    cpu = target == "c"
    blocks, threads, per = dimensions(c)
    domain = B.gpu_lbvh_compute_domain(l, cpu)
    merge = B.gpu_lbvh_domain_merge(l, cpu)
    morton = B.gpu_lbvh_compute_morton(l, c["groups"])
    build = B.gpu_lbvh_build(l, c["groups"])
    escapes = B.gpu_lbvh_escapes(l, c["groups"])
    leaves = B.gpu_lbvh_refit_leaves(l, c["groups"])
    pairs = B.gpu_lbvh_find_collision_pairs(l, c, target)
    chunks = B.gpu_lbvh_refit_chunks(l, target)
    frontier = B.gpu_lbvh_refit_frontier(l, target)
    plan = B.gpu_lbvh_refit_plan(l)
    reset = B.gpu_lbvh_reset_collision_pairs(sizes["PairCount"])
    flat = B.gpu_flat_list_dispatch(l)

    @T.macro
    def one(
        Pairs,
        Groups,
        Filters,
        Poses,
        Shapes,
        Vertices,
        Partials,
        Domain,
        Keys,
        Sorted,
        Tree,
        Frontier,
        FrontierLen,
        Rounds,
        PairCount,
        Indirect,
        BatchIds,
        Mins,
        Maxs,
        WgPairs,
        Scan,
        Temp,
        Base,
        block,
        lane,
    ):
        batch = block // per
        group = block % per
        gid = group * threads + lane
        if kind == "lbvh_domain":
            domain(Poses, Partials, BatchIds, Mins, Maxs, group, batch, per, lane)
        if kind == "lbvh_domain_merge":
            merge(Partials, Domain, Mins, Maxs, batch, lane)
        if kind == "lbvh_morton":
            morton(Poses, Domain, Keys, BatchIds, batch, gid)
        if kind == "lbvh_build":
            build(Keys, Tree, BatchIds, batch, gid)
        if kind == "lbvh_escapes":
            escapes(Tree, BatchIds, batch, gid)
        if kind == "lbvh_leaves" and c["colliders"] > 0:
            leaves(
                Poses,
                Shapes,
                Sorted,
                Tree,
                BatchIds,
                FrontierLen,
                Vertices,
                batch,
                gid,
            )
        if kind == "lbvh_pairs" and c["colliders"] > 0:
            pairs(
                Tree,
                Pairs,
                PairCount,
                Groups,
                BatchIds,
                Filters,
                WgPairs,
                Scan,
                Temp,
                Base,
                group,
                batch,
                lane,
            )
        if kind == "lbvh_refit_chunks" and c["colliders"] >= 2:
            chunks(Tree, Frontier, FrontierLen, BatchIds, Base, group, batch, lane)
        if kind == "lbvh_refit_frontier" and c["colliders"] >= 2:
            frontier(Tree, Frontier, FrontierLen, BatchIds, Rounds, Base, batch, lane)
        if kind == "lbvh_refit_plan":
            plan(FrontierLen, Rounds, BatchIds)
        if kind == "lbvh_reset":
            reset(PairCount, gid)
        if kind == "flat_list_dispatch":
            flat(PairCount, Indirect, BatchIds)

    if target == "c":

        @T.prim_func
        def kernel(
            Pairs: T.Tensor((sizes["Pairs"],), "uint32"),
            Groups: T.Tensor((sizes["Groups"],), "uint32"),
            Filters: T.Tensor((sizes["Filters"],), "uint32"),
            Poses: T.Tensor((sizes["Poses"],), "uint32"),
            Shapes: T.Tensor((sizes["Shapes"],), "uint32"),
            Vertices: T.Tensor((sizes["Vertices"],), "uint32"),
            Partials: T.Tensor((sizes["Partials"],), "uint32"),
            Domain: T.Tensor((sizes["Domain"],), "uint32"),
            Keys: T.Tensor((sizes["Keys"],), "uint32"),
            Sorted: T.Tensor((sizes["Sorted"],), "uint32"),
            Tree: T.Tensor((sizes["Tree"],), "uint32"),
            Frontier: T.Tensor((sizes["Frontier"],), "uint32"),
            FrontierLen: T.Tensor((sizes["FrontierLen"],), "uint32"),
            Rounds: T.Tensor((sizes["Rounds"],), "uint32"),
            PairCount: T.Tensor((sizes["PairCount"],), "uint32"),
            Indirect: T.Tensor((sizes["Indirect"],), "uint32"),
            BatchIds: T.Tensor((sizes["BatchIds"],), "uint32"),
        ):
            with T.Kernel(1):
                Mins = T.alloc_local((128, 3), "float32")
                Maxs = T.alloc_local((128, 3), "float32")
                WgPairs = T.alloc_local((512, 2), "uint32")
                Scan = T.alloc_local((64,), "uint32")
                Temp = T.alloc_local((64,), "uint32")
                Base = T.alloc_local((1,), "uint32")
                for block in T.serial(blocks):
                    for lane in T.serial(
                        1
                        if kind
                        in (
                            "lbvh_domain",
                            "lbvh_domain_merge",
                            "lbvh_pairs",
                            "lbvh_refit_chunks",
                            "lbvh_refit_frontier",
                            "lbvh_refit_plan",
                            "flat_list_dispatch",
                        )
                        else threads
                    ):
                        one(
                            Pairs,
                            Groups,
                            Filters,
                            Poses,
                            Shapes,
                            Vertices,
                            Partials,
                            Domain,
                            Keys,
                            Sorted,
                            Tree,
                            Frontier,
                            FrontierLen,
                            Rounds,
                            PairCount,
                            Indirect,
                            BatchIds,
                            Mins,
                            Maxs,
                            WgPairs,
                            Scan,
                            Temp,
                            Base,
                            block,
                            lane,
                        )
    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            Pairs: A.Tensor((sizes["Pairs"],), "uint32"),
            Groups: A.Tensor((sizes["Groups"],), "uint32"),
            Filters: A.Tensor((sizes["Filters"],), "uint32"),
            Poses: A.Tensor((sizes["Poses"],), "uint32"),
            Shapes: A.Tensor((sizes["Shapes"],), "uint32"),
            Vertices: A.Tensor((sizes["Vertices"],), "uint32"),
            Partials: A.Tensor((sizes["Partials"],), "uint32"),
            Domain: A.Tensor((sizes["Domain"],), "uint32"),
            Keys: A.Tensor((sizes["Keys"],), "uint32"),
            Sorted: A.Tensor((sizes["Sorted"],), "uint32"),
            Tree: A.Tensor((sizes["Tree"],), "uint32"),
            Frontier: A.Tensor((sizes["Frontier"],), "uint32"),
            FrontierLen: A.Tensor((sizes["FrontierLen"],), "uint32"),
            Rounds: A.Tensor((sizes["Rounds"],), "uint32"),
            PairCount: A.Tensor((sizes["PairCount"],), "uint32"),
            Indirect: A.Tensor((sizes["Indirect"],), "uint32"),
            BatchIds: A.Tensor((sizes["BatchIds"],), "uint32"),
        ):
            with A.Kernel(blocks) as block:
                with A.SimtVF(threads=threads):
                    Mins = T.alloc_shared((128, 3), "float32")
                    Maxs = T.alloc_shared((128, 3), "float32")
                    WgPairs = T.alloc_local((8, 2), "uint32")
                    Scan = T.alloc_shared((64,), "uint32")
                    Temp = T.alloc_shared((64,), "uint32")
                    Base = T.alloc_shared((1,), "uint32")
                    one(
                        Pairs,
                        Groups,
                        Filters,
                        Poses,
                        Shapes,
                        Vertices,
                        Partials,
                        Domain,
                        Keys,
                        Sorted,
                        Tree,
                        Frontier,
                        FrontierLen,
                        Rounds,
                        PairCount,
                        Indirect,
                        BatchIds,
                        Mins,
                        Maxs,
                        WgPairs,
                        Scan,
                        Temp,
                        Base,
                        block,
                        A.get_thread_binding(),
                    )
    else:

        @T.prim_func
        def kernel(
            Pairs: T.Tensor((sizes["Pairs"],), "uint32"),
            Groups: T.Tensor((sizes["Groups"],), "uint32"),
            Filters: T.Tensor((sizes["Filters"],), "uint32"),
            Poses: T.Tensor((sizes["Poses"],), "uint32"),
            Shapes: T.Tensor((sizes["Shapes"],), "uint32"),
            Vertices: T.Tensor((sizes["Vertices"],), "uint32"),
            Partials: T.Tensor((sizes["Partials"],), "uint32"),
            Domain: T.Tensor((sizes["Domain"],), "uint32"),
            Keys: T.Tensor((sizes["Keys"],), "uint32"),
            Sorted: T.Tensor((sizes["Sorted"],), "uint32"),
            Tree: T.Tensor((sizes["Tree"],), "uint32"),
            Frontier: T.Tensor((sizes["Frontier"],), "uint32"),
            FrontierLen: T.Tensor((sizes["FrontierLen"],), "uint32"),
            Rounds: T.Tensor((sizes["Rounds"],), "uint32"),
            PairCount: T.Tensor((sizes["PairCount"],), "uint32"),
            Indirect: T.Tensor((sizes["Indirect"],), "uint32"),
            BatchIds: T.Tensor((sizes["BatchIds"],), "uint32"),
        ):
            with T.Kernel(blocks, threads=threads) as block:
                T.import_source(
                    "inline int nexus_lbvh_refit_barrier() { threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup); return 0; }\n"
                    if target == "metal"
                    else None
                )
                Mins = T.alloc_shared((128, 3), "float32")
                Maxs = T.alloc_shared((128, 3), "float32")
                WgPairs = T.alloc_shared((512, 2), "uint32")
                Scan = T.alloc_shared((64,), "uint32")
                Temp = T.alloc_shared((64,), "uint32")
                Base = T.alloc_shared((1,), "uint32")
                one(
                    Pairs,
                    Groups,
                    Filters,
                    Poses,
                    Shapes,
                    Vertices,
                    Partials,
                    Domain,
                    Keys,
                    Sorted,
                    Tree,
                    Frontier,
                    FrontierLen,
                    Rounds,
                    PairCount,
                    Indirect,
                    BatchIds,
                    Mins,
                    Maxs,
                    WgPairs,
                    Scan,
                    Temp,
                    Base,
                    block,
                    T.get_thread_binding(),
                )

    return kernel
