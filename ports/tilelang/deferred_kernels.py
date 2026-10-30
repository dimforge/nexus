# ruff: noqa: SIM117
"""Native deferred work-list, contact, sort-key and reduction launch adapters."""

import tilelang.language as T
from shaders.broad_phase.narrow_phase import (
    gpu_narrow_phase_pfm_pfm,
    gpu_narrow_phase_shape_shape_deferred,
    gpu_pfm_pair_sort_keys_permuted,
    gpu_pfm_sort_keys,
    gpu_pfm_sub_shape_sort_keys,
    gpu_reduce_contacts,
)

BUFFERS = (
    "Pairs",
    "PairCount",
    "Poses",
    "Shapes",
    "PfmPairs",
    "PfmCount",
    "Vertices",
    "Indices",
    "BatchIds",
    "RbdParams",
    "Contacts",
    "Order",
    "Plan",
    "Parents",
    "Materials",
    "Keys",
)


def make_kernel(l, c, target, stage):
    sizes = c["buffer_words"]
    groups = c["groups"]
    deferred = gpu_narrow_phase_shape_shape_deferred(l, c)
    manifold = gpu_narrow_phase_pfm_pfm(l, c)
    reduction = gpu_reduce_contacts(l, c)
    pair_keys = gpu_pfm_sort_keys(l, c)
    sub_keys = gpu_pfm_sub_shape_sort_keys(l, c)
    perm_keys = gpu_pfm_pair_sort_keys_permuted(l, c)

    @T.macro
    def one(
        Pairs,
        PairCount,
        Poses,
        Shapes,
        PfmPairs,
        PfmCount,
        Vertices,
        Indices,
        BatchIds,
        RbdParams,
        Contacts,
        Order,
        Plan,
        Parents,
        Materials,
        Keys,
        gid,
    ):
        if stage == "deferred_pairs":
            deferred(
                Pairs,
                PairCount,
                Poses,
                Shapes,
                PfmPairs,
                PfmCount,
                Vertices,
                Indices,
                BatchIds,
                RbdParams,
                gid,
            )
        elif stage == "deferred_manifolds":
            manifold(
                Contacts,
                PfmPairs,
                Order,
                Plan,
                Vertices,
                Indices,
                Parents,
                Materials,
                RbdParams,
                gid,
            )
        elif stage == "reduce_contacts":
            reduction(Contacts, Keys, Plan, RbdParams, gid)
        elif stage == "pfm_pair_keys":
            pair_keys(PfmPairs, Plan, Order, Keys, gid)
        elif stage == "pfm_subshape_keys":
            sub_keys(PfmPairs, Plan, Order, Keys, gid)
        else:
            perm_keys(PfmPairs, Plan, Order, Keys, gid)

    if target == "c":

        @T.prim_func
        def kernel(
            Pairs: T.Tensor((sizes["Pairs"],), "uint32"),
            PairCount: T.Tensor((sizes["PairCount"],), "uint32"),
            Poses: T.Tensor((sizes["Poses"],), "uint32"),
            Shapes: T.Tensor((sizes["Shapes"],), "uint32"),
            PfmPairs: T.Tensor((sizes["PfmPairs"],), "uint32"),
            PfmCount: T.Tensor((sizes["PfmCount"],), "uint32"),
            Vertices: T.Tensor((sizes["Vertices"],), "uint32"),
            Indices: T.Tensor((sizes["Indices"],), "uint32"),
            BatchIds: T.Tensor((sizes["BatchIds"],), "uint32"),
            RbdParams: T.Tensor((sizes["RbdParams"],), "uint32"),
            Contacts: T.Tensor((sizes["Contacts"],), "uint32"),
            Order: T.Tensor((sizes["Order"],), "uint32"),
            Plan: T.Tensor((sizes["Plan"],), "uint32"),
            Parents: T.Tensor((sizes["Parents"],), "uint32"),
            Materials: T.Tensor((sizes["Materials"],), "uint32"),
            Keys: T.Tensor((sizes["Keys"],), "uint32"),
        ):
            with T.Kernel(1):
                for gid in T.serial(groups * 64):
                    one(
                        Pairs,
                        PairCount,
                        Poses,
                        Shapes,
                        PfmPairs,
                        PfmCount,
                        Vertices,
                        Indices,
                        BatchIds,
                        RbdParams,
                        Contacts,
                        Order,
                        Plan,
                        Parents,
                        Materials,
                        Keys,
                        gid,
                    )
    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            Pairs: A.Tensor((sizes["Pairs"],), "uint32"),
            PairCount: A.Tensor((sizes["PairCount"],), "uint32"),
            Poses: A.Tensor((sizes["Poses"],), "uint32"),
            Shapes: A.Tensor((sizes["Shapes"],), "uint32"),
            PfmPairs: A.Tensor((sizes["PfmPairs"],), "uint32"),
            PfmCount: A.Tensor((sizes["PfmCount"],), "uint32"),
            Vertices: A.Tensor((sizes["Vertices"],), "uint32"),
            Indices: A.Tensor((sizes["Indices"],), "uint32"),
            BatchIds: A.Tensor((sizes["BatchIds"],), "uint32"),
            RbdParams: A.Tensor((sizes["RbdParams"],), "uint32"),
            Contacts: A.Tensor((sizes["Contacts"],), "uint32"),
            Order: A.Tensor((sizes["Order"],), "uint32"),
            Plan: A.Tensor((sizes["Plan"],), "uint32"),
            Parents: A.Tensor((sizes["Parents"],), "uint32"),
            Materials: A.Tensor((sizes["Materials"],), "uint32"),
            Keys: A.Tensor((sizes["Keys"],), "uint32"),
        ):
            with A.Kernel(groups) as group:
                with A.SimtVF(threads=64):
                    one(
                        Pairs,
                        PairCount,
                        Poses,
                        Shapes,
                        PfmPairs,
                        PfmCount,
                        Vertices,
                        Indices,
                        BatchIds,
                        RbdParams,
                        Contacts,
                        Order,
                        Plan,
                        Parents,
                        Materials,
                        Keys,
                        group * 64 + A.get_thread_binding(),
                    )
    else:

        @T.prim_func
        def kernel(
            Pairs: T.Tensor((sizes["Pairs"],), "uint32"),
            PairCount: T.Tensor((sizes["PairCount"],), "uint32"),
            Poses: T.Tensor((sizes["Poses"],), "uint32"),
            Shapes: T.Tensor((sizes["Shapes"],), "uint32"),
            PfmPairs: T.Tensor((sizes["PfmPairs"],), "uint32"),
            PfmCount: T.Tensor((sizes["PfmCount"],), "uint32"),
            Vertices: T.Tensor((sizes["Vertices"],), "uint32"),
            Indices: T.Tensor((sizes["Indices"],), "uint32"),
            BatchIds: T.Tensor((sizes["BatchIds"],), "uint32"),
            RbdParams: T.Tensor((sizes["RbdParams"],), "uint32"),
            Contacts: T.Tensor((sizes["Contacts"],), "uint32"),
            Order: T.Tensor((sizes["Order"],), "uint32"),
            Plan: T.Tensor((sizes["Plan"],), "uint32"),
            Parents: T.Tensor((sizes["Parents"],), "uint32"),
            Materials: T.Tensor((sizes["Materials"],), "uint32"),
            Keys: T.Tensor((sizes["Keys"],), "uint32"),
        ):
            with T.Kernel(groups, threads=64) as group:
                one(
                    Pairs,
                    PairCount,
                    Poses,
                    Shapes,
                    PfmPairs,
                    PfmCount,
                    Vertices,
                    Indices,
                    BatchIds,
                    RbdParams,
                    Contacts,
                    Order,
                    Plan,
                    Parents,
                    Materials,
                    Keys,
                    group * 64 + T.get_thread_binding(),
                )

    return kernel
