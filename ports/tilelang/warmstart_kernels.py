"""Target launch adapters for the mirrored warmstart entry points."""

import tilelang.language as T
from shaders.dynamics.tile_kernels import (
    gpu_scale_constraint_impulses,
    gpu_gather_warmstart_velocities,
    gpu_hub_warmstart_constraints,
)


def make_kernel(layout, case, target):
    scale = gpu_scale_constraint_impulses(layout)
    gather = gpu_gather_warmstart_velocities(layout)
    hub = gpu_hub_warmstart_constraints(layout)
    kind = case["kind"]
    n = case["dispatch"]

    @T.macro
    def one(
        Tiles,
        Velocities,
        Counts,
        Ids,
        Indices,
        HubFirstSlot,
        HubSlotConstraint,
        HubCounts,
        Params,
        Buckets,
        BatchIds,
        i,
    ):
        if kind == "scale_impulses":
            scale(Tiles, Buckets, BatchIds, Params, i)
        elif kind == "gather_warmstart":
            gather(Counts, Ids, Indices, Tiles, Velocities, HubFirstSlot, BatchIds, i)
        else:
            hub(Velocities, Tiles, HubSlotConstraint, HubCounts, Indices, i)

    if case["count"] == 0:

        @T.macro
        def empty(
            Tiles,
            Velocities,
            Counts,
            Ids,
            Indices,
            HubFirstSlot,
            HubSlotConstraint,
            HubCounts,
            Params,
            Buckets,
            BatchIds,
            i,
        ):
            T.evaluate(0)

        one = empty
    n0 = case["tiles"] * layout["tile_words"]
    n1 = case["velocities"] * layout["velocity_words"]
    n2 = case["storage"]
    n3 = case["ids_storage"]
    n4 = case["constraints"]
    n5 = case["storage"]
    n6 = case["hub_storage"]
    n7 = layout["hub_words"]
    n8 = layout["params_words"]
    n9 = case["bucket_storage"]
    n10 = layout["batch_words"]
    if target == "c":

        @T.prim_func
        def kernel(
            Tiles: T.Tensor((n0,), "uint32"),
            Velocities: T.Tensor((n1,), "uint32"),
            Counts: T.Tensor((n2,), "uint32"),
            Ids: T.Tensor((n3,), "uint32"),
            Indices: T.Tensor((n4,), "uint32"),
            HubFirstSlot: T.Tensor((n5,), "uint32"),
            HubSlotConstraint: T.Tensor((n6,), "uint32"),
            HubCounts: T.Tensor((n7,), "uint32"),
            Params: T.Tensor((n8,), "uint32"),
            Buckets: T.Tensor((n9,), "uint32"),
            BatchIds: T.Tensor((n10,), "uint32"),
        ):
            with T.Kernel(1):
                for i in T.serial(n):
                    one(
                        Tiles,
                        Velocities,
                        Counts,
                        Ids,
                        Indices,
                        HubFirstSlot,
                        HubSlotConstraint,
                        HubCounts,
                        Params,
                        Buckets,
                        BatchIds,
                        i,
                    )
    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            Tiles: A.Tensor((n0,), "uint32"),
            Velocities: A.Tensor((n1,), "uint32"),
            Counts: A.Tensor((n2,), "uint32"),
            Ids: A.Tensor((n3,), "uint32"),
            Indices: A.Tensor((n4,), "uint32"),
            HubFirstSlot: A.Tensor((n5,), "uint32"),
            HubSlotConstraint: A.Tensor((n6,), "uint32"),
            HubCounts: A.Tensor((n7,), "uint32"),
            Params: A.Tensor((n8,), "uint32"),
            Buckets: A.Tensor((n9,), "uint32"),
            BatchIds: A.Tensor((n10,), "uint32"),
        ):
            with A.Kernel(max((n + 63) // 64, 1)) as block:
                with A.SimtVF(threads=64):
                    one(
                        Tiles,
                        Velocities,
                        Counts,
                        Ids,
                        Indices,
                        HubFirstSlot,
                        HubSlotConstraint,
                        HubCounts,
                        Params,
                        Buckets,
                        BatchIds,
                        block * 64 + A.get_thread_binding(),
                    )
    elif target in ("metal", "cuda"):

        @T.prim_func
        def kernel(
            Tiles: T.Tensor((n0,), "uint32"),
            Velocities: T.Tensor((n1,), "uint32"),
            Counts: T.Tensor((n2,), "uint32"),
            Ids: T.Tensor((n3,), "uint32"),
            Indices: T.Tensor((n4,), "uint32"),
            HubFirstSlot: T.Tensor((n5,), "uint32"),
            HubSlotConstraint: T.Tensor((n6,), "uint32"),
            HubCounts: T.Tensor((n7,), "uint32"),
            Params: T.Tensor((n8,), "uint32"),
            Buckets: T.Tensor((n9,), "uint32"),
            BatchIds: T.Tensor((n10,), "uint32"),
        ):
            with T.Kernel(max((n + 63) // 64, 1), threads=64) as block:
                one(
                    Tiles,
                    Velocities,
                    Counts,
                    Ids,
                    Indices,
                    HubFirstSlot,
                    HubSlotConstraint,
                    HubCounts,
                    Params,
                    Buckets,
                    BatchIds,
                    block * 64 + T.get_thread_binding(),
                )
    else:
        raise ValueError(f"Unsupported target {target}")
    return kernel
