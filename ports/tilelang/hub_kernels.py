"""Launch adapters for the Rust-mirrored mass splitting entry points."""

import tilelang.language as T
from shaders.dynamics.mass_splitting import (
    gpu_hub_assign,
    gpu_hub_split_constraints,
    gpu_hub_dispatch,
    gpu_hub_scatter,
    gpu_hub_average,
)


def make_kernel(layout, case, target):
    assign = gpu_hub_assign(layout)
    split = gpu_hub_split_constraints(layout)
    dispatch = gpu_hub_dispatch(layout)
    scatter = gpu_hub_scatter(layout)
    average = gpu_hub_average(layout, target == "c")
    kind, groups, threads = case["kind"], case["groups"], case["threads"]
    n0 = case["storage"]
    n1 = case["ids_storage"]
    n2 = case["storage"] * layout["world_mass_words"]
    n3 = case["storage"]
    n4 = case["storage"]
    n5 = case["hub_storage"]
    n6 = layout["hub_words"]
    n7 = case["constraints"] * layout["link_words"]
    n8 = case["slot_storage"]
    n9 = case["slot_storage"]
    n10 = case["velocities"] * layout["velocity_words"]
    n11 = 9
    n12 = 9
    n13 = layout["batch_words"]

    @T.macro
    def one(
        Counts,
        Ids,
        WorldMass,
        BodyGroup,
        FirstSlots,
        HubList,
        HubCounts,
        Links,
        SlotBody,
        SlotConstraint,
        Velocities,
        SlotsIndirect,
        AverageIndirect,
        BatchIds,
        Sums,
        group,
        lane,
    ):
        if case["count"] == 0 and kind != "hub_dispatch":
            T.evaluate(0)
        elif kind == "hub_assign":
            assign(
                Counts,
                WorldMass,
                BodyGroup,
                FirstSlots,
                HubList,
                HubCounts,
                BatchIds,
                group * 64 + lane,
                groups,
            )
        elif kind == "hub_split":
            split(
                Counts,
                Ids,
                FirstSlots,
                Links,
                SlotBody,
                SlotConstraint,
                HubCounts,
                HubList,
                group,
                lane,
                groups,
            )
        elif kind == "hub_dispatch":
            dispatch(HubCounts, SlotsIndirect, AverageIndirect)
        elif kind == "hub_scatter":
            scatter(Velocities, SlotBody, HubCounts, group * 64 + lane)
        else:
            average(
                Velocities,
                HubList,
                FirstSlots,
                Counts,
                HubCounts,
                Sums,
                group,
                lane,
                groups,
            )

    if target == "c":

        @T.prim_func
        def kernel(
            Counts: T.Tensor((n0,), "uint32"),
            Ids: T.Tensor((n1,), "uint32"),
            WorldMass: T.Tensor((n2,), "uint32"),
            BodyGroup: T.Tensor((n3,), "uint32"),
            FirstSlots: T.Tensor((n4,), "uint32"),
            HubList: T.Tensor((n5,), "uint32"),
            HubCounts: T.Tensor((n6,), "uint32"),
            Links: T.Tensor((n7,), "uint32"),
            SlotBody: T.Tensor((n8,), "uint32"),
            SlotConstraint: T.Tensor((n9,), "uint32"),
            Velocities: T.Tensor((n10,), "uint32"),
            SlotsIndirect: T.Tensor((n11,), "uint32"),
            AverageIndirect: T.Tensor((n12,), "uint32"),
            BatchIds: T.Tensor((n13,), "uint32"),
        ):
            with T.Kernel(1):
                Sums = T.alloc_local((64, 6), "float32")
                for group in T.serial(groups):
                    if kind == "hub_average":
                        one(
                            Counts,
                            Ids,
                            WorldMass,
                            BodyGroup,
                            FirstSlots,
                            HubList,
                            HubCounts,
                            Links,
                            SlotBody,
                            SlotConstraint,
                            Velocities,
                            SlotsIndirect,
                            AverageIndirect,
                            BatchIds,
                            Sums,
                            group,
                            0,
                        )
                    else:
                        for lane in T.serial(threads):
                            one(
                                Counts,
                                Ids,
                                WorldMass,
                                BodyGroup,
                                FirstSlots,
                                HubList,
                                HubCounts,
                                Links,
                                SlotBody,
                                SlotConstraint,
                                Velocities,
                                SlotsIndirect,
                                AverageIndirect,
                                BatchIds,
                                Sums,
                                group,
                                lane,
                            )

    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            Counts: A.Tensor((n0,), "uint32"),
            Ids: A.Tensor((n1,), "uint32"),
            WorldMass: A.Tensor((n2,), "uint32"),
            BodyGroup: A.Tensor((n3,), "uint32"),
            FirstSlots: A.Tensor((n4,), "uint32"),
            HubList: A.Tensor((n5,), "uint32"),
            HubCounts: A.Tensor((n6,), "uint32"),
            Links: A.Tensor((n7,), "uint32"),
            SlotBody: A.Tensor((n8,), "uint32"),
            SlotConstraint: A.Tensor((n9,), "uint32"),
            Velocities: A.Tensor((n10,), "uint32"),
            SlotsIndirect: A.Tensor((n11,), "uint32"),
            AverageIndirect: A.Tensor((n12,), "uint32"),
            BatchIds: A.Tensor((n13,), "uint32"),
        ):
            with A.Kernel(groups) as group:
                with A.SimtVF(threads=threads):
                    Sums = T.alloc_shared((64, 6), "float32")
                    one(
                        Counts,
                        Ids,
                        WorldMass,
                        BodyGroup,
                        FirstSlots,
                        HubList,
                        HubCounts,
                        Links,
                        SlotBody,
                        SlotConstraint,
                        Velocities,
                        SlotsIndirect,
                        AverageIndirect,
                        BatchIds,
                        Sums,
                        group,
                        A.get_thread_binding(),
                    )

    elif target in ("metal", "cuda"):

        @T.prim_func
        def kernel(
            Counts: T.Tensor((n0,), "uint32"),
            Ids: T.Tensor((n1,), "uint32"),
            WorldMass: T.Tensor((n2,), "uint32"),
            BodyGroup: T.Tensor((n3,), "uint32"),
            FirstSlots: T.Tensor((n4,), "uint32"),
            HubList: T.Tensor((n5,), "uint32"),
            HubCounts: T.Tensor((n6,), "uint32"),
            Links: T.Tensor((n7,), "uint32"),
            SlotBody: T.Tensor((n8,), "uint32"),
            SlotConstraint: T.Tensor((n9,), "uint32"),
            Velocities: T.Tensor((n10,), "uint32"),
            SlotsIndirect: T.Tensor((n11,), "uint32"),
            AverageIndirect: T.Tensor((n12,), "uint32"),
            BatchIds: T.Tensor((n13,), "uint32"),
        ):
            with T.Kernel(groups, threads=threads) as group:
                Sums = T.alloc_shared((64, 6), "float32")
                one(
                    Counts,
                    Ids,
                    WorldMass,
                    BodyGroup,
                    FirstSlots,
                    HubList,
                    HubCounts,
                    Links,
                    SlotBody,
                    SlotConstraint,
                    Velocities,
                    SlotsIndirect,
                    AverageIndirect,
                    BatchIds,
                    Sums,
                    group,
                    T.get_thread_binding(),
                )
    else:
        raise ValueError(f"Unsupported target {target}")
    return kernel
