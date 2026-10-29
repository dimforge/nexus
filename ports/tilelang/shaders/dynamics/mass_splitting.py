"""Counterpart of dynamics/mass_splitting.rs, preserving its 64-lane reduction."""

import tilelang.language as T
from kernel_support import f32


def gpu_hub_assign(layout):
    h, b = layout["hub"], layout["batch"]
    mw, im = layout["world_mass_words"], layout["world_mass"]["inv_mass"]
    threshold = layout["hub_min_constraints"]

    @T.macro
    def assign(
        Counts,
        WorldMass,
        BodyGroup,
        FirstSlots,
        HubList,
        HubCounts,
        BatchIds,
        tid,
        groups,
    ):
        count = T.cast(
            BatchIds[b["num_batches"]] * BatchIds[b["colliders_batch_capacity"]],
            "int32",
        )
        pool = T.cast(HubCounts[h["pool"]], "int32")
        for chunk in T.serial(T.ceildiv(count, groups * 64)):
            body = tid + chunk * groups * 64
            if body < count:
                start = T.alloc_local((1,), "int32")
                start[0] = 0
                if body > 0:
                    start[0] = T.cast(Counts[body - 1], "int32")
                n = T.cast(Counts[body], "int32") - start[0]
                m = body * mw + im
                if (
                    (n > threshold)
                    & (BodyGroup[body] == T.cast(body, "uint32"))
                    & (
                        (f32(WorldMass, m) != 0.0)
                        | (f32(WorldMass, m + 1) != 0.0)
                        | (f32(WorldMass, m + 2) != 0.0)
                    )
                ):
                    s0 = T.cast(
                        T.atomic_add(
                            HubCounts[h["slots"]], T.cast(n, "uint32"), return_prev=True
                        ),
                        "int32",
                    )
                    if s0 + n <= pool:
                        k = T.cast(
                            T.atomic_add(
                                HubCounts[h["hubs"]], T.uint32(1), return_prev=True
                            ),
                            "int32",
                        )
                        if k < pool // threshold + 1:
                            FirstSlots[body] = T.cast(s0, "uint32")
                            HubList[T.cast(k, "int32")] = T.cast(body, "uint32")

    return assign


def gpu_hub_split_constraints(layout):
    h, l, lw = layout["hub"], layout["link"], layout["link_words"]
    threshold = layout["hub_min_constraints"]

    @T.macro
    def split(
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
    ):
        hubs = T.cast(
            T.min(HubCounts[h["hubs"]], HubCounts[h["pool"]] // threshold + 1), "int32"
        )
        for round in T.serial(T.ceildiv(hubs, groups)):
            hub = group + round * groups
            if hub < hubs:
                body = T.cast(HubList[hub], "int32")
                s0 = T.cast(FirstSlots[body], "int32")
                start = T.alloc_local((1,), "int32")
                start[0] = 0
                if body > 0:
                    start[0] = T.cast(Counts[body - 1], "int32")
                end = T.cast(Counts[body], "int32")
                for chunk in T.serial(T.ceildiv(end - start[0], 64)):
                    entry = start[0] + lane + chunk * 64
                    if entry < end:
                        slot = s0 + entry - start[0]
                        cid = T.cast(Ids[entry], "int32")
                        base = cid * lw
                        value = HubCounts[h["base"]] + T.cast(slot, "uint32")
                        scale = T.reinterpret(
                            T.cast(end - start[0], "float32"), "uint32"
                        )
                        if Links[base + l["solver_body_a"]] == T.cast(body, "uint32"):
                            Links[base + l["vel_slot_a"]] = value
                            Links[base + l["mass_scale_a"]] = scale
                        else:
                            Links[base + l["vel_slot_b"]] = value
                            Links[base + l["mass_scale_b"]] = scale
                        SlotBody[slot] = T.cast(body, "uint32")
                        SlotConstraint[slot] = T.cast(cid, "uint32")

    return split


def gpu_hub_dispatch(layout):
    h, threshold = layout["hub"], layout["hub_min_constraints"]

    @T.macro
    def dispatch(HubCounts, SlotsIndirect, AverageIndirect):
        slots = T.min(HubCounts[h["slots"]], HubCounts[h["pool"]])
        hubs = T.min(HubCounts[h["hubs"]], HubCounts[h["pool"]] // threshold + 1)
        SlotsIndirect[0] = (slots + 63) // 64
        SlotsIndirect[1] = T.uint32(1)
        SlotsIndirect[2] = T.uint32(1)
        AverageIndirect[0] = T.min(hubs, T.uint32(layout["hub_average_workgroups"]))
        AverageIndirect[1] = T.uint32(1)
        AverageIndirect[2] = T.uint32(1)

    return dispatch


def gpu_hub_scatter(layout):
    h, vw = layout["hub"], layout["velocity_words"]

    @T.macro
    def scatter(Velocities, SlotBody, HubCounts, slot):
        if slot < T.cast(T.min(HubCounts[h["slots"]], HubCounts[h["pool"]]), "int32"):
            body = T.cast(SlotBody[slot], "int32")
            dest = T.cast(HubCounts[h["base"]], "int32") + slot
            for word in T.unroll(vw):
                Velocities[dest * vw + word] = Velocities[body * vw + word]

    return scatter


def gpu_hub_average(layout, serial):
    h, vw = layout["hub"], layout["velocity_words"]
    vl, va = layout["velocity_linear"], layout["velocity_angular"]
    threshold = layout["hub_min_constraints"]

    @T.macro
    def load_lane(Velocities, Sums, start, n, lane):
        local = T.alloc_local((6,), "float32")
        for axis in T.unroll(6):
            local[axis] = 0.0
        for chunk in T.serial(T.ceildiv(n, 64)):
            slot = lane + chunk * 64
            if slot < n:
                v = (start + slot) * vw
                for axis in T.unroll(3):
                    local[axis] = local[axis] + f32(Velocities, v + vl + axis)
                    local[axis + 3] = local[axis + 3] + f32(Velocities, v + va + axis)
        for axis in T.unroll(6):
            Sums[lane, axis] = local[axis]

    @T.macro
    def reduce_lane(Sums, lane, stride):
        for axis in T.unroll(6):
            Sums[lane, axis] = Sums[lane, axis] + Sums[lane + stride, axis]

    @T.macro
    def store(Velocities, Sums, body, n):
        if n > 0:
            inv_n = T.float32(1.0) / T.cast(n, "float32")
            for axis in T.unroll(3):
                Velocities[body * vw + vl + axis] = T.reinterpret(
                    Sums[0, axis] * inv_n, "uint32"
                )
                Velocities[body * vw + va + axis] = T.reinterpret(
                    Sums[0, axis + 3] * inv_n, "uint32"
                )

    @T.macro
    def average(
        Velocities, HubList, FirstSlots, Counts, HubCounts, Sums, group, lane, groups
    ):
        hubs = T.cast(
            T.min(HubCounts[h["hubs"]], HubCounts[h["pool"]] // threshold + 1), "int32"
        )
        # Same active workgroup rounds as Rust's non-web path. This branch is
        # uniform within each workgroup; every participating lane reaches barriers.
        rounds = T.alloc_local((1,), "int32")
        rounds[0] = 0
        if hubs > group:
            rounds[0] = T.ceildiv(hubs - group, groups)
        for round in T.serial(rounds[0]):
            hub = group + round * groups
            body = T.cast(HubList[hub], "int32")
            start = T.cast(HubCounts[h["base"]] + FirstSlots[body], "int32")
            first = T.alloc_local((1,), "int32")
            first[0] = 0
            if body > 0:
                first[0] = T.cast(Counts[body - 1], "int32")
            n = T.cast(Counts[body], "int32") - first[0]
            if serial:
                for virtual_lane in T.serial(64):
                    load_lane(Velocities, Sums, start, n, virtual_lane)
                for step in T.unroll(6):
                    stride = 32 >> step
                    for virtual_lane in T.serial(stride):
                        reduce_lane(Sums, virtual_lane, stride)
                store(Velocities, Sums, body, n)
            else:
                load_lane(Velocities, Sums, start, n, lane)
                for step in T.unroll(6):
                    T.sync_threads()
                    stride = 32 >> step
                    if lane < stride:
                        reduce_lane(Sums, lane, stride)
                T.sync_threads()
                if lane == 0:
                    store(Velocities, Sums, body, n)
                T.sync_threads()

    return average
