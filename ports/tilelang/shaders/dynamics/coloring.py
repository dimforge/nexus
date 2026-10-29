"""Counterpart of dynamics/coloring.rs; graph-color choices and conflict pruning."""

import tilelang.language as T


def hash(packed_key):
    key = T.Cast("uint32", packed_key) * T.uint32(0xCC9E2D51)
    return ((key << 15) | (key >> 17)) * T.uint32(0x1B873593)


def first_free_color():
    @T.macro
    def pick(Mask, Result):
        Result[0] = T.uint32(64)
        for c in T.serial(64):
            if ((Mask[c // 32] >> (c % 32)) & T.uint32(1)) == 0:
                Result[0] = T.min(Result[0], T.Cast("uint32", c))

    return pick


def nth_free_color():
    @T.macro
    def pick(Mask, rank, Result):
        free_count = T.alloc_local((1,), "uint32")
        free_count[0] = T.uint32(0)
        Result[0] = T.uint32(63)
        for c in T.serial(64):
            if ((Mask[c // 32] >> (c % 32)) & T.uint32(1)) == 0:
                if free_count[0] == rank:
                    Result[0] = T.min(Result[0], T.Cast("uint32", c))
                free_count[0] += T.uint32(1)

    return pick


def _constraint_ranges(layout, drop_hubs):
    l = layout["link"]
    lw = layout["link_words"]

    @T.macro
    def ranges(Counts, Links, BodyGroup, Ranges, i):
        for side in T.unroll(2, explicit=True):
            body = T.Cast(
                "int32",
                Links[
                    i * lw
                    + T.if_then_else(side == 0, l["solver_body_a"], l["solver_body_b"])
                ],
            )
            group = T.Cast("int32", BodyGroup[body])
            Ranges[side, 0] = 0
            if group > 0:
                Ranges[side, 0] = T.Cast("int32", Counts[group - 1])
            Ranges[side, 1] = T.Cast("int32", Counts[group])
            if drop_hubs:
                slot = T.Cast(
                    "int32",
                    Links[
                        i * lw
                        + T.if_then_else(side == 0, l["vel_slot_a"], l["vel_slot_b"])
                    ],
                )
                if slot != body:
                    Ranges[side, 1] = Ranges[side, 0]

    return ranges


def gpu_reset_luby(layout):
    lw = layout["link_words"]
    length = layout["link"]["len"]

    @T.macro
    def reset(Colors, Rands, Links, Plan, i):
        if i < T.Cast("int32", Plan[layout["plan"]["bound"]]):
            Colors[i] = T.if_then_else(
                Links[i * lw + length] == 0, T.uint32(0), T.uint32(0xFFFFFFFF)
            )
            Rands[i] = hash(T.Cast("uint32", i))

    return reset


def gpu_step_graph_coloring_luby(layout):
    ranges = _constraint_ranges(layout, False)

    @T.macro
    def step(
        Counts,
        Ids,
        Links,
        Colors,
        Rands,
        NumColors,
        BodyGroup,
        CurrentColor,
        Plan,
        i,
        groups,
    ):
        bound = T.Cast("int32", Plan[layout["plan"]["bound"]])
        Ranges = T.alloc_local((2, 2), "int32")
        greatest = T.alloc_local((1,), "int32")
        for chunk in T.serial(T.ceildiv(bound, groups * 64)):
            cid = i + chunk * groups * 64
            if cid < bound:
                if Colors[cid] == T.uint32(0xFFFFFFFF):
                    ranges(Counts, Links, BodyGroup, Ranges, cid)
                    greatest[0] = 1
                    for side in T.unroll(2, explicit=True):
                        for offset in T.serial(Ranges[side, 1] - Ranges[side, 0]):
                            if greatest[0] != 0:
                                other = T.Cast("int32", Ids[Ranges[side, 0] + offset])
                                # A winner from THIS round must still block smaller
                                # neighbors: the Rust reference has the same fix.
                                previous = (Colors[other] != T.uint32(0xFFFFFFFF)) & (
                                    Colors[other] != CurrentColor[0]
                                )
                                wins = (Rands[cid] > Rands[other]) | (
                                    (Rands[cid] == Rands[other]) & (cid >= other)
                                )
                                if not (previous | wins):
                                    greatest[0] = 0
                    if greatest[0] != 0:
                        Colors[cid] = CurrentColor[0]
                    else:
                        T.atomic_add(NumColors[0], T.uint32(1))

    return step


def gpu_reset_topo_gc(layout):
    lw = layout["link_words"]
    length = layout["link"]["len"]

    @T.macro
    def reset(Colors, Colored, Links, Plan, NumColors, Pending, i):
        if i == 0:
            NumColors[0] = T.uint32(0)
        if i < T.Cast("int32", Plan[layout["plan"]["bound"]]):
            Colors[i] = T.uint32(0)
            Colored[i] = T.if_then_else(
                Links[i * lw + length] == 0, T.uint32(1), T.uint32(0)
            )
            Pending[i] = T.uint32(0xFFFFFFFF)

    return reset


def gpu_reset_completion_flag_topo_gc():
    @T.macro
    def reset(NumColors, ContactsIndirect, ColoringIndirect):
        ColoringIndirect[0] = T.if_then_else(
            NumColors[0] != 0, T.uint32(0), ContactsIndirect[0]
        )
        ColoringIndirect[1] = T.if_then_else(
            NumColors[0] != 0, T.uint32(1), ContactsIndirect[1]
        )
        ColoringIndirect[2] = T.if_then_else(
            NumColors[0] != 0, T.uint32(1), ContactsIndirect[2]
        )
        NumColors[0] = T.uint32(1)

    return reset


def gpu_clear_completion_flag_topo_gc():
    @T.macro
    def clear(NumColors, ContactsIndirect, ColoringIndirect):
        for d in T.unroll(3):
            ColoringIndirect[d] = ContactsIndirect[d]
        NumColors[0] = T.uint32(0)

    return clear


def gpu_step_graph_coloring_topo_gc(layout):
    ranges = _constraint_ranges(layout, True)
    first = first_free_color()
    nth = nth_free_color()

    @T.macro
    def step(
        Counts,
        Ids,
        Links,
        Colors,
        Colored,
        NumColors,
        Plan,
        BodyGroup,
        BatchIds,
        Pending,
        i,
        groups,
    ):
        bound = T.Cast("int32", Plan[layout["plan"]["bound"]])
        deterministic = BatchIds[layout["batch"]["deterministic"]] != 0
        Ranges = T.alloc_local((2, 2), "int32")
        Mask = T.alloc_local((2,), "uint32")
        Rank = T.alloc_local((1,), "uint32")
        Result = T.alloc_local((1,), "uint32")
        for chunk in T.serial(T.ceildiv(bound, groups * 64)):
            cid = i + chunk * groups * 64
            if cid < bound:
                if Colored[cid] == 0:
                    ranges(Counts, Links, BodyGroup, Ranges, cid)
                    Mask[0] = T.uint32(1)
                    Mask[1] = T.uint32(0)
                    for side in T.unroll(2, explicit=True):
                        for offset in T.serial(Ranges[side, 1] - Ranges[side, 0]):
                            other = T.Cast("int32", Ids[Ranges[side, 0] + offset])
                            if other != cid:
                                color = Colors[other]
                                if color < T.uint32(32):
                                    Mask[0] |= T.uint32(1) << color
                                else:
                                    Mask[1] |= T.uint32(1) << (color - T.uint32(32))
                    if deterministic:
                        Rank[0] = T.uint32(0)
                        for side in T.unroll(2, explicit=True):
                            for offset in T.serial(Ranges[side, 1] - Ranges[side, 0]):
                                other = T.Cast("int32", Ids[Ranges[side, 0] + offset])
                                if (other < cid) & (Colored[other] == 0):
                                    Rank[0] += T.uint32(1)
                        nth(Mask, Rank[0], Result)
                        Pending[cid] = Result[0]
                    else:
                        first(Mask, Result)
                        Colors[cid] = Result[0]
                        Colored[cid] = T.uint32(1)
                    NumColors[0] = T.uint32(0)
                elif deterministic:
                    Pending[cid] = T.uint32(0xFFFFFFFF)

    return step


def gpu_fix_conflicts_topo_gc(layout):
    ranges = _constraint_ranges(layout, True)
    lw = layout["link_words"]
    length = layout["link"]["len"]

    @T.macro
    def fix(
        Counts,
        Ids,
        Links,
        Colors,
        Colored,
        NumColors,
        Plan,
        BodyGroup,
        BatchIds,
        Pending,
        i,
        groups,
    ):
        bound = T.Cast("int32", Plan[layout["plan"]["bound"]])
        deterministic = BatchIds[layout["batch"]["deterministic"]] != 0
        Ranges = T.alloc_local((2, 2), "int32")
        Pick = T.alloc_local((1,), "uint32")
        Own = T.alloc_local((1,), "uint32")
        for chunk in T.serial(T.ceildiv(bound, groups * 64)):
            cid = i + chunk * groups * 64
            if cid < bound:
                if Links[cid * lw + length] != 0:
                    Pick[0] = T.uint32(0xFFFFFFFF)
                    if deterministic:
                        Pick[0] = Pending[cid]
                    Own[0] = Colors[cid]
                    if Pick[0] != T.uint32(0xFFFFFFFF):
                        Own[0] = Pick[0]
                        Colored[cid] = T.uint32(2)
                    if NumColors[0] == 0:
                        ranges(Counts, Links, BodyGroup, Ranges, cid)
                        for side in T.unroll(2, explicit=True):
                            if (side == 0) | (Colored[cid] != 0):
                                for offset in T.serial(
                                    Ranges[side, 1] - Ranges[side, 0]
                                ):
                                    # The Rust loop breaks after finding a conflict.
                                    # Further comparisons cannot change its zero mark.
                                    other = T.Cast(
                                        "int32", Ids[Ranges[side, 0] + offset]
                                    )
                                    if cid < other:
                                        other_pick = T.alloc_local((1,), "uint32")
                                        other_pick[0] = T.uint32(0xFFFFFFFF)
                                        if deterministic:
                                            other_pick[0] = Pending[other]
                                        other_color = T.alloc_local((1,), "uint32")
                                        other_color[0] = Colors[other]
                                        if other_pick[0] != T.uint32(0xFFFFFFFF):
                                            other_color[0] = other_pick[0]
                                        if Own[0] == other_color[0]:
                                            Colored[cid] = T.uint32(0)
                        if (Pick[0] != T.uint32(0xFFFFFFFF)) & (
                            Colored[cid] == T.uint32(2)
                        ):
                            Colors[cid] = Pick[0]
                            Colored[cid] = T.uint32(1)

    return fix
