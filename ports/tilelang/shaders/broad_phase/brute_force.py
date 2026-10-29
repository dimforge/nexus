"""Counterpart of broad_phase/brute_force.rs."""

import tilelang.language as T
from kernel_support import f32
from ..shapes.shape import compute_aabb
from ..bounding_volumes.aabb import intersects
from .lbvh import scratch_start


def gpu_bf_compute_aabbs(layout, case):
    compute = compute_aabb(layout)
    batch = layout["batch"]
    box = layout["aabb"]

    @T.macro
    def one(Poses, Shapes, Aabbs, BatchIds, Vertices, gid):
        n = T.cast(BatchIds[batch["colliders_len"]], "int32")
        nb = T.cast(BatchIds[batch["num_batches"]], "int32")
        if gid < n * nb:
            b = gid // n
            i = gid % n
            index = i * nb + b
            out = (scratch_start(BatchIds, batch, b) + i) * layout["aabb_words"]
            mins = T.alloc_local((3,), "float32")
            maxs = T.alloc_local((3,), "float32")
            compute(Shapes, Poses, Vertices, index, mins, maxs)
            for d in T.serial(3):
                Aabbs[out + box["mins"] + d] = T.reinterpret(mins[d], "uint32")
                Aabbs[out + box["maxs"] + d] = T.reinterpret(maxs[d], "uint32")
            Aabbs[out + box["padding0"]] = T.uint32(0)
            Aabbs[out + box["padding1"]] = T.uint32(0)

    return one


def _test_pair(layout):
    b = layout["batch"]
    g = layout["groups"]
    a = layout["aabb"]
    p = layout["rbd_params"]

    @T.macro
    def one(Aabbs, Pairs, PairCount, Groups, BatchIds, Filters, RbdParams, batch, i, j):
        nb = T.cast(BatchIds[b["num_batches"]], "int32")
        gi = (i * nb + batch) * layout["group_words"]
        gj = (j * nb + batch) * layout["group_words"]
        left = (Groups[gi + g["memberships"]] & Groups[gj + g["filter"]]) != T.uint32(0)
        right = (Groups[gj + g["memberships"]] & Groups[gi + g["filter"]]) != T.uint32(
            0
        )
        either = (Groups[gi + g["test_mode"]] == T.uint32(1)) & (
            Groups[gj + g["test_mode"]] == T.uint32(1)
        )
        allowed = T.if_then_else(either, left | right, left & right)
        fi = (i * nb + batch) * 2
        fj = (j * nb + batch) * 2
        if (
            allowed
            and Filters[fi] != Filters[fj]
            and (Filters[fi + 1] == T.uint32(0) or Filters[fi + 1] != Filters[fj + 1])
        ):
            start = scratch_start(BatchIds, b, batch)
            ai = (start + i) * layout["aabb_words"]
            aj = (start + j) * layout["aabb_words"]
            dilation = f32(RbdParams, p["normalized_prediction_distance"]) * f32(
                RbdParams, p["length_unit"]
            )
            a_min = T.alloc_local((3,), "float32")
            a_max = T.alloc_local((3,), "float32")
            b_min = T.alloc_local((3,), "float32")
            b_max = T.alloc_local((3,), "float32")
            for d in T.serial(3):
                a_min[d] = f32(Aabbs, ai + a["mins"] + d) - dilation
                a_max[d] = f32(Aabbs, ai + a["maxs"] + d) + dilation
                b_min[d] = f32(Aabbs, aj + a["mins"] + d)
                b_max[d] = f32(Aabbs, aj + a["maxs"] + d)
            if intersects(a_min, a_max, b_min, b_max):
                target = T.alloc_local((1,), "uint32")
                target[0] = T.atomic_add(PairCount[0], T.uint32(1), return_prev=True)
                if target[0] < BatchIds[b["collision_pairs_capacity"]]:
                    Pairs[
                        T.cast(target[0], "int32") * layout["pair_words"]
                        + layout["pair_colliders"]
                    ] = T.cast(i * nb + batch, "uint32")
                    Pairs[
                        T.cast(target[0], "int32") * layout["pair_words"]
                        + layout["pair_colliders"]
                        + 1
                    ] = T.cast(j * nb + batch, "uint32")

    return one


def gpu_bf_find_pairs(layout, case):
    test = _test_pair(layout)
    b = layout["batch"]

    @T.macro
    def one(Aabbs, Pairs, PairCount, Groups, BatchIds, Filters, RbdParams, gid):
        n = T.cast(BatchIds[b["colliders_len"]], "int32")
        nb = T.cast(BatchIds[b["num_batches"]], "int32")
        nn = n * n
        if gid < nn * nb:
            batch = gid // nn
            r = gid % nn
            i = r // n
            j = r % n
            if i < j:
                test(
                    Aabbs,
                    Pairs,
                    PairCount,
                    Groups,
                    BatchIds,
                    Filters,
                    RbdParams,
                    batch,
                    i,
                    j,
                )

    return one


def gpu_bf_find_pairs_serial(layout, case):
    test = _test_pair(layout)
    b = layout["batch"]

    @T.macro
    def one(Aabbs, Pairs, PairCount, Groups, BatchIds, Filters, RbdParams, gid):
        n = T.cast(BatchIds[b["colliders_len"]], "int32")
        nb = T.cast(BatchIds[b["num_batches"]], "int32")
        if nb > 0:
            batch = gid % nb
            i = gid // nb
            if i < n:
                for j in T.serial(i + 1, n):
                    test(
                        Aabbs,
                        Pairs,
                        PairCount,
                        Groups,
                        BatchIds,
                        Filters,
                        RbdParams,
                        batch,
                        i,
                        j,
                    )

    return one
