"""Counterpart of dynamics/canonical_order.rs::gpu_stabilize_body_constraint_ids.

One group per body, 64 lanes. The (ID, original position) tie break gives every
entry a unique output rank, including duplicates. There are no shared-memory or
subgroup assumptions and no in-kernel global barriers.
"""

import tilelang.language as T


def gpu_stabilize_body_constraint_ids(case):
    @T.macro
    def stabilize(Counts, Ids, Sorted, body, lane):
        if body < case["rows"]:
            start = T.alloc_local((1,), "int32")
            start[0] = 0
            if body > 0:
                start[0] = T.cast(Counts[body - 1], "int32")
            end = T.cast(Counts[body], "int32")
            rank = T.alloc_local((1,), "int32")
            for block in T.serial(T.ceildiv(end - start[0], 64)):
                e = start[0] + block * 64 + lane
                if e < end:
                    value = Ids[e]
                    rank[0] = 0
                    for offset in T.serial(end - start[0]):
                        q = start[0] + offset
                        other = Ids[q]
                        if (other < value) | ((other == value) & (q < e)):
                            rank[0] = rank[0] + 1
                    Sorted[start[0] + rank[0]] = value

    return stabilize
