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


def gpu_contact_sort_keys(layout):
    c, m, b, p = layout["contact"], layout["manifold"], layout["batch"], layout["plan"]

    @T.macro
    def one(
        Contacts,
        Plan,
        InOrder,
        Keys,
        Values,
        Selector,
        SortCount,
        BatchIds,
        gid,
        groups,
    ):
        count = T.cast(Plan[p["bound"]], "int32")
        selector = T.cast(Selector[0], "int64")
        field = T.cast(selector % 4, "int32")
        if gid == 0:
            SortCount[0] = T.cast(count, "uint32")
        for i in T.serial(gid, count, groups * 64):
            index = T.alloc_local((1,), "int32")
            index[0] = i
            if selector < 4:
                index[0] = T.cast(InOrder[i], "int32")
            base = index[0] * layout["contact_words"]
            key = T.alloc_local((1,), "uint32")
            key[0] = T.uint32(0xFFFFFFFF)
            if Contacts[base + c["contact"] + m["len"]] != 0:
                if field == 3:
                    key[0] = (
                        Contacts[base + c["colliders"]]
                        << BatchIds[b["contact_sort_collider_shift"]]
                    ) | Contacts[base + c["colliders"] + 1]
                elif field == 2:
                    key[0] = Contacts[base + c["colliders"]]
                elif field == 1:
                    key[0] = Contacts[base + c["colliders"] + 1]
                else:
                    key[0] = Contacts[base + c["subshape"]]
            Keys[i] = key[0]
            Values[i] = T.cast(index[0], "uint32")

    return one


def gpu_permute_contacts(layout):
    @T.macro
    def one(Src, Dst, Order, Plan, gid, groups):
        for i in T.serial(
            gid, T.cast(Plan[layout["plan"]["bound"]], "int32"), groups * 64
        ):
            index = T.cast(Order[i], "int32")
            source = index * layout["contact_words"]
            dest = i * layout["contact_words"]
            if (
                Src[source + layout["contact"]["contact"] + layout["manifold"]["len"]]
                != 0
            ):
                for d in T.serial(layout["contact_words"]):
                    Dst[dest + d] = Src[source + d]
            else:
                # IndexedManifold::default is derived from Rust in the layout.
                for d in T.serial(layout["contact_words"]):
                    Dst[dest + d] = T.uint32(0)

    return one
