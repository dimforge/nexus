"""Counterpart of utils/radix_sort/init_batch_ids.rs."""

import tilelang.language as T


def gpu_init_sort_batched(layout):
    f = layout["sort_uniforms"]

    @T.macro
    def one(Config, NumKeys, Keys, Values, OutKeys, OutValues, BatchIds, gid):
        per_batch = T.cast(Config[f["max_keys_per_batch"]], "int32")
        batches = T.cast(Config[f["shift"]], "int32")
        if gid < per_batch * batches:
            batch = gid // per_batch
            local = gid % per_batch
            BatchIds[gid] = T.cast(batch, "uint32")
            if local < T.cast(NumKeys[batch], "int32"):
                OutKeys[gid] = Keys[gid]
                OutValues[gid] = Values[gid]
            else:
                OutKeys[gid] = T.uint32(0xFFFFFFFF)
                OutValues[gid] = T.uint32(0)

    return one
