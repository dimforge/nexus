"""Counterpart of utils/radix_sort/init_indirect_dispatches.rs."""

import tilelang.language as T
from .sorting import div_ceil


def gpu_init_sort_dispatch(batches):
    @T.macro
    def one(NumKeys, SortDispatch, ReduceDispatch):
        count = T.alloc_local((1,), "uint32")
        count[0] = T.uint32(0)
        for b in T.serial(batches):
            count[0] = T.max(count[0], NumKeys[b])
        groups = div_ceil(count[0], T.uint32(1024))
        SortDispatch[0] = groups
        SortDispatch[1] = T.uint32(batches)
        SortDispatch[2] = T.uint32(1)
        ReduceDispatch[0] = T.uint32(16) * div_ceil(groups, T.uint32(1024))
        ReduceDispatch[1] = T.uint32(batches)
        ReduceDispatch[2] = T.uint32(1)

    return one
