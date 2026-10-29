"""Counterpart of utils/radix_sort/sort_reduce.rs; native reduction order."""

import tilelang.language as T
from .sorting import num_workgroups, div_ceil


def gpu_sort_reduce(layout, cpu):
    f = layout["sort_uniforms"]

    @T.macro
    def one(Config, NumKeys, Counts, Reduced, Sums, group, batch, lane):
        groups = num_workgroups(T.cast(NumKeys[batch], "int32"))
        max_groups = num_workgroups(T.cast(Config[f["max_keys_per_batch"]], "int32"))
        reduce_per_bin = div_ceil(groups, 1024)
        active = group < 16 * reduce_per_bin
        for t in T.serial(256 if cpu else 1):
            who = t if cpu else lane
            Sums[who] = T.uint32(0)
            if active:
                bin_id = group // reduce_per_bin
                base = group % reduce_per_bin * 1024
                for item in T.serial(4):
                    index = base + item * 256 + who
                    if index < groups:
                        Sums[who] = (
                            Sums[who]
                            + Counts[batch * 16 * max_groups + bin_id * groups + index]
                        )
        for step in T.serial(8):
            if not cpu:
                T.sync_threads()
            for t in T.serial(256 if cpu else 1):
                who = t if cpu else lane
                if who < (128 >> step):
                    Sums[who] = Sums[who] + Sums[who + (128 >> step)]
        if active and lane == 0:
            Reduced[batch * 1024 + group] = Sums[0]

    return one
