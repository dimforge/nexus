"""Counterpart of utils/radix_sort/sort_scan_add.rs."""

import tilelang.language as T
from .sorting import num_workgroups, div_ceil
from radix_support import inclusive_scan


def gpu_sort_scan_add(layout, cpu, counts_words):
    f = layout["sort_uniforms"]
    scan = inclusive_scan(cpu)

    @T.macro
    def one(Config, NumKeys, Reduced, Counts, Sums, Temp, Lds, group, batch, lane):
        groups = num_workgroups(T.cast(NumKeys[batch], "int32"))
        max_groups = num_workgroups(T.cast(Config[f["max_keys_per_batch"]], "int32"))
        reduce_per_bin = div_ceil(groups, 1024)
        active = group < 16 * reduce_per_bin
        for t in T.serial(256 if cpu else 1):
            who = t if cpu else lane
            for item in T.serial(4):
                index = item * 256 + who
                Lds[index % 4, index // 4] = T.uint32(0)
                if active:
                    pos = (
                        batch * 16 * max_groups
                        + (group // reduce_per_bin) * groups
                        + (group % reduce_per_bin) * 1024
                        + index
                    )
                    if pos < counts_words:
                        Lds[index % 4, index // 4] = Counts[pos]
        if not cpu:
            T.sync_threads()
        for t in T.serial(256 if cpu else 1):
            who = t if cpu else lane
            Sums[who] = T.uint32(0)
            for item in T.serial(4):
                value = Lds[item, who]
                Lds[item, who] = Sums[who]
                Sums[who] = Sums[who] + value
        scan(Sums, Temp, lane)
        if active:
            for t in T.serial(256 if cpu else 1):
                who = t if cpu else lane
                offset = T.alloc_local((1,), "uint32")
                offset[0] = Reduced[batch * 1024 + group]
                if who > 0:
                    offset[0] = offset[0] + Sums[who - 1]
                for item in T.serial(4):
                    Lds[item, who] = Lds[item, who] + offset[0]
        if not cpu:
            T.sync_threads()
        if active:
            for t in T.serial(256 if cpu else 1):
                who = t if cpu else lane
                for item in T.serial(4):
                    index = item * 256 + who
                    data_index = group % reduce_per_bin * 1024 + index
                    if data_index < groups:
                        Counts[
                            batch * 16 * max_groups
                            + (group // reduce_per_bin) * groups
                            + data_index
                        ] = Lds[index % 4, index // 4]

    return one
