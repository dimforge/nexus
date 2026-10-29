"""Counterpart of utils/radix_sort/sort_scan.rs; transposed exclusive prefix."""

import tilelang.language as T
from .sorting import num_workgroups, div_ceil
from radix_support import inclusive_scan


def gpu_sort_scan(cpu):
    scan = inclusive_scan(cpu)

    @T.macro
    def one(NumKeys, Reduced, Sums, Temp, Lds, batch, lane):
        reduce_groups = 16 * div_ceil(
            num_workgroups(T.cast(NumKeys[batch], "int32")), 1024
        )
        for t in T.serial(256 if cpu else 1):
            who = t if cpu else lane
            for item in T.serial(4):
                index = item * 256 + who
                Lds[index % 4, index // 4] = Reduced[batch * 1024 + index]
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
        for t in T.serial(256 if cpu else 1):
            who = t if cpu else lane
            if who > 0:
                for item in T.serial(4):
                    Lds[item, who] = Lds[item, who] + Sums[who - 1]
        if not cpu:
            T.sync_threads()
        for t in T.serial(256 if cpu else 1):
            who = t if cpu else lane
            for item in T.serial(4):
                index = item * 256 + who
                if index < reduce_groups:
                    Reduced[batch * 1024 + index] = Lds[index % 4, index // 4]

    return one
