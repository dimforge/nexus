"""Counterpart of utils/radix_sort/sort_count.rs; native 256-lane histogram."""

import tilelang.language as T
from .sorting import num_workgroups, extract_digit


def gpu_sort_count(layout, cpu):
    f = layout["sort_uniforms"]

    @T.macro
    def one(Config, NumKeys, Keys, Counts, Histogram, group, batch, lane):
        count = T.cast(NumKeys[batch], "int32")
        groups = num_workgroups(count)
        cap = T.cast(Config[f["max_keys_per_batch"]], "int32")
        max_groups = num_workgroups(cap)
        active = group < groups
        for t in T.serial(256 if cpu else 1):
            who = t if cpu else lane
            if active and who < 16:
                Histogram[who] = T.uint32(0)
        if not cpu:
            T.sync_threads()
        if active:
            for t in T.serial(256 if cpu else 1):
                who = t if cpu else lane
                for item in T.serial(4):
                    index = group * 1024 + who + item * 256
                    if index < count:
                        digit = T.cast(
                            extract_digit(
                                Keys[batch * cap + index], Config[f["shift"]]
                            ),
                            "int32",
                        )
                        T.atomic_add(Histogram[digit], T.uint32(1))
        if not cpu:
            T.sync_threads()
        for t in T.serial(256 if cpu else 1):
            who = t if cpu else lane
            if active and who < 16:
                Counts[batch * 16 * max_groups + who * groups + group] = Histogram[who]

    return one
