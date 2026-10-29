"""Counterpart of utils/radix_sort/sort_scatter.rs; two stable 2-bit lane sorts."""

import tilelang.language as T
from .sorting import num_workgroups, extract_digit
from radix_support import inclusive_scan


def gpu_sort_scatter(layout, cpu):
    f = layout["sort_uniforms"]
    scan = inclusive_scan(cpu)
    scan_hist = inclusive_scan(cpu, 16)

    @T.macro
    def one(
        Config,
        NumKeys,
        Keys,
        Values,
        Counts,
        OutKeys,
        OutValues,
        Aux,
        OutAux,
        Scratch,
        Temp,
        BinOffsets,
        Histogram,
        Local,
        group,
        batch,
        lane,
    ):
        n = T.cast(NumKeys[batch], "int32")
        groups = num_workgroups(n)
        capacity = T.cast(Config[f["max_keys_per_batch"]], "int32")
        max_groups = num_workgroups(capacity)
        active = group < groups
        has_aux = Config[f["has_aux"]] != T.uint32(0)
        shift = Config[f["shift"]]
        for t in T.serial(256 if cpu else 1):
            who = t if cpu else lane
            if active and who < 16:
                BinOffsets[who] = Counts[batch * 16 * max_groups + who * groups + group]
        if not cpu:
            T.sync_threads()
        for item in T.serial(4):
            for t in T.serial(256 if cpu else 1):
                who = t if cpu else lane
                if who < 16:
                    Histogram[who] = T.uint32(0)
                Local[who, 0] = T.uint32(0xFFFFFFFF)
                Local[who, 1] = T.uint32(0)
                Local[who, 2] = T.uint32(0)
                index = group * 1024 + who + item * 256
                if active and index < n:
                    Local[who, 0] = Keys[batch * capacity + index]
                    Local[who, 1] = Values[batch * capacity + index]
                    if has_aux:
                        Local[who, 2] = Aux[batch * capacity + index]
            for bit in T.serial(2):
                for t in T.serial(256 if cpu else 1):
                    who = t if cpu else lane
                    digit = extract_digit(Local[who, 0], shift)
                    Local[who, 3] = (digit >> T.cast(bit * 2, "uint32")) & T.uint32(3)
                    Scratch[who] = T.uint32(1) << (Local[who, 3] * T.uint32(8))
                scan(Scratch, Temp, lane)
                for t in T.serial(256 if cpu else 1):
                    who = t if cpu else lane
                    packed = Scratch[255]
                    prefix = T.alloc_local((1,), "uint32")
                    prefix[0] = (
                        (packed << T.uint32(8))
                        + (packed << T.uint32(16))
                        + (packed << T.uint32(24))
                    )
                    if who > 0:
                        prefix[0] = prefix[0] + Scratch[who - 1]
                    Local[who, 4] = (
                        prefix[0] >> (Local[who, 3] * T.uint32(8))
                    ) & T.uint32(255)
                if not cpu:
                    T.sync_threads()
                for field in T.serial(3):
                    if field < 2 or has_aux:
                        for t in T.serial(256 if cpu else 1):
                            who = t if cpu else lane
                            Scratch[T.cast(Local[who, 4], "int32")] = Local[who, field]
                        if not cpu:
                            T.sync_threads()
                        for t in T.serial(256 if cpu else 1):
                            who = t if cpu else lane
                            Local[who, field] = Scratch[who]
                        if not cpu:
                            T.sync_threads()
            for t in T.serial(256 if cpu else 1):
                who = t if cpu else lane
                digit = T.cast(extract_digit(Local[who, 0], shift), "int32")
                T.atomic_add(Histogram[digit], T.uint32(1))
            if not cpu:
                T.sync_threads()
            for t in T.serial(256 if cpu else 1):
                who = t if cpu else lane
                if who < 16:
                    Scratch[who] = Histogram[who]
            scan_hist(Scratch, Temp, lane)
            for t in T.serial(256 if cpu else 1):
                who = t if cpu else lane
                digit = T.cast(extract_digit(Local[who, 0], shift), "int32")
                Local[who, 4] = T.uint32(0)
                if active:
                    Local[who, 4] = BinOffsets[digit]
            if not cpu:
                T.sync_threads()
            if active:
                for t in T.serial(256 if cpu else 1):
                    who = t if cpu else lane
                    digit = T.cast(extract_digit(Local[who, 0], shift), "int32")
                    offset = T.alloc_local((1,), "uint32")
                    offset[0] = T.cast(who, "uint32")
                    if digit > 0:
                        offset[0] = offset[0] - Scratch[digit - 1]
                    total = Local[who, 4] + offset[0]
                    if total < T.cast(n, "uint32"):
                        index = batch * capacity + T.cast(total, "int32")
                        OutKeys[index] = Local[who, 0]
                        OutValues[index] = Local[who, 1]
                        if has_aux:
                            OutAux[index] = Local[who, 2]
                    if who < 16:
                        BinOffsets[who] = BinOffsets[who] + Histogram[who]
            if not cpu:
                T.sync_threads()

    return one
