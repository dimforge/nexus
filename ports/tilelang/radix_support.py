"""Backend execution helpers for cooperative integer phases, including serial C."""

import tilelang.language as T


def inclusive_scan(cpu, count=256):
    @T.macro
    def one(Scratch, Temp, lane):
        for step in T.serial(8 if count == 256 else 4):
            if not cpu:
                T.sync_threads()
            for t in T.serial(count if cpu else 1):
                who = t if cpu else lane
                if who < count:
                    value = T.alloc_local((1,), "uint32")
                    value[0] = Scratch[who]
                    if who >= (1 << step):
                        value[0] = value[0] + Scratch[who - (1 << step)]
                    Temp[who] = value[0]
            if not cpu:
                T.sync_threads()
            for t in T.serial(count if cpu else 1):
                who = t if cpu else lane
                if who < count:
                    Scratch[who] = Temp[who]
        if not cpu:
            T.sync_threads()

    return one
