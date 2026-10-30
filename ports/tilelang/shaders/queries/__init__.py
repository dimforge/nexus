"""Counterpart of queries/mod.rs vector approximate equality."""

import tilelang.language as T


@T.macro
def relative_eq(a, b, Result):
    close = T.alloc_local((1,), "bool")
    relative = T.alloc_local((1,), "bool")
    close[0] = True
    relative[0] = True
    for d in T.serial(3):
        delta = a[d] - b[d]
        diff = T.if_then_else(delta < 0.0, -delta, delta)
        aa = T.if_then_else(a[d] < 0.0, -a[d], a[d])
        bb = T.if_then_else(b[d] < 0.0, -b[d], b[d])
        close[0] = close[0] & (diff <= T.float32(1.1920929e-7))
        relative[0] = relative[0] & (diff <= T.max(aa, bb) * T.float32(1.1920929e-7))
    Result[0] = close[0] | relative[0]
