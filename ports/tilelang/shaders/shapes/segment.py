"""Counterpart of shapes/segment.rs."""

import tilelang.language as T
from kernel_support import dot


@T.macro
def project_local_point(a, b, pt, out):
    ab = T.alloc_local((3,), "float32")
    ap = T.alloc_local((3,), "float32")
    for d in T.serial(3):
        ab[d] = b[d] - a[d]
        ap[d] = pt[d] - a[d]
    ab_ap = dot(ab, ap)
    sqnab = dot(ab, ab)
    if ab_ap <= T.float32(0):
        for d in T.serial(3):
            out[d] = a[d]
    elif ab_ap >= sqnab:
        for d in T.serial(3):
            out[d] = b[d]
    else:
        u = ab_ap / sqnab
        for d in T.serial(3):
            out[d] = a[d] + ab[d] * u
