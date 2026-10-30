"""Counterpart of shapes/segment.rs."""

import tilelang.language as T
from kernel_support import dot

from ..queries import relative_eq
from ..queries.projection import edge, vertex


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


@T.macro
def project_local_point_and_get_location(a, b, pt, solid_flag, Values, Flags):
    ab = T.alloc_local((3,), "float32")
    ap = T.alloc_local((3,), "float32")
    out = T.alloc_local((3,), "float32")
    inside = T.alloc_local((1,), "bool")
    for d in T.serial(3):
        ab[d] = b[d] - a[d]
        ap[d] = pt[d] - a[d]
    ab_ap = dot(ab, ap)
    sqnab = dot(ab, ab)
    if ab_ap <= 0.0:
        relative_eq(a, pt, inside)
        vertex(a, T.uint32(0), inside[0], Values, Flags)
    elif ab_ap >= sqnab:
        relative_eq(b, pt, inside)
        vertex(b, T.uint32(1), inside[0], Values, Flags)
    else:
        u = ab_ap / sqnab
        for d in T.serial(3):
            out[d] = a[d] + ab[d] * u
        relative_eq(out, pt, inside)
        edge(out, 1.0 - u, u, T.uint32(0), inside[0], Values, Flags)
