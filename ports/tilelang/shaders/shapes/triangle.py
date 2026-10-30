"""Counterpart of shapes/triangle.rs; local AABB."""

import tilelang.language as T
from kernel_support import cross, dot

from ..queries import relative_eq
from ..queries.projection import edge, location, solid, vertex


@T.macro
def aabb(a, b, c, mins, maxs):
    for d in T.serial(3):
        mins[d] = T.min(T.min(a[d], b[d]), c[d])
        maxs[d] = T.max(T.max(a[d], b[d]), c[d])


@T.macro
def stable_check_edges_voronoi(
    ab, ac, bc, ap, bp, cp, ab_ap, ab_bp, ac_ap, ac_cp, ac_bp, ab_cp, Info, Feature
):
    normal = T.alloc_local((3,), "float32")
    tmp = T.alloc_local((3,), "float32")
    cross(ab, ac, normal)
    cross(ab, ap, tmp)
    vc = dot(normal, tmp)
    for d in T.serial(3):
        Info[d] = 0.0
    if (vc < 0.0) & (ab_ap >= 0.0) & (ab_bp <= 0.0):
        Feature[0] = T.uint32(0)
    else:
        cross(ac, cp, tmp)
        vb = -dot(normal, tmp)
        if (vb < 0.0) & (ac_ap >= 0.0) & (ac_cp <= 0.0):
            Feature[0] = T.uint32(1)
        else:
            cross(bc, bp, tmp)
            va = dot(normal, tmp)
            if (va < 0.0) & (ac_bp - ab_bp >= 0.0) & (ab_cp - ac_cp >= 0.0):
                Feature[0] = T.uint32(2)
            else:
                Feature[0] = T.if_then_else(
                    dot(normal, ap) >= 0.0, T.uint32(3), T.uint32(4)
                )
                Info[0] = va
                Info[1] = vb
                Info[2] = vc


@T.macro
def project_local_point_and_get_location(a, b, c, pt, solid_flag, Values, Flags):
    ab = T.alloc_local((3,), "float32")
    ac = T.alloc_local((3,), "float32")
    ap = T.alloc_local((3,), "float32")
    bp = T.alloc_local((3,), "float32")
    cp = T.alloc_local((3,), "float32")
    bc = T.alloc_local((3,), "float32")
    inside = T.alloc_local((1,), "bool")
    out = T.alloc_local((3,), "float32")
    bary = T.alloc_local((3,), "float32")
    for d in T.serial(3):
        ab[d] = b[d] - a[d]
        ac[d] = c[d] - a[d]
        ap[d] = pt[d] - a[d]
        bp[d] = pt[d] - b[d]
        cp[d] = pt[d] - c[d]
        bc[d] = c[d] - b[d]
    ab_ap = dot(ab, ap)
    ac_ap = dot(ac, ap)
    ab_bp = dot(ab, bp)
    ac_bp = dot(ac, bp)
    ab_cp = dot(ab, cp)
    ac_cp = dot(ac, cp)
    if (ab_ap <= 0.0) & (ac_ap <= 0.0):
        relative_eq(pt, a, inside)
        vertex(a, T.uint32(0), inside[0], Values, Flags)
    elif (ab_bp >= 0.0) & (ac_bp <= ab_bp):
        relative_eq(pt, b, inside)
        vertex(b, T.uint32(1), inside[0], Values, Flags)
    elif (ac_cp >= 0.0) & (ab_cp <= ac_cp):
        relative_eq(pt, c, inside)
        vertex(c, T.uint32(2), inside[0], Values, Flags)
    else:
        info = T.alloc_local((3,), "float32")
        feature = T.alloc_local((1,), "uint32")
        stable_check_edges_voronoi(
            ab,
            ac,
            bc,
            ap,
            bp,
            cp,
            ab_ap,
            ab_bp,
            ac_ap,
            ac_cp,
            ac_bp,
            ab_cp,
            info,
            feature,
        )
        if feature[0] == 0:
            v = ab_ap / dot(ab, ab)
            for d in T.serial(3):
                out[d] = a[d] + ab[d] * v
            relative_eq(pt, out, inside)
            edge(out, 1.0 - v, v, T.uint32(0), inside[0], Values, Flags)
        elif feature[0] == 1:
            w = ac_ap / dot(ac, ac)
            for d in T.serial(3):
                out[d] = a[d] + ac[d] * w
            relative_eq(pt, out, inside)
            edge(out, 1.0 - w, w, T.uint32(2), inside[0], Values, Flags)
        elif feature[0] == 2:
            w = dot(bc, bp) / dot(bc, bc)
            for d in T.serial(3):
                out[d] = b[d] + bc[d] * w
            relative_eq(pt, out, inside)
            edge(out, 1.0 - w, w, T.uint32(1), inside[0], Values, Flags)
        elif info[0] + info[1] + info[2] != 0.0:
            denom = 1.0 / ((info[0] + info[1]) + info[2])
            v = info[1] * denom
            w = info[2] * denom
            bary[0] = (1.0 - v) - w
            bary[1] = v
            bary[2] = w
            for d in T.serial(3):
                out[d] = (a[d] + ab[d] * v) + ac[d] * w
            relative_eq(pt, out, inside)
            location(out, bary, T.uint32(2), feature[0], inside[0], Values, Flags)
        elif solid_flag:
            solid(pt, Values, Flags)
        else:
            v = ab_ap / (ab_ap - ab_bp)
            w = ac_ap / (ac_ap - ac_cp)
            u = (ac_bp - ab_bp) / (((ac_bp - ab_bp) + ab_cp) - ac_cp)
            d_ab = dot(ap, ap) - (dot(ab, ab) * v) * v
            d_ac = dot(ap, ap) - (dot(ac, ac) * w) * w
            d_bc = dot(bp, bp) - (dot(bc, bc) * u) * u
            if d_ab < d_ac:
                if d_ab < d_bc:
                    for d in T.serial(3):
                        out[d] = a[d] + ab[d] * v
                    edge(out, 1.0 - v, v, T.uint32(0), True, Values, Flags)
                else:
                    for d in T.serial(3):
                        out[d] = b[d] + bc[d] * u
                    edge(out, 1.0 - u, u, T.uint32(1), True, Values, Flags)
            elif d_ac < d_bc:
                for d in T.serial(3):
                    out[d] = a[d] + ac[d] * w
                edge(out, 1.0 - w, w, T.uint32(2), True, Values, Flags)
            else:
                for d in T.serial(3):
                    out[d] = b[d] + bc[d] * u
                edge(out, 1.0 - u, u, T.uint32(1), True, Values, Flags)


@T.macro
def local_support_point(a, b, c, direction, Output):
    da = dot(direction, a)
    db = dot(direction, b)
    dc = dot(direction, c)
    for d in T.serial(3):
        Output[d] = T.if_then_else(
            da > db,
            T.if_then_else(da > dc, a[d], c[d]),
            T.if_then_else(db > dc, b[d], c[d]),
        )
