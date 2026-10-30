"""Counterpart of shapes/tetrahedron.rs Voronoi projection."""

import tilelang.language as T
from kernel_support import cross, dot

from ..queries.projection import edge, location, solid, vertex


@T.macro
def check_edge(
    index, a, nabc, nabd, ap, ab, ap_ab, bp_ab, Distances, Values, Flags, Valid
):
    ab_ab = ap_ab - bp_ab
    prod = T.alloc_local((3,), "float32")
    cross(ap, ab, prod)
    dabc = dot(prod, nabc)
    dabd = dot(prod, nabd)
    Distances[0] = dabc
    Distances[1] = dabd
    Valid[0] = False
    if (
        (ab_ab != 0.0)
        & (dabc >= 0.0)
        & (dabd >= 0.0)
        & (ap_ab >= 0.0)
        & (ap_ab <= ab_ab)
    ):
        u = ap_ab / ab_ab
        point = T.alloc_local((3,), "float32")
        for d in T.serial(3):
            point[d] = a[d] + ab[d] * u
        edge(point, 1.0 - u, u, index, False, Values, Flags)
        Valid[0] = True


@T.macro
def check_face(
    index, a, b, c, ap, bp, cp, ab, ac, ad, dabc, dbca, dacb, Values, Flags, Valid
):
    Valid[0] = False
    if (dabc < 0.0) & (dbca < 0.0) & (dacb < 0.0):
        n = T.alloc_local((3,), "float32")
        cross(ab, ac, n)
        if dot(n, ad) * dot(n, ap) < 0.0:
            length = T.sqrt(dot(n, n))
            if not (length < T.float32(1.0e-7)):
                normal = T.alloc_local((3,), "float32")
                prod = T.alloc_local((3,), "float32")
                for d in T.serial(3):
                    normal[d] = n[d] / length
                cross(ap, bp, prod)
                vc = dot(normal, prod)
                cross(bp, cp, prod)
                va = dot(normal, prod)
                cross(cp, ap, prod)
                vb = dot(normal, prod)
                inverse = 1.0 / ((va + vb) + vc)
                bary = T.alloc_local((3,), "float32")
                point = T.alloc_local((3,), "float32")
                bary[0] = va * inverse
                bary[1] = vb * inverse
                bary[2] = vc * inverse
                for d in T.serial(3):
                    point[d] = (a[d] * bary[0] + b[d] * bary[1]) + c[d] * bary[2]
                location(point, bary, T.uint32(2), index, False, Values, Flags)
                Valid[0] = True


@T.macro
def project_local_point_and_get_location(a, b, c, d, pt, solid_flag, Values, Flags):
    ab = T.alloc_local((3,), "float32")
    ac = T.alloc_local((3,), "float32")
    ad = T.alloc_local((3,), "float32")
    ap = T.alloc_local((3,), "float32")
    bc = T.alloc_local((3,), "float32")
    bd = T.alloc_local((3,), "float32")
    bp = T.alloc_local((3,), "float32")
    cd = T.alloc_local((3,), "float32")
    cp = T.alloc_local((3,), "float32")
    dp = T.alloc_local((3,), "float32")
    for k in T.serial(3):
        ab[k] = b[k] - a[k]
        ac[k] = c[k] - a[k]
        ad[k] = d[k] - a[k]
        ap[k] = pt[k] - a[k]
        bc[k] = c[k] - b[k]
        bd[k] = d[k] - b[k]
        bp[k] = pt[k] - b[k]
        cd[k] = d[k] - c[k]
        cp[k] = pt[k] - c[k]
        dp[k] = pt[k] - d[k]
    ap_ab = dot(ap, ab)
    ap_ac = dot(ap, ac)
    ap_ad = dot(ap, ad)
    bp_bc = dot(bp, bc)
    bp_bd = dot(bp, bd)
    bp_ab = dot(bp, ab)
    cp_ac = dot(cp, ac)
    cp_bc = dot(cp, bc)
    cp_cd = dot(cp, cd)
    dp_cd = dot(dp, cd)
    dp_bd = dot(dp, bd)
    dp_ad = dot(dp, ad)
    if (ap_ab <= 0.0) & (ap_ac <= 0.0) & (ap_ad <= 0.0):
        vertex(a, T.uint32(0), False, Values, Flags)
    elif (bp_bc <= 0.0) & (bp_bd <= 0.0) & (bp_ab >= 0.0):
        vertex(b, T.uint32(1), False, Values, Flags)
    elif (cp_cd <= 0.0) & (cp_bc >= 0.0) & (cp_ac >= 0.0):
        vertex(c, T.uint32(2), False, Values, Flags)
    elif (dp_ad >= 0.0) & (dp_bd >= 0.0) & (dp_cd >= 0.0):
        vertex(d, T.uint32(3), False, Values, Flags)
    else:
        nabc = T.alloc_local((3,), "float32")
        nabd = T.alloc_local((3,), "float32")
        nacd = T.alloc_local((3,), "float32")
        nbcd = T.alloc_local((3,), "float32")
        negabc = T.alloc_local((3,), "float32")
        negabd = T.alloc_local((3,), "float32")
        negacd = T.alloc_local((3,), "float32")
        negbcd = T.alloc_local((3,), "float32")
        negab = T.alloc_local((3,), "float32")
        cross(ab, ac, nabc)
        cross(ab, ad, nabd)
        cross(ac, ad, nacd)
        cross(bc, bd, nbcd)
        for k in T.serial(3):
            negabc[k] = -nabc[k]
            negabd[k] = -nabd[k]
            negacd[k] = -nacd[k]
            negbcd[k] = -nbcd[k]
            negab[k] = -ab[k]
        distances = T.alloc_local((6, 2), "float32")
        edge_dist = T.alloc_local((2,), "float32")
        valid = T.alloc_local((1,), "bool")
        done = T.alloc_local((1,), "bool")
        done[0] = False
        check_edge(
            T.uint32(0),
            a,
            nabc,
            nabd,
            ap,
            ab,
            ap_ab,
            bp_ab,
            edge_dist,
            Values,
            Flags,
            valid,
        )
        for k in T.serial(2):
            distances[0, k] = edge_dist[k]
        done[0] = valid[0]
        if not done[0]:
            check_edge(
                T.uint32(1),
                a,
                nacd,
                negabc,
                ap,
                ac,
                ap_ac,
                cp_ac,
                edge_dist,
                Values,
                Flags,
                valid,
            )
            for k in T.serial(2):
                distances[1, k] = edge_dist[k]
            done[0] = valid[0]
        if not done[0]:
            check_edge(
                T.uint32(2),
                a,
                negabd,
                negacd,
                ap,
                ad,
                ap_ad,
                dp_ad,
                edge_dist,
                Values,
                Flags,
                valid,
            )
            for k in T.serial(2):
                distances[2, k] = edge_dist[k]
            done[0] = valid[0]
        if not done[0]:
            check_edge(
                T.uint32(3),
                b,
                nabc,
                nbcd,
                bp,
                bc,
                bp_bc,
                cp_bc,
                edge_dist,
                Values,
                Flags,
                valid,
            )
            for k in T.serial(2):
                distances[3, k] = edge_dist[k]
            done[0] = valid[0]
        if not done[0]:
            check_edge(
                T.uint32(4),
                b,
                negbcd,
                nabd,
                bp,
                bd,
                bp_bd,
                dp_bd,
                edge_dist,
                Values,
                Flags,
                valid,
            )
            for k in T.serial(2):
                distances[4, k] = edge_dist[k]
            done[0] = valid[0]
        if not done[0]:
            check_edge(
                T.uint32(5),
                c,
                nacd,
                nbcd,
                cp,
                cd,
                cp_cd,
                dp_cd,
                edge_dist,
                Values,
                Flags,
                valid,
            )
            for k in T.serial(2):
                distances[5, k] = edge_dist[k]
            done[0] = valid[0]
        if not done[0]:
            check_face(
                T.uint32(0),
                a,
                b,
                c,
                ap,
                bp,
                cp,
                ab,
                ac,
                ad,
                distances[0, 0],
                distances[3, 0],
                distances[1, 1],
                Values,
                Flags,
                valid,
            )
            done[0] = valid[0]
        if not done[0]:
            check_face(
                T.uint32(1),
                a,
                b,
                d,
                ap,
                bp,
                dp,
                ab,
                ad,
                ac,
                distances[2, 0],
                distances[0, 1],
                distances[4, 1],
                Values,
                Flags,
                valid,
            )
            done[0] = valid[0]
        if not done[0]:
            check_face(
                T.uint32(2),
                a,
                c,
                d,
                ap,
                cp,
                dp,
                ac,
                ad,
                ab,
                distances[1, 0],
                distances[5, 0],
                distances[2, 1],
                Values,
                Flags,
                valid,
            )
            done[0] = valid[0]
        if not done[0]:
            check_face(
                T.uint32(3),
                b,
                c,
                d,
                bp,
                cp,
                dp,
                bc,
                bd,
                negab,
                distances[3, 1],
                distances[5, 1],
                distances[4, 0],
                Values,
                Flags,
                valid,
            )
            done[0] = valid[0]
        if not done[0]:
            solid(pt, Values, Flags)
