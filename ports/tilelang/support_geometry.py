"""Independent f64 support optimality, containment and GJK witness checks."""

from itertools import combinations

import numpy as np
from geometry_validation import closest_on_hull


def rotate(q, v):
    return v + 2 * np.cross(q[:3], np.cross(q[:3], v) + q[3] * v)


def inverse(q, v):
    return rotate(np.r_[-q[:3], q[3]], v)


def support_value(l, shape, vertices, d):
    f = shape.view(np.float32).astype(np.float64)
    sf = l["shape"]
    aa = sf["a"]
    bb = sf["b"]
    cc = sf["c"]
    ty = int(shape[aa + 3])
    a = f[aa : aa + 3]
    b = f[bb : bb + 3]
    if ty == l["shape_types"]["ball"]:
        return a[0] * np.linalg.norm(d)
    if ty == l["shape_types"]["cuboid"]:
        return np.dot(a, np.abs(d))
    if ty == l["shape_types"]["capsule"]:
        return max(np.dot(a, d), np.dot(b, d)) + f[bb + 3] * np.linalg.norm(d)
    if ty == l["shape_types"]["cone"]:
        return max(a[0] * d[1], -a[0] * d[1] + a[1] * np.linalg.norm(d[[0, 2]]))
    if ty == l["shape_types"]["cylinder"]:
        return a[0] * abs(d[1]) + a[1] * np.linalg.norm(d[[0, 2]])
    if ty == l["shape_types"]["convex_poly"]:
        return np.max(vertices[int(shape[aa]) : int(shape[aa + 1])] @ d)
    if ty == l["shape_types"]["triangle"]:
        return np.max(np.array([a, b, f[cc : cc + 3]]) @ d)
    raise ValueError(ty)


def contains(l, shape, vertices, p, boundary=False):
    f = shape.view(np.float32).astype(np.float64)
    sf = l["shape"]
    aa = sf["a"]
    bb = sf["b"]
    cc = sf["c"]
    ty = int(shape[aa + 3])
    a = f[aa : aa + 3]
    b = f[bb : bb + 3]
    eps = 2e-6 * max(1.0, float(np.linalg.norm(p)), float(np.linalg.norm(a)))
    if ty == l["shape_types"]["ball"]:
        gap = np.linalg.norm(p) - a[0]
        return abs(gap) <= eps if boundary else gap <= eps
    if ty == l["shape_types"]["cuboid"]:
        gap = np.abs(p) - a
        return np.max(gap) <= eps and (not boundary or np.min(np.abs(gap)) <= eps)
    if ty == l["shape_types"]["capsule"]:
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1) if np.dot(ab, ab) else 0.0
        gap = np.linalg.norm(p - (a + t * ab)) - f[bb + 3]
        return abs(gap) <= eps if boundary else gap <= eps
    if ty == l["shape_types"]["cone"]:
        radial = np.linalg.norm(p[[0, 2]])
        allowed = a[1] * (a[0] - p[1]) / (2 * a[0])
        return (
            -a[0] - eps <= p[1] <= a[0] + eps
            and radial <= allowed + eps
            and (not boundary or min(abs(radial - allowed), abs(p[1] + a[0])) <= eps)
        )
    if ty == l["shape_types"]["cylinder"]:
        gaps = np.array([abs(p[1]) - a[0], np.linalg.norm(p[[0, 2]]) - a[1]])
        return gaps.max() <= eps and (not boundary or np.min(np.abs(gaps)) <= eps)
    hull = (
        vertices[int(shape[aa]) : int(shape[aa + 1])]
        if ty == l["shape_types"]["convex_poly"]
        else np.array([a, b, f[cc : cc + 3]])
    )
    closest = closest_on_hull(hull, p)
    # In-plane triangle points are boundary points of the 3D convex set.
    if np.linalg.norm(closest - p) > eps:
        return False
    if boundary and ty == l["shape_types"]["convex_poly"]:
        for ids in combinations(range(len(hull)), 3):
            va, vb, vc = hull[list(ids)]
            normal = np.cross(vb - va, vc - va)
            length = np.linalg.norm(normal)
            if length <= 1e-12:
                continue
            normal /= length
            side = (hull - va) @ normal
            if (np.min(side) >= -eps or np.max(side) <= eps) and abs(
                np.dot(p - va, normal)
            ) <= eps:
                return True
        return False

    return True


def check(l, c, b, a):
    n = c["count"]
    sw = l["shape_words"]
    pw = l["pose_words"]
    vw = l["vertex_words"]
    si = l["support_input"]
    so = l["support_output"]
    sf = l["simplex"]
    cw = l["cso_words"]
    shapes = b["Shapes"].reshape(-1, sw)
    poses = b["Poses"].view(np.float32).reshape(-1, pw).astype(np.float64)
    vertices = (
        b["Vertices"]
        .view(np.float32)
        .reshape(-1, vw)[:, l["vertex_vector"] : l["vertex_vector"] + 3]
        .astype(np.float64)
    )
    inputs = b["Inputs"].reshape(-1, l["support_input_words"])
    output = a["Output"].reshape(-1, l["support_output_words"])
    states = a["State"].reshape(-1, l["simplex_words"])
    for i in range(n):
        sa, sb = shapes[i * 2 : i * 2 + 2]
        q = poses[i, l["pose_rotation"] : l["pose_rotation"] + 4]
        shift = poses[i, l["pose_translation"] : l["pose_translation"] + 3]
        f = output[i].view(np.float32).astype(np.float64)
        direction = (
            inputs[i, si["direction"] : si["direction"] + 3]
            .view(np.float32)
            .astype(np.float64)
        )
        if c["kind"] == "gjk_closest":
            status = int(output[i, so["status"]])
            limit = float(
                inputs[i, si["max_distance"] : si["max_distance"] + 1].view(np.float32)[
                    0
                ]
            )
            axis = f[so["direction"] : so["direction"] + 3]
            if status == 0:
                assert np.array_equal(
                    output[i, so["a"] : so["status"]],
                    np.zeros(so["status"] - so["a"], np.uint32),
                ), ("intersection result", i)
                dim = int(states[i, sf["dim"]])
                points = (
                    states[i, sf["vertices"] : sf["vertices"] + 4 * cw]
                    .view(np.float32)
                    .reshape(4, 3, 3)[: dim + 1]
                    .astype(np.float64)
                )
                for p, aa, bb in points:
                    # A CSO vertex is the rounded f32 subtraction of its two
                    # witnesses. Check that construction exactly: a fixed f64
                    # residual threshold incorrectly rejects larger valid shapes.
                    np.testing.assert_array_equal(
                        p,
                        (aa - bb).astype(np.float32).astype(np.float64),
                        err_msg=f"CSO decomposition {i}",
                    )
                    assert contains(l, sa, vertices, aa, True), (
                        "simplex A surface",
                        i,
                        aa,
                    )
                    assert contains(l, sb, vertices, inverse(q, bb - shift), True), (
                        "simplex B surface",
                        i,
                        bb,
                    )
                closest = closest_on_hull(points[:, 0], np.zeros(3))
                assert np.linalg.norm(closest) <= 2e-6, (
                    "origin in simplex",
                    i,
                    closest,
                )
            else:
                np.testing.assert_array_equal(
                    states[i],
                    b["State"].reshape(-1, l["simplex_words"])[i],
                    err_msg="nonintersection preserves supplied simplex",
                )
                assert abs(np.linalg.norm(axis) - 1) < 2e-6, ("unit separating axis", i)
                minimum = -(
                    support_value(l, sa, vertices, axis)
                    + np.dot(-axis, shift)
                    + support_value(l, sb, vertices, inverse(q, -axis))
                )
                if status == 3:
                    assert minimum > limit - 2e-6, (
                        "separating lower bound",
                        i,
                        minimum,
                        limit,
                    )
                if status == 1:
                    aa, bb = f[so["a"] : so["a"] + 3], f[so["b"] : so["b"] + 3]
                    assert contains(l, sa, vertices, aa), ("closest A", i)
                    assert contains(l, sb, vertices, inverse(q, bb - shift)), (
                        "closest B",
                        i,
                    )
                    distance = np.linalg.norm(aa - bb)
                    assert distance >= minimum - 4e-6, (
                        "GJK distance bounds",
                        i,
                        distance,
                        minimum,
                    )
                    # Native affine-dependence rejection may return before its
                    # relative distance-gap convergence condition is met.
        elif c["kind"] == "cso_support":
            p, aa, bb = (
                f[so["a"] : so["a"] + 3],
                f[so["b"] : so["b"] + 3],
                f[so["direction"] : so["direction"] + 3],
            )
            assert np.max(np.abs(p - (aa - bb))) < 2e-6, ("cso", i)
            for s, point, axis in [
                (sa, aa, direction),
                (sb, inverse(q, bb - shift), inverse(q, -direction)),
            ]:
                assert contains(l, s, vertices, point, True), ("support surface", i)
                assert abs(
                    np.dot(point, axis) - support_value(l, s, vertices, axis)
                ) < 2e-6 * max(1, np.linalg.norm(axis)), ("support optimum", i)
        else:
            point = f[so["a"] : so["a"] + 3]
            axis = direction
            if c["kind"] == "support_world":
                point = inverse(q, point - shift)
                axis = inverse(q, direction)
            assert contains(l, sa, vertices, point, True), ("support surface", i)
            assert abs(
                np.dot(point, axis) - support_value(l, sa, vertices, axis)
            ) < 2e-6 * max(1, np.linalg.norm(axis)), ("support optimum", i)
