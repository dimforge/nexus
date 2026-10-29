"""Geometry checks for the nonunique normal at an axial interior contact.

These checks never generate runtime contacts. They independently validate a
rare symmetry family when float cancellation changes its selected direction.
"""

import numpy as np


def rotate(q, v):
    return (
        v * (q[3] * q[3] - np.dot(q[:3], q[:3]))
        + q[:3] * (2 * np.dot(v, q[:3]))
        + np.cross(q[:3], v) * (2 * q[3])
    )


def conjugate(q):
    result = q.copy()
    result[:3] *= -1
    return result


def symmetry_frame(layout, before, first, second):
    """Return the convex frame only when the ball center is on its interior axis."""
    shapes = before["Shapes"].reshape(-1, layout["shape_words"])
    sf = layout["shape"]
    types = layout["shape_types"]
    ty1, ty2 = (int(shapes[i, sf["a"] + 3]) for i in (first, second))
    if ty1 == types["ball"] and ty2 in (
        types["capsule"],
        types["cone"],
        types["cylinder"],
    ):
        convex, ball, flipped = second, first, True
    elif ty2 == types["ball"] and ty1 in (
        types["capsule"],
        types["cone"],
        types["cylinder"],
    ):
        convex, ball, flipped = first, second, False
    else:
        return None
    poses = (
        before["Poses"]
        .view(np.float32)
        .reshape(-1, layout["pose_words"])
        .astype(np.float64)
    )
    qr, tr = layout["pose_rotation"], layout["pose_translation"]
    qc, qb = poses[convex, qr : qr + 4], poses[ball, qr : qr + 4]
    tc, tb = poses[convex, tr : tr + 3], poses[ball, tr : tr + 3]
    inv = conjugate(qc)
    center = rotate(inv, -tc) + rotate(inv, tb)
    shape = shapes[convex].view(np.float32).astype(np.float64)
    a, b = shape[sf["a"] : sf["a"] + 3], shape[sf["b"] : sf["b"] + 3]
    ty = int(shapes[convex, sf["a"] + 3])
    radius_ball = float(shapes[ball].view(np.float32)[sf["a"]])
    scale = max(
        1.0,
        np.max(np.abs(center)),
        np.max(np.abs(a)),
        np.max(np.abs(b)) if ty == types["capsule"] else 0.0,
    )
    uncertainty = 32 * np.finfo(np.float32).eps * scale
    if ty == types["capsule"]:
        axis = b - a
        square = float(np.dot(axis, axis))
        if square == 0:
            return (
                None  # Coincident sphere case has a separately deterministic fallback.
            )
        u = float(np.dot(center - a, axis) / square)
        radial = center - (a + axis * u)
        if not (0.001 < u < 0.999 and np.linalg.norm(radial) <= uncertainty):
            return None
        convex_radius = float(shape[sf["b"] + 3])
    else:
        if not (
            -a[0] + uncertainty < center[1] < a[0] - uncertainty
            and np.linalg.norm(center[[0, 2]]) <= uncertainty
        ):
            return None
        convex_radius = a[1]
    if convex_radius <= uncertainty:
        return None
    return dict(
        center=center,
        shape=shape,
        ty=ty,
        qc=qc,
        qb=qb,
        flipped=flipped,
        ball_radius=radius_ball,
        scale=scale,
        uncertainty=uncertainty,
    )


def validate_symmetry_contact(layout, frame, row, atol, rtol):
    """Require an actual boundary point, a unit normal and the closest distance."""
    cf, mf, pf = layout["contact"], layout["manifold"], layout["manifold_point"]
    m = cf["contact"]
    p = m + mf["points_a"]
    f = row.view(np.float32).astype(np.float64)
    normal = f[m + mf["normal_a"] : m + mf["normal_a"] + 3]
    point = f[p + pf["pt"] : p + pf["pt"] + 3]
    distance = float(f[p + pf["dist"]])
    tolerance = atol + rtol * frame["scale"]
    if not np.isfinite(np.r_[normal, point, distance]).all():
        raise AssertionError("Nonfinite symmetry contact")
    if abs(np.linalg.norm(normal) - 1) > max(8e-6, atol + rtol):
        raise AssertionError("Nonunit symmetry normal")
    if frame["flipped"]:
        if not np.allclose(point, normal * frame["ball_radius"], atol=atol, rtol=rtol):
            raise AssertionError("Ball support point does not follow its normal")
        # normal is expressed in ball's local frame; reverse it into convex's.
        convex_normal = -rotate(conjugate(frame["qc"]), rotate(frame["qb"], normal))
        convex_point = frame["center"] - convex_normal * (
            distance + frame["ball_radius"]
        )
    else:
        convex_normal = normal
        convex_point = point
    delta = convex_point - frame["center"]
    length = float(np.linalg.norm(delta))
    if length <= 0 or not np.allclose(
        delta / length, convex_normal, atol=atol, rtol=rtol
    ):
        raise AssertionError("Symmetry normal is not the boundary direction")
    if abs(distance - (-length - frame["ball_radius"])) > tolerance:
        raise AssertionError("Incorrect symmetry penetration distance")
    sf = layout["shape"]
    types = layout["shape_types"]
    shape = frame["shape"]
    a, b = shape[sf["a"] : sf["a"] + 3], shape[sf["b"] : sf["b"] + 3]
    if frame["ty"] == types["capsule"]:
        axis = b - a
        square = np.dot(axis, axis)
        u = np.clip(np.dot(convex_point - a, axis) / square, 0.0, 1.0)
        axis_point = a + u * axis
        radius = shape[sf["b"] + 3]
        surface_error = abs(np.linalg.norm(convex_point - axis_point) - radius)
        center_u = np.clip(np.dot(frame["center"] - a, axis) / square, 0.0, 1.0)
        nearest = radius - np.linalg.norm(frame["center"] - (a + center_u * axis))
    elif frame["ty"] == types["cylinder"]:
        height, radius = a[:2]
        radial = np.linalg.norm(convex_point[[0, 2]])
        surface_error = min(
            max(abs(radial - radius), max(abs(convex_point[1]) - height, 0)),
            max(abs(abs(convex_point[1]) - height), max(radial - radius, 0)),
        )
        nearest = min(
            height - abs(frame["center"][1]),
            radius - np.linalg.norm(frame["center"][[0, 2]]),
        )
    else:
        height, radius = a[:2]
        radial = np.linalg.norm(convex_point[[0, 2]])
        target_radius = radius * (height - convex_point[1]) / (2 * height)
        surface_error = min(
            max(abs(radial - target_radius), max(abs(convex_point[1]) - height, 0)),
            max(abs(convex_point[1] + height), max(radial - radius, 0)),
        )
        # Distance to the generatrix or base disk, computed in the radial plane.
        center = np.array([np.linalg.norm(frame["center"][[0, 2]]), frame["center"][1]])
        apex = np.array([0.0, height])
        axis = np.array([radius, -2 * height])
        u = np.clip(np.dot(center - apex, axis) / np.dot(axis, axis), 0.0, 1.0)
        nearest = min(np.linalg.norm(center - (apex + axis * u)), center[1] + height)
    if surface_error > tolerance:
        raise AssertionError(
            f"Symmetry point is inside/off the surface: {surface_error}"
        )
    if abs(length - nearest) > tolerance:
        raise AssertionError("Symmetry contact is not a closest boundary point")
