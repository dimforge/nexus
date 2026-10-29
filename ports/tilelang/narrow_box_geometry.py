"""Independent box surface checks for scale-aware f32 contact comparison.

This is validation code only. It does not construct runtime contact manifolds.
"""

import numpy as np
from narrow_symmetry import conjugate, rotate


def box_frame(layout, before, first, second, contract):
    policy = contract.get("box_geometry_roundoff")
    if policy is None:
        return None
    shapes = before["Shapes"].reshape(-1, layout["shape_words"])
    a = layout["shape"]["a"]
    if any(
        shapes[i, a + 3] != layout["shape_types"]["cuboid"] for i in (first, second)
    ):
        return None
    poses = (
        before["Poses"]
        .view(np.float32)
        .reshape(-1, layout["pose_words"])
        .astype(np.float64)
    )
    q, t = layout["pose_rotation"], layout["pose_translation"]
    qa, qb = (poses[i, q : q + 4] for i in (first, second))
    delta = poses[first, t : t + 3] - poses[second, t : t + 3]
    extents = [
        shapes[i].view(np.float32)[a : a + 3].astype(np.float64)
        for i in (first, second)
    ]
    # Local sizes and relative displacement, not an arbitrary world-origin scale.
    scale = max(float(np.max(np.abs(delta))), *(float(np.max(e)) for e in extents))
    tolerance = max(
        contract["atol"],
        policy["epsilon_multiplier"] * np.finfo(np.float32).eps * scale,
    )
    return qa, qb, delta, extents, tolerance


def validate_box_contact(layout, frame, row, length, atol, rtol):
    qa, qb, delta, extents, tolerance = frame
    cf, mf, pf = layout["contact"], layout["manifold"], layout["manifold_point"]
    m = cf["contact"]
    f = row.view(np.float32).astype(np.float64)
    normal = f[m + mf["normal_a"] : m + mf["normal_a"] + 3]
    if abs(np.linalg.norm(normal) - 1) > atol + rtol:
        raise AssertionError("Nonunit box contact normal")
    for i in range(length):
        p = m + mf["points_a"] + i * layout["manifold_point_words"]
        point = f[p + pf["pt"] : p + pf["pt"] + 3]
        distance = f[p + pf["dist"]]
        other = rotate(conjugate(qb), rotate(qa, point + distance * normal) + delta)
        for local, half_extent in zip((point, other), extents):
            residual = np.abs(local) - half_extent
            if np.max(residual) > tolerance or np.min(np.abs(residual)) > tolerance:
                raise AssertionError("Box contact endpoint is off its surface")
