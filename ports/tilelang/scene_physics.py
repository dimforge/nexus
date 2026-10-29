"""Independent physical checks for the explicit awake ball/box scene fixtures."""


def state(layout, case, arrays):
    import numpy as np

    n = case["body_count"]
    pose = arrays["Poses"].view(np.float32).reshape(-1, layout["pose_words"])[:n]
    v = arrays["Velocities"].view(np.float32).reshape(-1, layout["velocity_words"])[:n]
    position = pose[
        :, layout["pose_translation"] : layout["pose_translation"] + 3
    ].astype(np.float64)
    rotation = pose[:, layout["pose_rotation"] : layout["pose_rotation"] + 4].astype(
        np.float64
    )
    linear = v[:, layout["velocity_linear"] : layout["velocity_linear"] + 3].astype(
        np.float64
    )
    angular = v[:, layout["velocity_angular"] : layout["velocity_angular"] + 3].astype(
        np.float64
    )
    if not all(np.isfinite(a).all() for a in (position, rotation, linear, angular)):
        raise AssertionError("Nonfinite scene body state")
    if np.any(np.abs(np.linalg.norm(rotation, axis=1) - 1) > 2e-6):
        raise AssertionError("Scene quaternion normalization")
    return position, rotation, linear, angular


def check_physics(layout, case, before, result, contract):
    import numpy as np

    p, q, v, w = state(layout, case, result)
    initial, _, iv, iw = state(layout, case, before)
    n, nb = case["body_count"], case["batches"]
    fixed = min(n, nb)
    # Compare words, including signed zero and padding, for every static body.
    for name, key in (("Poses", "pose_words"), ("Velocities", "velocity_words")):
        np.testing.assert_array_equal(
            result[name][: fixed * layout[key]],
            before[name][: fixed * layout[key]],
            err_msg="Static body changed",
        )
    dynamic = np.arange(fixed, n)
    if not len(dynamic):
        return {"penetration": 0.0, "energy": 0.0, "speed": 0.0, "spin": 0.0}
    radius = np.full(n, 0.5)
    if case["scene"] == "boxes":
        # Vertical support of the rotated cube, independently from native AABB code.
        x, y, z, s = q.T
        radius = 0.5 * (
            np.abs(2 * (x * y + s * z))
            + np.abs(1 - 2 * (x * x + z * z))
            + np.abs(2 * (y * z - s * x))
        )
    penetration = 0.0
    for i in dynamic:
        ground = i % nb
        # Ground boxes are finite. Bodies beyond their horizontal footprint are
        # allowed to fall; the wide fixture deliberately exercises that boundary.
        if abs(p[i, 0] - p[ground, 0]) < 19.5 and abs(p[i, 2] - p[ground, 2]) < 19.5:
            penetration = max(penetration, float(radius[i] - p[i, 1]))
        for j in dynamic:
            if j <= i or i % nb != j % nb:
                continue
            d = p[j] - p[i]
            if case["scene"] == "boxes":
                # These fixtures remain stacked in columns. This conservative
                # vertical-support test is not advertised as a general OBB oracle.
                if abs(d[0]) < 0.8 and abs(d[2]) < 0.8:
                    penetration = max(
                        penetration, float(radius[i] + radius[j] - abs(d[1]))
                    )
            else:
                penetration = max(penetration, float(1.0 - np.linalg.norm(d)))
    if penetration > contract["physics"]["max_penetration"]:
        raise AssertionError(f"Unbounded scene penetration: {penetration}")
    inertia = 1 / 6 if case["scene"] == "boxes" else 0.1
    gravity = float(-before["Gravity"].view(np.float32)[1])

    def energy(position, linear, angular):
        return float(
            np.sum(
                0.5 * np.sum(linear[dynamic] ** 2, axis=1)
                + 0.5 * inertia * np.sum(angular[dynamic] ** 2, axis=1)
                + gravity * position[dynamic, 1]
            )
        )

    total = energy(p, v, w)
    original = energy(initial, iv, iw)
    upper = (
        original * (1 + contract["physics"]["energy_relative_allowance"])
        + len(dynamic) * contract["physics"]["energy_per_body_allowance"]
    )
    if total > upper:
        raise AssertionError(f"Scene gained excessive energy: {total} > {upper}")
    speed = float(np.max(np.linalg.norm(v[dynamic], axis=1)))
    spin = float(np.max(np.linalg.norm(w[dynamic], axis=1)))
    if (
        case["frames"] == 120
        and case["scene"] in ("stack", "free", "boxes", "impact")
        and (speed > 0.01 or spin > 0.01)
    ):
        raise AssertionError("Resting scene did not settle")
    if case["scene"] == "impact" and case["frames"] == 1 and np.any(v[dynamic, 1] <= 0):
        raise AssertionError("Restitution scene did not rebound")
    if case["scene"] == "friction" and case["frames"] == 120:
        if np.any(v[dynamic, 0] <= 0) or np.any(v[dynamic, 0] >= iv[dynamic, 0]):
            raise AssertionError("Friction did not slow sliding bodies")
        if np.max(np.abs(v[dynamic, 0] + 0.5 * w[dynamic, 2])) > 0.01:
            raise AssertionError("Friction scene did not reach rolling contact")
    return {"penetration": penetration, "energy": total, "speed": speed, "spin": spin}
