"""Native twist ABI, write ownership, exact guards and friction-bound checks."""

BUFFERS = ("Tiles", "Velocities", "Poses", "Params")
OPERATIONS = ("twist_effective_mass", "twist_warmstart", "twist_solve", "twist_scale")


def validate_case(l, c, arrays):
    import numpy as np
    from shaders.dynamics.twist_tiles import validate_layout

    validate_layout(l)
    if c["kind"] not in OPERATIONS:
        raise ValueError("Unknown twist operation")
    for key in ("count", "bodies", "solver_storage", "tiles", "iterations"):
        if type(c[key]) is not int or not 0 <= c[key] < 2**31:
            raise ValueError("Invalid twist dimensions")
    if (
        min(c[k] for k in ("bodies", "solver_storage", "tiles", "iterations")) < 1
        or c["count"] > c["tiles"] * 32
    ):
        raise ValueError("Invalid twist storage")
    if type(c["use_bias"]) is not bool or type(c["solve_friction"]) is not bool:
        raise ValueError("Invalid twist switches")
    if not np.isfinite(c["scale"]) or not 0 <= c["scale"] <= 1:
        raise ValueError("Invalid impulse scale")
    sizes = {
        "Tiles": c["tiles"] * l["tile_words"],
        "Velocities": c["solver_storage"] * l["velocity_words"],
        "Poses": c["bodies"] * l["pose_words"],
        "Params": 6,
    }
    if set(arrays) != set(sizes) or any(
        a.dtype != np.uint32 or a.shape != (sizes[n],) or a.size >= 2**31
        for n, a in arrays.items()
    ):
        raise ValueError("Invalid twist buffer shape")
    if not np.isfinite(arrays["Params"].view(np.float32)).all():
        raise ValueError("Nonfinite coefficients")
    h, t = l["header"], l["twist"]
    raw, vals = arrays["Tiles"], arrays["Tiles"].view(np.float32)
    used = set()
    for i in range(c["count"]):
        b = i // 32 * l["tile_words"] + i % 32 * 4
        count = int(raw[b + h["normal"] + 3])
        if count > 4:
            raise ValueError("Invalid point count")
        floats = []
        for offset in h.values():
            floats.extend(range(b + offset, b + offset + 3))
        floats.extend((b + h["tangent"] + 3, b + h["diag_b"] + 3, b + h["off_b"] + 3))
        for name in ("offset_b", "center_a", "friction", "frame_a", "frame_b"):
            floats.extend(range(b + t[name], b + t[name] + 4))
        for k in range(count):
            floats.extend(
                range(
                    b + t["arm_a"] + k * l["twist_arm_words"],
                    b + t["arm_a"] + k * l["twist_arm_words"] + 4,
                )
            )
            floats.extend(
                b + t[name] + k
                for name in ("normal_impulse", "dist", "normal_vel", "radius")
            )
        if not np.isfinite(vals[floats]).all():
            raise ValueError("Nonfinite twist fields")
        for side in ("a", "b"):
            body = int(raw[b + h[f"body_{side}"] + 3])
            slot = int(raw[b + h["diag_a" if side == "a" else "off_a"] + 3])
            if body >= c["bodies"] or slot >= c["solver_storage"]:
                raise ValueError("Invalid twist body or velocity slot")
            if slot in used:
                raise ValueError("Aliased twist velocity slots")
            used.add(slot)
            q = vals[b + t[f"frame_{side}"] : b + t[f"frame_{side}"] + 4]
            pose = arrays["Poses"].view(np.float32)[
                body * l["pose_words"] : (body + 1) * l["pose_words"]
            ]
            v = arrays["Velocities"].view(np.float32)[
                slot * l["velocity_words"] : (slot + 1) * l["velocity_words"]
            ]
            pq = pose[l["pose_rotation"] : l["pose_rotation"] + 4]
            if (
                not np.isfinite(pose[:7]).all()
                or abs(np.linalg.norm(pq) - 1) > 2e-6
                or abs(np.linalg.norm(q) - 1) > 2e-6
            ):
                raise ValueError("Invalid native pose/frame")
            if not np.isfinite(
                np.r_[
                    v[l["velocity_linear"] : l["velocity_linear"] + 3],
                    v[l["velocity_angular"] : l["velocity_angular"] + 3],
                ]
            ).all():
                raise ValueError("Nonfinite velocity")
        if abs(np.linalg.norm(vals[b + h["normal"] : b + h["normal"] + 3]) - 1) > 2e-6:
            raise ValueError("Invalid contact normal")


def output_masks(l, c, before):
    import numpy as np

    masks = {name: np.zeros(a.size, bool) for name, a in before.items()}
    h, t = l["header"], l["twist"]
    a, f = before["Tiles"], before["Tiles"].view(np.float32)
    op = c["kind"]
    for i in range(c["count"]):
        b = i // 32 * l["tile_words"] + i % 32 * 4
        count = int(a[b + h["normal"] + 3])
        if op == "twist_effective_mass" and count:
            for offset, width in (
                (t["center_a"], 4),
                (t["offset_b"] + 3, 1),
                (t["friction"] + 2, 1),
                (h["diag_b"] + 3, 1),
            ):
                masks["Tiles"][b + offset : b + offset + width] = True
            for k in range(count):
                masks["Tiles"][b + t["arm_a"] + k * l["twist_arm_words"] + 3] = True
                masks["Tiles"][b + t["radius"] + k] = True
            if count == 1:
                masks["Tiles"][b + h["off_b"] + 3] = True
        if op in ("twist_scale", "twist_solve"):
            masks["Tiles"][
                b + t["normal_impulse"] : b + t["normal_impulse"] + count
            ] = True
            if op == "twist_scale" or count and c["solve_friction"]:
                masks["Tiles"][b + t["friction"] : b + t["friction"] + 2] = True
                masks["Tiles"][b + h["off_b"] + 3] = True
        if count and op in ("twist_warmstart", "twist_solve"):
            for side in ("a", "b"):
                if any(
                    np.any(f[b + h[name] : b + h[name] + 3] != 0)
                    for name in (f"body_{side}", f"diag_{side}", f"off_{side}")
                ):
                    slot = (
                        int(a[b + h["diag_a" if side == "a" else "off_a"] + 3])
                        * l["velocity_words"]
                    )
                    for field in ("velocity_linear", "velocity_angular"):
                        masks["Velocities"][slot + l[field] : slot + l[field] + 3] = (
                            True
                        )
    return masks


def check_outputs(l, c, before, actual, expected, contract):
    import numpy as np

    validate_case(l, c, before)
    masks = output_masks(l, c, before)
    errors = {}
    for name in BUFFERS:
        a, e, mask = actual[name], expected[name], masks[name]
        if (
            a.dtype != np.uint32
            or e.dtype != np.uint32
            or a.shape != before[name].shape
            or e.shape != a.shape
        ):
            raise ValueError("Invalid twist output shape")
        np.testing.assert_array_equal(a[~mask], before[name][~mask])
        np.testing.assert_array_equal(e[~mask], before[name][~mask])
        af, ef = a.view(np.float32)[mask], e.view(np.float32)[mask]
        if not np.isfinite(af).all() or not np.isfinite(ef).all():
            raise AssertionError("Nonfinite twist result")
        np.testing.assert_allclose(af, ef, atol=contract["atol"], rtol=contract["rtol"])
        errors[name] = float(np.max(np.abs(af - ef))) if af.size else 0.0
    if c["kind"] == "twist_solve":
        h, t = l["header"], l["twist"]
        raw, vals = actual["Tiles"], actual["Tiles"].view(np.float32)
        for i in range(c["count"]):
            b = i // 32 * l["tile_words"] + i % 32 * 4
            count = int(raw[b + h["normal"] + 3])
            impulses = vals[b + t["normal_impulse"] : b + t["normal_impulse"] + count]
            if np.any(impulses < 0):
                raise AssertionError("Negative twist normal impulse")
            if count and c["solve_friction"]:
                friction = float(vals[b + h["tangent"] + 3])
                limit = float(np.sum(impulses.astype(float))) * friction
                twist_limit = (
                    float(
                        np.dot(
                            impulses.astype(float),
                            vals[b + t["radius"] : b + t["radius"] + count].astype(
                                float
                            ),
                        )
                    )
                    * friction
                )
                tangent = vals[b + t["friction"] : b + t["friction"] + 2]
                twist = float(vals[b + h["off_b"] + 3])
                for value, bound in (
                    (float(np.linalg.norm(tangent.astype(float))), limit),
                    (abs(twist), twist_limit),
                ):
                    if value > bound + contract["atol"] + contract["rtol"] * abs(bound):
                        raise AssertionError("Manifold friction bound exceeded")
                if count == 1 and twist != 0:
                    raise AssertionError("Single-point twist impulse")
    return errors
