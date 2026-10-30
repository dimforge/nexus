"""Explicit per-case, per-field numerical budgets for contact-sensitive trajectories."""

import numpy as np


def policy_for(c, k):
    policy = k.get("numerical_cases", {}).get(c["name"], {})
    if policy:
        for key, value in policy["configuration"].items():
            if c.get(key) != value:
                raise ValueError("Numerical policy configuration mismatch")
    return policy


def tolerances(c, k, field, default_atol=None):
    policy = policy_for(c, k)
    if policy:
        limits = policy["fields"].get(field)
        if limits is not None:
            return limits["atol"], limits["rtol"]
    return k["atol"] if default_atol is None else default_atol, k["rtol"]


def check(c, k, field, a, e, default_atol=None):
    atol, rtol = tolerances(c, k, field, default_atol)
    np.testing.assert_allclose(a, e, atol=atol, rtol=rtol, err_msg=field)


def field_contract(c, k, field):
    atol, rtol = tolerances(c, k, field)
    return dict(k, atol=atol, rtol=rtol)


def constraint_geometry(l, c, r):
    """Recompute the effective response from the stored geometry in f64."""
    h, p = l["header"], l["point"]
    lf = l["link"]
    tiles = r["Tiles"]
    values = tiles.view(np.float32).astype(float)
    for cid in range(int(r["Plan"][l["plan"]["bound"]])):
        if not r["Links"][cid * l["link_words"] + lf["len"]]:
            continue
        slot = int(r["Indices"][cid])
        base = slot // 32 * l["tile_words"] + slot % 32 * 4
        vector = lambda name, base=base: values[base + h[name] : base + h[name] + 3]
        normal, tangent = vector("normal"), vector("tangent")
        axes = [normal, tangent, np.cross(normal, tangent)]
        np.testing.assert_allclose(
            np.array(axes) @ np.array(axes).T,
            np.eye(3),
            atol=3e-6,
            rtol=0,
            err_msg="Orthonormal contact frame",
        )
        mass = vector("body_a") + vector("body_b")
        inertia = []
        if np.any(mass < 0):
            raise AssertionError("Negative inverse contact mass")
        for side in ("a", "b"):
            diag, off = vector("diag_" + side), vector("off_" + side)
            matrix = np.diag(diag)
            matrix[0, 1] = matrix[1, 0] = off[0]
            matrix[0, 2] = matrix[2, 0] = off[1]
            matrix[1, 2] = matrix[2, 1] = off[2]
            if np.min(np.linalg.eigvalsh(matrix)) < -2e-6:
                raise AssertionError("Nonpositive contact inertia")
            inertia.append(matrix)
        length = int(tiles[base + h["normal"] + 3])
        for point in range(length):
            start = base + l["points"] + point * l["point_words"]
            ra = values[start + p["arm_a"] : start + p["arm_a"] + 3]
            rb = values[start + p["arm_b"] : start + p["arm_b"] + 3]
            ca = np.array([np.cross(ra, axis) for axis in axes])
            cb = np.array([np.cross(rb, axis) for axis in axes])
            response = np.array(
                [
                    np.dot(axis * mass, axis)
                    + ca[j] @ inertia[0] @ ca[j]
                    + cb[j] @ inertia[1] @ cb[j]
                    for j, axis in enumerate(axes)
                ]
            )
            expected = np.r_[
                1 / response[0] if response[0] else 0,
                response[1:],
                2 * (ca[1] @ inertia[0] @ ca[2] + cb[1] @ inertia[1] @ cb[2]),
            ]
            actual = values[
                [
                    start + p["arm_b"] + 3,
                    start + p["friction"] + 2,
                    start + p["friction"] + 3,
                    base + l["coupling"] + point,
                ]
            ]
            np.testing.assert_allclose(
                actual,
                expected,
                atol=2e-6,
                rtol=2e-5,
                err_msg="Independent contact effective response",
            )


def constraint_response(l, c, a, e, k):
    """Compare the tangent response as a matrix, including near-zero coupling."""
    h, p = l["header"], l["point"]
    lf = l["link"]
    policy = (
        k.get("numerical_cases", {})
        .get(c["name"], {})
        .get("fields", {})
        .get("TileResponse")
    )
    for cid in range(int(a["Plan"][l["plan"]["bound"]])):
        if not a["Links"][cid * l["link_words"] + lf["len"]]:
            continue
        rows = []
        for r in (a, e):
            slot = int(r["Indices"][cid])
            base = slot // 32 * l["tile_words"] + slot % 32 * 4
            values = r["Tiles"].view(np.float32).astype(float)
            length = int(r["Tiles"][base + h["normal"] + 3])
            matrices = []
            for point in range(length):
                off = base + l["points"] + point * l["point_words"] + p["friction"]
                cross = values[base + l["coupling"] + point] * 0.5
                matrices.append(
                    np.array([[values[off + 2], cross], [cross, values[off + 3]]])
                )
            rows.append(matrices)
        if len(rows[0]) != len(rows[1]):
            raise AssertionError("Tangent response point count")
        for actual, expected in zip(*rows):
            if not policy:
                check(
                    c,
                    k,
                    "TileResponse",
                    np.array([actual[0, 0], actual[1, 1], 2 * actual[0, 1]]),
                    np.array([expected[0, 0], expected[1, 1], 2 * expected[0, 1]]),
                )
            else:
                atol, rtol = tolerances(c, k, "TileResponse")
                error = np.linalg.norm(actual - expected)
                scale = max(np.linalg.norm(actual), np.linalg.norm(expected))
                if not np.isfinite(error) or error > atol + rtol * scale:
                    raise AssertionError(
                        f"Tangent response matrix error {error} exceeds {atol + rtol * scale}"
                    )


def impulse_response(l, r, cid):
    """Independent f64 force/torque reduction and inverse-mass response."""
    h, p = l["header"], l["point"]
    slot = int(r["Indices"][cid])
    base = slot // 32 * l["tile_words"] + slot % 32 * 4
    tile = r["Tiles"]
    v = tile.view(np.float32).astype(float)
    vector = lambda off: v[off : off + 3]
    normal, tangent = vector(base + h["normal"]), vector(base + h["tangent"])
    bitangent = np.cross(normal, tangent)
    force = np.zeros(3)
    torques = np.zeros((2, 3))
    length = int(tile[base + h["normal"] + 3])
    if not 1 <= length <= 4:
        raise AssertionError("Invalid impulse response point count")
    for i in range(length):
        off = base + l["points"] + i * l["point_words"]
        impulse = (
            tangent * v[off + p["friction"]] + bitangent * v[off + p["friction"] + 1]
        )
        impulse += normal * v[off + p["arm_a"] + 3]
        force += impulse
        for j, side in enumerate(("a", "b")):
            torques[j] += np.cross(vector(off + p["arm_" + side]), impulse)
    result = np.zeros((2, 6))
    for j, side in enumerate(("a", "b")):
        diag, off = vector(base + h["diag_" + side]), vector(base + h["off_" + side])
        inertia = np.diag(diag)
        inertia[0, 1] = inertia[1, 0] = off[0]
        inertia[0, 2] = inertia[2, 0] = off[1]
        inertia[1, 2] = inertia[2, 1] = off[2]
        sign = 1 if j == 0 else -1
        result[j, :3] = sign * vector(base + h["body_" + side]) * force
        result[j, 3:] = sign * (inertia @ torques[j])
    return result


def check_friction_response(l, c, a, e, k, cid):
    """Compare complete body responses while retaining each normal impulse check."""
    mode = policy_for(c, k).get("friction_comparison")
    if mode is None:
        return False
    if mode != "body_velocity_delta":
        raise ValueError("Unknown friction comparison")
    # These comparisons deliberately retain the global strict tolerance; neither
    # the trajectory nor stored-cache allowances may relax the response check.
    np.testing.assert_allclose(
        impulse_response(l, a, cid),
        impulse_response(l, e, cid),
        atol=k["atol"],
        rtol=k["rtol"],
        err_msg="Contact body response",
    )
    for r in (a, e):
        slot = int(r["Indices"][cid])
        start = (
            slot // 32 * l["tile_words"]
            + l["warmstart"]
            + slot % 32 * 2 * l["velocity_words"]
        )
        cached = (
            r["Tiles"]
            .view(np.float32)[start : start + 2 * l["velocity_words"]]
            .reshape(2, l["velocity_words"])
        )
        cached = np.c_[
            cached[:, l["velocity_linear"] : l["velocity_linear"] + 3],
            cached[:, l["velocity_angular"] : l["velocity_angular"] + 3],
        ]
        np.testing.assert_allclose(
            cached,
            impulse_response(l, r, cid),
            atol=k["atol"],
            rtol=k["rtol"],
            err_msg="Independent cached impulse response",
        )
    return True
