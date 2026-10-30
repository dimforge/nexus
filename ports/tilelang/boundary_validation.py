"""Scoped CFM branch equivalence; complete per-manifold outputs stay coupled."""

import numpy as np
from boundary_interval import mask as interval_mask
from replay import (
    BUFFERS as BASE,
)
from replay import (
    check_outputs as base_check,
)
from replay import (
    validate_case as base_validate,
)

BUFFERS = (
    *BASE,
    "SoftTiles",
    "HardTiles",
    "SoftVelocities",
    "HardVelocities",
    "Uncertain",
)
MAX_INTERVAL_WIDTH = 1e-5


def variants(b):
    return {
        kind: {
            **{n: b[n] for n in BASE},
            "Tiles": b[kind + "Tiles"],
            "Velocities": b[kind + "Velocities"],
        }
        for kind in ("Soft", "Hard")
    }


def validate_case(l, c, b):
    sizes = {n: c["buffer_words"][n] for n in BASE}
    for kind in ("Soft", "Hard"):
        for n in ("Tiles", "Velocities"):
            sizes[kind + n] = sizes[n]
    sizes["Uncertain"] = max(c["count"], 1)
    if (
        c.get("policy") != "single-ambiguous-cfm-v1"
        or c.get("kind")
        or not 1 <= c["iterations"] <= 3
        or set(b) != set(sizes)
        or c["buffer_words"] != sizes
        or any(v.dtype != np.uint32 or v.shape != (sizes[n],) for n, v in b.items())
    ):
        raise ValueError("Invalid boundary storage or iteration contract")
    base_validate(l, c, {n: b[n] for n in BASE})
    poses = b["Poses"].view(np.float32).reshape(-1, l["pose_words"])
    tf = b["Tiles"].view(np.float32)
    h = l["header"]
    for i in range(c["count"]):
        base, off = (i // 32) * l["tile_words"], (i % 32) * 4
        normal = tf[base + h["normal"] + off : base + h["normal"] + off + 3].astype(
            float
        )
        if not np.isfinite(normal).all() or not np.isclose(
            np.dot(normal, normal), 1, atol=2e-6, rtol=0
        ):
            raise ValueError("Invalid boundary normal")
        for key in ("body_a", "body_b"):
            pose = poses[int(b["Tiles"][base + h[key] + off + 3])]
            q = pose[l["pose_rotation"] : l["pose_rotation"] + 4].astype(float)
            t = pose[l["pose_translation"] : l["pose_translation"] + 3]
            if (
                not np.isfinite(q).all()
                or not np.isfinite(t).all()
                or not np.isclose(np.dot(q, q), 1, atol=2e-6, rtol=0)
            ):
                raise ValueError("Invalid boundary pose")
    uncertain, width, maximum = interval_mask(l, c, b)
    if not np.isfinite(width) or width > MAX_INTERVAL_WIDTH or maximum > 1:
        raise ValueError(
            "Boundary classification outside bounded single-ambiguous-point scope"
        )
    expected = np.full(max(c["count"], 1), 0xDEADBEEF, np.uint32)
    expected[: c["count"]] = uncertain
    np.testing.assert_array_equal(
        b["Uncertain"], expected, err_msg="Independent boundary mask"
    )
    return uncertain, width


def indices(l, c, b, i):
    h, p = l["header"], l["point"]
    base, off = (i // 32) * l["tile_words"], (i % 32) * 4
    count = int(b["Tiles"][base + h["normal"] + off + 3])
    result = {"Tiles": [], "Velocities": []}
    for key in ("diag_a", "off_a"):
        slot = int(b["Tiles"][base + h[key] + off + 3]) * l["velocity_words"]
        for field in ("velocity_linear", "velocity_angular"):
            result["Velocities"] += list(range(slot + l[field], slot + l[field] + 3))
    for j in range(count):
        point = base + l["points"] + j * l["point_words"] + off
        result["Tiles"].append(point + p["arm_a"] + 3)
        if c["solve_friction"]:
            result["Tiles"] += list(
                range(point + p["friction"], point + p["friction"] + 2)
            )
    return result


def vector(data, ids):
    return np.concatenate([data[n].view(np.float32)[idx] for n, idx in ids.items()])


def classify(l, c, b, a, e, k):
    uncertain, width = validate_case(l, c, b)
    options = variants(b)
    base = {n: b[n] for n in BASE}
    for r in (a, e):
        if set(r) != set(b) or any(
            r[n].dtype != np.uint32 or r[n].shape != b[n].shape for n in b
        ):
            raise ValueError("Invalid boundary outputs")
        for n in set(BUFFERS) - set(BASE):
            np.testing.assert_array_equal(r[n], b[n], err_msg="Boundary readonly " + n)
        base_check(l, c, base, {n: r[n] for n in BASE}, {n: r[n] for n in BASE}, k)
    for option in options.values():
        base_check(l, c, base, option, option, k)
    matched = {"Soft": 0, "Hard": 0}
    max_error = 0
    for i in range(c["count"]):
        ids = indices(l, c, b, i)
        actual, native = vector(a, ids), vector(e, ids)
        candidates = {name: vector(v, ids) for name, v in options.items()}
        if not uncertain[i] or not c["use_bias"]:
            for v in candidates.values():
                np.testing.assert_array_equal(
                    v, native, err_msg="Nonambiguous native branch"
                )
        for v in (native, actual):
            matches = [
                (
                    name,
                    float(
                        np.max(
                            np.abs(v.astype(float) - candidate.astype(float)), initial=0
                        )
                    ),
                )
                for name, candidate in candidates.items()
                if np.allclose(v, candidate, atol=k["atol"], rtol=k["rtol"])
            ]
            if not matches:
                raise AssertionError(
                    f"Contact {i} does not match a complete allowed CFM branch"
                )
            if v is actual:
                name, error = min(matches, key=lambda pair: pair[1])
                matched[name] += 1
                max_error = max(max_error, error)
    return {
        "uncertain_contacts": int(np.count_nonzero(uncertain)),
        "max_distance_interval_width": width,
        "matched_branches": matched,
        "maximum_branch_residual": max_error,
    }


def check_outputs(l, c, b, a, e, k):
    classify(l, c, b, a, e, k)
    masks = base_validate(l, c, {n: b[n] for n in BASE})
    return {
        n: float(
            np.max(
                np.abs(
                    a[n].view(np.float32)[masks[n]].astype(float)
                    - e[n].view(np.float32)[masks[n]].astype(float)
                ),
                initial=0,
            )
        )
        if n in masks
        else 0.0
        for n in BUFFERS
    }
