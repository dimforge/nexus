"""Independent compact contact geometry and semantic comparison of full scenes."""

import numpy as np
import scene_numeric
from twist_colored_validation import check_bounds, validate_tile


def tile_words(l, slot):
    base = slot // 32 * l["tile_words"] + slot % 32 * 4
    for off in l["header"].values():
        yield from range(base + off, base + off + 4)
    for field, off in l["twist"].items():
        for point in range(4 if field == "arm_a" else 1):
            start = base + off + point * l["twist_arm_words"]
            yield from range(start, start + 4)
    start = (
        slot // 32 * l["tile_words"]
        + l["warmstart"]
        + slot % 32 * 2 * l["velocity_words"]
    )
    yield from range(start, start + 2 * l["velocity_words"])


def geometry(l, raw, base, count):
    """Reconstruct masses, central response and angular radius from geometry in f64."""
    h, t = l["header"], l["twist"]
    v = raw.view(np.float32)
    vec = lambda off: v[base + off : base + off + 3].astype(float)
    normal, tangent = vec(h["normal"]), vec(h["tangent"])
    axes = [normal, tangent, np.cross(normal, tangent)]
    np.testing.assert_allclose(
        np.array(axes) @ np.array(axes).T, np.eye(3), atol=3e-6, rtol=0
    )
    mass = vec(h["body_a"]) + vec(h["body_b"])
    if np.any(mass < 0):
        raise AssertionError("Negative inverse twist mass")
    inertia = []
    for side in ("a", "b"):
        diag, off = vec(h["diag_" + side]), vec(h["off_" + side])
        matrix = np.diag(diag)
        matrix[0, 1] = matrix[1, 0] = off[0]
        matrix[0, 2] = matrix[2, 0] = off[1]
        matrix[1, 2] = matrix[2, 1] = off[2]
        if np.min(np.linalg.eigvalsh(matrix)) < -2e-6:
            raise AssertionError("Nonpositive twist inertia")
        inertia.append(matrix)
    arms = [vec(t["arm_a"] + k * l["twist_arm_words"]) for k in range(count)]
    center = np.mean(arms, axis=0)
    offset = vec(t["offset_b"])
    actual, expected = list(vec(t["center_a"])), list(center)
    for k, ra in enumerate(arms):
        ca, cb = np.cross(ra, normal), np.cross(ra + offset, normal)
        response = (
            np.dot(normal * mass, normal) + ca @ inertia[0] @ ca + cb @ inertia[1] @ cb
        )
        expected.extend([1 / response if response else 0, np.linalg.norm(ra - center)])
        actual.extend(
            [
                v[base + t["arm_a"] + k * l["twist_arm_words"] + 3],
                v[base + t["radius"] + k],
            ]
        )
    ca = [np.cross(center, axis) for axis in axes[1:]]
    cb = [np.cross(center + offset, axis) for axis in axes[1:]]
    for j, axis in enumerate(axes[1:]):
        expected.append(
            np.dot(axis * mass, axis)
            + ca[j] @ inertia[0] @ ca[j]
            + cb[j] @ inertia[1] @ cb[j]
        )
    expected.append(2 * (ca[0] @ inertia[0] @ ca[1] + cb[0] @ inertia[1] @ cb[1]))
    angular = normal @ (inertia[0] + inertia[1]) @ normal
    expected.append(1 / angular if abs(angular) > 1e-20 else 0)
    actual.extend(
        [
            v[base + t["offset_b"] + 3],
            v[base + t["center_a"] + 3],
            v[base + t["friction"] + 2],
            v[base + h["diag_b"] + 3],
        ]
    )
    np.testing.assert_allclose(
        actual,
        expected,
        atol=2e-6,
        rtol=2e-5,
        err_msg="Independent twist effective response",
    )


def row(l, c, result, cid, contract):
    h, t, lf = l["header"], l["twist"], l["link"]
    slot = int(result["Indices"][cid])
    base = slot // 32 * l["tile_words"] + slot % 32 * 4
    raw = result["Tiles"]
    count = int(raw[base + h["normal"] + 3])
    if not 1 <= count <= 4:
        raise AssertionError("Invalid scene twist point count")
    validate_tile(l, result, base, count)
    geometry(l, raw, base, count)
    check_bounds(l, raw, base, count, True, contract)
    words, floats, labels = [], [], []

    def field(off, width, mask, label):
        words.extend(range(base + off, base + off + width))
        floats.extend(mask)
        labels.extend([label] * width)

    for key, off in h.items():
        field(
            off, 4, [True] * 3 + [key in ("tangent", "diag_b", "off_b")], "TileHeader"
        )
    for key, off in t.items():
        if key == "arm_a":
            for k in range(count):
                field(off + k * l["twist_arm_words"], 4, [True] * 4, "TileGeometry")
        elif key in ("normal_impulse", "dist", "normal_vel", "radius"):
            field(
                off,
                4,
                [True] * count + [False] * (4 - count),
                "TileImpulse" if key == "normal_impulse" else "TileGeometry",
            )
            np.testing.assert_array_equal(
                raw[base + off + count : base + off + 4],
                0,
                err_msg="Inactive twist vector",
            )
        elif key == "friction":
            field(off, 4, [True] * 3 + [False], "TileImpulse")
            if raw[base + off + 3] != 0:
                raise AssertionError("Twist friction padding")
        else:
            field(off, 4, [True] * 4, "TileGeometry")
    cache = (
        slot // 32 * l["tile_words"]
        + l["warmstart"]
        + slot % 32 * 2 * l["velocity_words"]
    )
    for side in range(2):
        mask = np.zeros(l["velocity_words"], bool)
        for key in ("velocity_linear", "velocity_angular"):
            mask[l[key] : l[key] + 3] = True
        field(
            cache - base + side * l["velocity_words"],
            l["velocity_words"],
            mask,
            "TileCache",
        )
        if (
            raw[cache + side * l["velocity_words"] + l["velocity_padding1"]]
            != raw[base + h["body_a" if side == 0 else "body_b"] + 3]
        ):
            raise AssertionError("Twist scene cache body tag")
    values = raw[words].copy()
    if c.get("hub_pool"):
        for side, tag in (("a", "diag_a"), ("b", "off_a")):
            values[words.index(base + h[tag] + 3)] = result["Links"][
                cid * l["link_words"] + lf["solver_body_" + side]
            ]
    return values, np.array(floats), np.array(labels)


def check_constraints(l, c, actual, expected, contract, count):
    maximum = 0.0
    for cid in range(count):
        if not actual["Links"][cid * l["link_words"] + l["link"]["len"]]:
            continue
        a, mask, labels = row(l, c, actual, cid, contract)
        e, emask, elabels = row(l, c, expected, cid, contract)
        np.testing.assert_array_equal(mask, emask)
        np.testing.assert_array_equal(labels, elabels)
        np.testing.assert_array_equal(
            a[~mask], e[~mask], err_msg="Exact scene twist IDs/cache tags"
        )
        af, ef = a.view(np.float32), e.view(np.float32)
        if not np.isfinite(af[mask]).all() or not np.isfinite(ef[mask]).all():
            raise AssertionError("Nonfinite scene twist constraint")
        for label in np.unique(labels):
            chosen = mask & (labels == label)
            default = (
                contract["cached_warmstart_atol"]
                if label == "TileCache" and c["frames"] > 1
                else contract["atol"]
            )
            scene_numeric.check(c, contract, label, af[chosen], ef[chosen], default)
        maximum = max(maximum, float(np.max(np.abs(af[mask] - ef[mask]))))
    return {"Tiles": maximum}
