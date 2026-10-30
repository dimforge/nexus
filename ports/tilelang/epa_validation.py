"""EPA parity with independent convex witnesses and scoped nonuniqueness."""

import numpy as np
from geometry_validation import closest_on_hull
from support_geometry import contains, inverse, support_value
from support_validation import BUFFERS, validate_inputs
from support_validation import check_outputs as check_gjk
from support_validation import validate_case as validate_gjk

OPERATIONS = ("epa_closest", "gjk_epa")


def simplex_witness(l, row, sa, sb, vertices, q, shift, origin):
    sf = l["simplex"]
    dimension = int(row[sf["dim"]])
    if dimension > 3 or int(row[sf["previous_dim"]]) > 3:
        raise ValueError("Invalid EPA simplex dimension")
    if np.any(row[sf["previous_vertices"] : sf["previous_vertices"] + 4] > 3):
        raise ValueError("Invalid EPA simplex indices")
    if not np.isfinite(row[: sf["dim"]].view(np.float32)).all():
        raise ValueError("Nonfinite EPA simplex")
    points = (
        row[sf["vertices"] : sf["vertices"] + (dimension + 1) * l["cso_words"]]
        .view(np.float32)
        .reshape(-1, 3, 3)
        .astype(np.float64)
    )
    for point, aa, bb in points:
        if np.max(np.abs(point - (aa - bb))) > 3e-6:
            raise ValueError("Invalid EPA CSO decomposition")
        if not contains(l, sa, vertices, aa, True) or not contains(
            l, sb, vertices, inverse(q, bb - shift), True
        ):
            raise ValueError("EPA simplex witness is not on its shape surface")
    if origin and np.linalg.norm(closest_on_hull(points[:, 0], np.zeros(3))) > 2e-6:
        raise ValueError("EPA seed must contain the origin")


def validate_case(l, c, b):
    n, vertices, shapes, q, shift, _ = validate_inputs(l, c, b, OPERATIONS)
    rows = b["Inputs"].reshape(-1, l["support_input_words"])[:n]
    si = l["support_input"]
    if np.any(rows[:, si["exact"]] != 1) or np.any(rows[:, si["epa_active"]] > 1):
        raise ValueError("EPA requires an exact GJK query and a boolean active flag")
    if c["kind"] == "gjk_epa":
        validate_gjk(l, dict(c, kind="gjk_closest"), b)
    else:
        states = b["State"].reshape(-1, l["simplex_words"])
        for i in range(n):
            simplex_witness(
                l,
                states[i],
                shapes[2 * i],
                shapes[2 * i + 1],
                vertices,
                q[i].astype(float),
                shift[i].astype(float),
                bool(rows[i, si["epa_active"]]),
            )


def symmetric(l, shape, vertices):
    """Prove central symmetry about the shape origin, independent of EPA."""
    sf, types = l["shape"], l["shape_types"]
    ty = int(shape[sf["a"] + 3])
    if ty in (types["ball"], types["cuboid"], types["cylinder"]):
        return True
    f = shape.view(np.float32)
    if ty == types["capsule"]:
        return np.array_equal(f[sf["a"] : sf["a"] + 3], -f[sf["b"] : sf["b"] + 3])
    if ty == types["convex_poly"]:
        v = vertices[int(shape[sf["a"]]) : int(shape[sf["a"] + 1])]
        return all(np.any(np.all(v == -point, axis=1)) for point in v)
    return False


def penetration_witness(l, fields, dimension, sa, sb, vertices, q, shift, k):
    aa, bb, normal = fields[:3], fields[3:6], fields[6:9]
    # Preserve the explicit native zero-dimensional fallback. It does not return
    # the supplied support witnesses and is recorded as a reference limitation.
    if dimension == 0:
        np.testing.assert_array_equal(fields, [0, 0, 0, 0, 0, 0, 0, 1, 0])
        return 0.0, 0.0
    if not contains(l, sa, vertices, aa) or not contains(
        l, sb, vertices, inverse(q, bb - shift)
    ):
        raise AssertionError("EPA point outside its convex shape")
    np.testing.assert_allclose(
        np.linalg.norm(normal), 1, atol=2e-6, rtol=0, err_msg="EPA unit normal"
    )
    delta = aa - bb
    depth = float(np.dot(delta, normal))
    if depth < -2e-6 or np.linalg.norm(delta - normal * depth) > 2e-5:
        raise AssertionError("EPA penetration witnesses disagree with the normal")
    gap = float(
        support_value(l, sa, vertices, normal)
        + support_value(l, sb, vertices, inverse(q, -normal))
        - np.dot(shift, normal)
        - depth
    )
    if gap < -4e-6 or gap > k["max_support_gap"]:
        raise AssertionError("EPA penetration support bound exceeded")
    return depth, gap


def check_outputs(l, c, b, a, e, k):
    validate_case(l, c, b)
    for r in (a, e):
        if set(r) != set(b) or any(
            r[n].dtype != np.uint32 or r[n].shape != b[n].shape for n in b
        ):
            raise ValueError("Invalid returned EPA storage")
    si, so, sf = l["support_input"], l["support_output"], l["simplex"]
    iw, ow, sw = l["support_input_words"], l["support_output_words"], l["simplex_words"]
    inputs = b["Inputs"].reshape(-1, iw)
    active = inputs[: c["count"], si["epa_active"]] != 0
    states = [r["State"].reshape(-1, sw) for r in (a, e)]
    outputs = [r["Output"].reshape(-1, ow) for r in (a, e)]
    if c["kind"] == "gjk_epa":
        # Reconstruct the public GJK result that preceded EPA, then apply the
        # existing GJK checker, including exact nonintersection copied state.
        views = [{n: v.copy() for n, v in r.items()} for r in (a, e)]
        for view in views:
            rows = view["Output"].reshape(-1, ow)
            for i in np.flatnonzero(active):
                rows[i, so["a"] : so["status"] + 1] = 0
        check_gjk(l, dict(c, kind="gjk_closest"), b, *views, k)
    else:
        for r in (a, e):
            np.testing.assert_array_equal(
                r["State"], b["State"], err_msg="EPA readonly simplex"
            )
    masks = {n: np.zeros(v.size, bool) for n, v in b.items()}
    for i in range(c["count"]):
        masks["Output"][i * ow + so["a"] : i * ow + so["status"] + 1] = True
    if c["kind"] == "gjk_epa":
        masks["State"][:] = True  # Already checked by GJK.
    for n, m in masks.items():
        for r in (a, e):
            np.testing.assert_array_equal(
                r[n][~m], b[n][~m], err_msg="EPA guard/readonly " + n
            )
    n, vertices, shapes, q, shift, _ = validate_inputs(l, c, b, OPERATIONS)
    for i in range(n):
        status_a, status_e = (int(row[i, so["status"]]) for row in outputs)
        if status_a != status_e:
            raise AssertionError("EPA status mismatch")
        fs = [
            row[i, so["a"] : so["direction"] + 3].view(np.float32).astype(float)
            for row in outputs
        ]
        if not all(np.isfinite(f).all() for f in fs):
            raise AssertionError("Nonfinite EPA output")
        if not active[i]:
            if c["kind"] == "epa_closest":
                for f in fs:
                    np.testing.assert_array_equal(f, np.zeros(9))
                if status_a != 0:
                    raise AssertionError("Inactive EPA result")
            continue
        valid = status_a == 1
        if status_a not in ((0, 1) if c["kind"] == "epa_closest" else (1, 3)):
            raise AssertionError("Invalid EPA status")
        if not valid:
            fallback = np.zeros(9)
            if c["kind"] == "gjk_epa":
                fallback[6] = 1
            for f in fs:
                np.testing.assert_array_equal(f, fallback)
            continue
        dimensions = [int(row[i, sf["dim"]]) for row in states]
        if dimensions[0] != dimensions[1]:
            raise AssertionError("EPA simplex dimension mismatch")
        aa, bb = shapes[2 * i : 2 * i + 2]
        measures = [
            penetration_witness(
                l,
                f,
                dim,
                aa,
                bb,
                vertices,
                q[i].astype(float),
                shift[i].astype(float),
                k,
            )
            for f, dim in zip(fs, dimensions, strict=True)
        ]
        na, ne = fs[0][6:9], fs[1][6:9]
        if (
            np.array_equal(shift[i], np.zeros(3))
            and symmetric(l, aa, vertices)
            and symmetric(l, bb, vertices)
            and np.dot(na, ne) < 0
        ):
            ne = -ne
        np.testing.assert_allclose(
            na, ne, atol=k["atol"], rtol=k["rtol"], err_msg="EPA normal"
        )
        np.testing.assert_allclose(
            measures[0],
            measures[1],
            atol=k["atol"],
            rtol=k["rtol"],
            err_msg="EPA depth/support gap",
        )
    # Nonunique witness coordinates and opposite symmetric normals remain visible.
    errors = {}
    for name in BUFFERS:
        if name == "Output":
            af, ef = [
                rows[:n, so["a"] : so["direction"] + 3].view(np.float32)
                for rows in outputs
            ]
        elif name == "State":
            af, ef = [rows[:n, : sf["dim"]].view(np.float32) for rows in states]
        else:
            errors[name] = 0.0
            continue
        errors[name] = float(np.max(np.abs(af - ef))) if af.size else 0.0
    return errors
