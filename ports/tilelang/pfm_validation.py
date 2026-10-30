"""PFM ABI and geometric parity, including explicitly reported native limitations."""

import numpy as np
from epa_validation import symmetric
from support_geometry import contains, inverse, support_value
from support_validation import validate_shapes

OPERATIONS = ("pfm_subshape", "pfm_manifold")
BUFFERS = (
    "Shapes",
    "Poses",
    "Vertices",
    "Indices",
    "Inputs",
    "Output",
    "SubShapes",
    "Radii",
)


def geometry(l, b):
    shapes = b["Shapes"].reshape(-1, l["shape_words"])
    vertices = b["Vertices"].view(np.float32).reshape(-1, l["vertex_words"])
    vertices = vertices[:, l["vertex_vector"] : l["vertex_vector"] + 3].astype(float)
    poses = b["Poses"].view(np.float32).reshape(-1, l["pose_words"]).astype(float)
    q = poses[:, l["pose_rotation"] : l["pose_rotation"] + 4]
    shift = poses[:, l["pose_translation"] : l["pose_translation"] + 3]
    return shapes, vertices, q, shift


def validate_case(l, c, b):
    n, storage = c["count"], c["storage"]
    if (
        c["kind"] not in OPERATIONS
        or type(n) is not int
        or n < 0
        or storage != n + 3
        or c["groups"] != 1
        or c["iterations"] != 1
    ):
        raise ValueError("Invalid PFM dispatch")
    sizes = {
        "Shapes": storage * 2 * l["shape_words"],
        "SubShapes": storage * 2 * l["shape_words"],
        "Radii": storage * 2 * l["pfm_radius_words"],
        "Poses": storage * l["pose_words"],
        "Inputs": storage * l["pfm_input_words"],
        "Output": storage * l["feature_manifold_words"],
        "Vertices": c["buffer_words"]["Vertices"],
        "Indices": c["buffer_words"]["Indices"],
    }
    if (
        set(b) != set(sizes)
        or c["buffer_words"] != sizes
        or any(
            v.dtype != np.uint32 or v.shape != (sizes[name],) for name, v in b.items()
        )
    ):
        raise ValueError("Invalid PFM storage ABI")
    if sizes["Vertices"] % l["vertex_words"]:
        raise ValueError("Invalid PFM vertex stride")
    shapes, vertices, q, shift = geometry(l, b)
    if not np.isfinite(vertices).all() or np.any(np.abs(vertices) > 1e6):
        raise ValueError("Invalid PFM vertices")
    validate_shapes(l, shapes[: 2 * n], vertices)
    sf = l["shape"]
    for shape in shapes[: 2 * n]:
        if shape[sf["a"] + 3] == l["shape_types"]["convex_poly"]:
            first, end = map(int, shape[sf["b"] : sf["b"] + 2])
            base, limit = map(int, shape[sf["a"] : sf["a"] + 2])
            if not (0 <= first < end <= sizes["Indices"]) or (end - first) % 3:
                raise ValueError("Invalid PFM face range")
            ids = b["Indices"][first:end].astype(np.int64)
            if np.any(ids >= limit - base):
                raise ValueError("Invalid PFM vertex index")
            tri = vertices[base + ids].reshape(-1, 3, 3)
            if not np.any(
                np.linalg.norm(
                    np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
                )
                > 1e-12
            ):
                raise ValueError("PFM convex hull needs a nondegenerate face")
    if (
        not np.isfinite(q[:n]).all()
        or not np.isfinite(shift[:n]).all()
        or np.any(np.abs(shift[:n]) > 1e6)
        or not np.allclose(np.sum(q[:n] * q[:n], axis=1), 1, atol=2e-6, rtol=0)
    ):
        raise ValueError("Invalid PFM relative pose")
    prediction = (
        b["Inputs"]
        .reshape(-1, l["pfm_input_words"])[:n, l["pfm_input"]["prediction"]]
        .copy()
        .view(np.float32)
    )
    if (
        not np.isfinite(prediction).all()
        or np.any(prediction < 0)
        or np.any(prediction > 1e6)
    ):
        raise ValueError("Invalid PFM prediction")


def manifold_fields(l, r, i):
    row = r["Output"].reshape(-1, l["feature_manifold_words"])[i]
    mf = l["feature_manifold"]
    values = row.view(np.float32).astype(float)
    points = values[mf["points"] : mf["normal"]].reshape(4, l["feature_point_words"])
    normal = values[mf["normal"] : mf["normal"] + 3]
    return points, normal, int(row[mf["count"]])


def secondary_inside(l, shape, vertices, q, shift, point, normal):
    pt = l["feature_point"]["pt"]
    return contains(
        l,
        shape,
        vertices,
        inverse(
            q, point[pt : pt + 3] + normal * point[l["feature_point"]["dist"]] - shift
        ),
    )


def check_manifold_row(
    l, sa, sb, vertices, q, shift, prediction, aa, an, ac, ee, en, ec, k
):
    pi, di = l["feature_point"]["pt"], l["feature_point"]["dist"]
    if ac != ec or not 0 <= ac <= 4:
        raise AssertionError("Exact PFM contact count")
    for points, normal, count in ((aa, an, ac), (ee, en, ec)):
        if not np.isfinite(points).all() or not np.isfinite(normal).all():
            raise AssertionError("Nonfinite PFM output")
        if count:
            np.testing.assert_allclose(
                np.linalg.norm(normal),
                1,
                atol=2e-6,
                rtol=0,
                err_msg="PFM unit normal",
            )
            minimum = (
                np.dot(normal, shift)
                - support_value(l, sa, vertices, normal)
                - support_value(l, sb, vertices, inverse(q, -normal))
            )
            for point in points[:count]:
                if not contains(l, sa, vertices, point[pi : pi + 3]):
                    raise AssertionError("PFM primary point outside shape")
                if point[di] > prediction + 2e-6 or point[di] < minimum - 1e-5:
                    raise AssertionError(
                        "PFM distance violates prediction/support bound"
                    )
        else:
            np.testing.assert_array_equal(
                points, np.zeros_like(points), err_msg="Empty PFM points"
            )
            np.testing.assert_array_equal(
                normal, np.zeros(3), err_msg="Empty PFM normal"
            )
    flipped = (
        ac > 0
        and np.dot(an, en) < 0
        and np.array_equal(shift, np.zeros(3))
        and symmetric(l, sa, vertices)
        and symmetric(l, sb, vertices)
    )
    np.testing.assert_allclose(
        an,
        -en if flipped else en,
        atol=k["atol"],
        rtol=k["rtol"],
        err_msg="PFM normal",
    )
    np.testing.assert_allclose(
        aa[:ac, di], ee[:ec, di], atol=k["atol"], rtol=k["rtol"], err_msg="PFM depth"
    )
    for j in range(ac):
        ap, ep = aa[j, pi : pi + 3], ee[j, pi : pi + 3]
        different = not np.allclose(ap, ep, atol=k["atol"], rtol=k["rtol"])
        inside = [
            secondary_inside(l, sb, vertices, q, shift, point, normal)
            for point, normal in ((aa[j], an), (ee[j], en))
        ]
        if inside[1] and not inside[0]:
            raise AssertionError("PFM secondary witness regressed")
        if different:
            # Nonunique points must be valid witnesses in BOTH implementations.
            # Native approximate clipping misses never unlock relaxed comparison.
            if flipped:
                np.testing.assert_allclose(
                    ap,
                    -ep,
                    atol=k["atol"],
                    rtol=k["rtol"],
                    err_msg="PFM symmetric reflection",
                )
            elif np.linalg.norm(ap - ep) > k["max_tangent_drift"]:
                raise AssertionError("PFM nonunique tangent drift exceeded")
            if not all(inside):
                raise AssertionError(
                    "PFM nonunique point lacks valid secondary witness"
                )
            np.testing.assert_allclose(
                np.dot(ap, an),
                np.dot(ep, en),
                atol=k["atol"],
                rtol=k["rtol"],
                err_msg="PFM contact plane",
            )


def check_outputs(l, c, b, a, e, k):
    validate_case(l, c, b)
    n = c["count"]
    masks = {name: np.zeros(value.size, bool) for name, value in b.items()}
    masks["SubShapes"][: n * 2 * l["shape_words"]] = True
    masks["Radii"][: n * 2 * l["pfm_radius_words"]] = True
    manifold = c["kind"] == "pfm_manifold"
    mf, mw = l["feature_manifold"], l["feature_manifold_words"]
    if manifold:
        for i in range(n):
            masks["Output"][i * mw : i * mw + mf["count"] + 1] = True
    for r in (a, e):
        if set(r) != set(b) or any(
            r[name].dtype != np.uint32 or r[name].shape != b[name].shape for name in b
        ):
            raise ValueError("Invalid returned PFM storage")
        for name, mask in masks.items():
            np.testing.assert_array_equal(
                r[name][~mask], b[name][~mask], err_msg="PFM guard/readonly " + name
            )
    for name in ("SubShapes", "Radii"):
        np.testing.assert_array_equal(
            a[name], e[name], err_msg="Exact PFM conversion " + name
        )
    if not manifold:
        return dict.fromkeys(BUFFERS, 0.0)
    shapes, vertices, q, shift = geometry(l, b)
    for i in range(n):
        aa, an, ac = manifold_fields(l, a, i)
        ee, en, ec = manifold_fields(l, e, i)
        if ac != ec or ac > 4:
            raise AssertionError("Exact PFM contact count")
        sa, sb = shapes[2 * i : 2 * i + 2]
        prediction = float(
            b["Inputs"]
            .reshape(-1, l["pfm_input_words"])[
                i, l["pfm_input"]["prediction"] : l["pfm_input"]["prediction"] + 1
            ]
            .view(np.float32)[0]
        )
        check_manifold_row(
            l, sa, sb, vertices, q[i], shift[i], prediction, aa, an, ac, ee, en, ec, k
        )
    af, ef = [
        r["Output"].reshape(-1, mw)[:n, : mf["count"]].view(np.float32) for r in (a, e)
    ]
    result = dict.fromkeys(BUFFERS, 0.0)
    result["Output"] = float(np.max(np.abs(af - ef))) if af.size else 0.0
    return result


def diagnostics(l, c, b, a, e, k):
    del k
    if c["kind"] != "pfm_manifold":
        return {}
    shapes, vertices, q, shift = geometry(l, b)
    result = {}
    for label, r in (("actual", a), ("reference", e)):
        misses = []
        for i in range(c["count"]):
            points, normal, count = manifold_fields(l, r, i)
            for j in range(count):
                if not secondary_inside(
                    l, shapes[2 * i + 1], vertices, q[i], shift[i], points[j], normal
                ):
                    misses.append({"pair": i, "point": j})
        result[label + "_secondary_containment_misses"] = misses
    return result
