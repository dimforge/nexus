"""Exact identities/storage and independent convex projection checks."""

from itertools import combinations

import numpy as np

OPERATIONS = (
    "segment_projection",
    "triangle_projection",
    "tetrahedron_projection",
    "simplex_init",
    "simplex_add",
    "simplex_project",
)


def buffers(operation):
    return ("State",) if operation.startswith("simplex_") else ("Inputs", "Output")


def validate_case(l, c, b):
    if (
        l.get("wire_format") != "explicit_geometry_diagnostics_v1"
        or c.get("kind") not in OPERATIONS
    ):
        raise ValueError("Invalid geometry operation or wire layout")
    n = c["count"]
    if (
        type(n) is not int
        or n < 0
        or c["storage"] != n + 3
        or c["groups"] != 1
        or c["iterations"] != 1
    ):
        raise ValueError("Invalid geometry capacity or dispatch")
    names = buffers(c["kind"])
    sizes = (
        {"State": c["storage"] * l["simplex_words"]}
        if names == ("State",)
        else {
            "Inputs": c["storage"] * l["projection_input_words"],
            "Output": c["storage"] * l["projection_output_words"],
        }
    )
    if (
        set(b) != set(names)
        or c["buffer_words"] != sizes
        or any(
            b[name].dtype != np.uint32 or b[name].shape != (size,)
            for name, size in sizes.items()
        )
    ):
        raise ValueError("Invalid geometry storage ABI")
    if names == ("State",):
        f = l["simplex"]
        rows = b["State"].reshape(-1, l["simplex_words"])[:n]
        max_dim = 2 if c["kind"] == "simplex_add" else 3
        if (
            np.any(rows[:, f["dim"]] > max_dim)
            or np.any(rows[:, f["previous_dim"]] > 3)
            or np.any(rows[:, f["previous_vertices"] : f["previous_vertices"] + 4] > 3)
        ):
            raise ValueError("Invalid simplex dimension or previous index")
        floats = np.r_[
            rows[:, : f["dim"]].ravel(),
            rows[:, f["new_point"] : f["new_point"] + l["cso_words"]].ravel(),
        ].view(np.float32)
    else:
        f = l["projection_input"]
        rows = b["Inputs"].reshape(-1, l["projection_input_words"])[:n]
        if np.any(rows[:, f["solid"]] > 1):
            raise ValueError("Invalid projection solid flag")
        floats = np.concatenate(
            [
                rows[:, f[name] : f[name] + 3].ravel()
                for name in ("a", "b", "c", "d", "query")
            ]
        ).view(np.float32)
    if not np.isfinite(floats).all() or np.any(np.abs(floats) > 1e6):
        raise ValueError("Invalid finite geometry input range")


def closest_on_hull(vertices, point):
    """Independent f64 subset least-squares; at most four input vertices."""
    best = None
    distance = float("inf")
    for count in range(1, len(vertices) + 1):
        for ids in combinations(range(len(vertices)), count):
            v = vertices[list(ids)]
            if count == 1:
                weights = np.ones(1)
            else:
                matrix = (v[1:] - v[0]).T
                result, _, rank, _ = np.linalg.lstsq(matrix, point - v[0], rcond=None)
                if rank < count - 1:
                    continue
                weights = np.r_[1 - result.sum(), result]
            if np.min(weights) < -1e-10:
                continue
            candidate = weights @ v
            error = float(np.dot(candidate - point, candidate - point))
            if error < distance:
                best = candidate
                distance = error
    return best


def independent_projection(vertices, point, projected, contract):
    scale = max(1.0, float(np.max(np.abs(vertices))), float(np.max(np.abs(point))))
    # Near affine dependence has native fallback semantics; it is retained in
    # exact reference tests but does not certify a nondegenerate solid.
    if len(vertices) > 1:
        singular = np.linalg.svd((vertices[1:] - vertices[0]).T, compute_uv=False)
        if singular[-1] <= contract["geometry"]["rank_epsilon"] * scale:
            return
    closest = closest_on_hull(vertices, point)
    if (
        closest is None
        or np.max(np.abs(projected - closest))
        > contract["geometry"]["projection_atol"] * scale
    ):
        raise AssertionError("Independent closest-point geometry")


def check_projection(l, c, b, r, k):
    pi, po = l["projection_input"], l["projection_output"]
    inputs = b["Inputs"].reshape(-1, l["projection_input_words"])[: c["count"]]
    rows = r["Output"].reshape(-1, l["projection_output_words"])[: c["count"]]
    count = {
        "segment_projection": 2,
        "triangle_projection": 3,
        "tetrahedron_projection": 4,
    }[c["kind"]]
    for source, row in zip(inputs, rows):
        f = source.view(np.float32).astype(np.float64)
        out = row.view(np.float32).astype(np.float64)
        vertices = np.array(
            [f[pi[name] : pi[name] + 3] for name in ("a", "b", "c", "d")[:count]]
        )
        feature, index, inside = (
            int(row[po[name]]) for name in ("feature", "index", "inside")
        )
        if inside > 1 or feature > 3:
            raise AssertionError("Invalid projection identity")
        if feature == 0 and index >= count:
            raise AssertionError("Invalid vertex identity")
        if feature == 1 and index >= {2: 1, 3: 3, 4: 6}[count]:
            raise AssertionError("Invalid edge identity")
        if feature == 2 and (
            count == 2
            or (count == 3 and index not in (3, 4))
            or (count == 4 and index >= 4)
        ):
            raise AssertionError("Invalid face identity")
        independent_projection(
            vertices,
            f[pi["query"] : pi["query"] + 3],
            out[po["point"] : po["point"] + 3],
            k,
        )


def check_simplex(l, c, b, r, k):
    sf = l["simplex"]
    sw = l["simplex_words"]
    cw = l["cso_words"]
    before = b["State"].reshape(-1, sw)[: c["count"]]
    after = r["State"].reshape(-1, sw)[: c["count"]]
    for old, new in zip(before, after):
        dim = int(new[sf["dim"]])
        previous = int(new[sf["previous_dim"]])
        if (
            dim > 3
            or previous > 3
            or np.any(new[sf["previous_vertices"] : sf["previous_vertices"] + 4] > 3)
        ):
            raise AssertionError("Invalid output simplex state")
        if c["kind"] == "simplex_add" and dim not in (
            int(old[sf["dim"]]),
            int(old[sf["dim"]]) + 1,
        ):
            raise AssertionError("Invalid simplex growth")
        if c["kind"] == "simplex_project":
            old_dim = int(old[sf["dim"]])
            vertices = (
                old[sf["vertices"] : sf["vertices"] + 4 * cw]
                .view(np.float32)
                .reshape(4, 3, 3)[: old_dim + 1, 0]
                .astype(np.float64)
            )
            point = (
                new[sf["projected"] : sf["projected"] + 3]
                .view(np.float32)
                .astype(np.float64)
            )
            independent_projection(vertices, np.zeros(3), point, k)
            if dim > old_dim:
                raise AssertionError("Projection grew simplex")
            if dim < 3:
                weights = (
                    new[sf["projection"] : sf["projection"] + dim + 1]
                    .view(np.float32)
                    .astype(np.float64)
                )
                active = (
                    new[sf["vertices"] : sf["vertices"] + 4 * cw]
                    .view(np.float32)
                    .reshape(4, 3, 3)[: dim + 1, 0]
                    .astype(np.float64)
                )
                scale = max(1.0, float(np.max(np.abs(active))))
                if (
                    abs(weights.sum() - 1) > k["geometry"]["projection_atol"]
                    or np.min(weights) < -k["geometry"]["projection_atol"]
                    or np.max(np.abs(weights @ active - point))
                    > k["geometry"]["projection_atol"] * scale
                ):
                    raise AssertionError("Simplex barycentric reconstruction")


def check_outputs(l, c, b, a, e, k):
    validate_case(l, c, b)
    for r in (a, e):
        if set(r) != set(b) or any(
            r[name].dtype != np.uint32 or r[name].shape != b[name].shape for name in b
        ):
            raise ValueError("Invalid returned geometry storage")
    masks = {name: np.zeros(v.size, np.uint8) for name, v in b.items()}
    if c["kind"].startswith("simplex_"):
        sf = l["simplex"]
        mask = masks["State"].reshape(-1, l["simplex_words"])
        mask[: c["count"], : sf["new_point"]] = 2
        if c["kind"] == "simplex_project":
            for field in ("projection", "projected"):
                mask[: c["count"], sf[field] : sf[field] + 3] = 1
    else:
        po = l["projection_output"]
        mask = masks["Output"].reshape(-1, l["projection_output_words"])
        mask[: c["count"], : po["guard"]] = 2
        for field in ("point", "barycentric", "expanded"):
            mask[: c["count"], po[field] : po[field] + 3] = 1
    errors = {}
    for name, m in masks.items():
        for r in (a, e):
            np.testing.assert_array_equal(
                r[name][m == 0], b[name][m == 0], err_msg=f"Readonly/guard {name}"
            )
        np.testing.assert_array_equal(
            a[name][m == 2], e[name][m == 2], err_msg=f"Exact geometry state {name}"
        )
        af, ef = (r[name].view(np.float32)[m == 1] for r in (a, e))
        if not np.isfinite(af).all() or not np.isfinite(ef).all():
            raise AssertionError("Nonfinite geometry result")
        np.testing.assert_allclose(af, ef, atol=k["atol"], rtol=k["rtol"])
        errors[name] = float(np.max(np.abs(af - ef))) if af.size else 0.0
    for r in (a, e):
        if c["kind"].startswith("simplex_"):
            check_simplex(l, c, b, r, k)
        else:
            check_projection(l, c, b, r, k)
    return errors
