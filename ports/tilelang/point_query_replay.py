"""Public helper replay with exact flags and independently checked query geometry."""

import numpy as np
from stage_replay import parse_args, run_stage

BUFFERS = ("Queries", "Vertices", "Indices", "Count", "Results")
OPERATIONS = ("mesh_projection", "mesh_pseudo_normal", "ray_point")


def validate_case(l, c, b):
    dimensions = (
        "count",
        "storage",
        "index_root",
        "vertex_root",
        "nodes",
        "vertices_count",
        "triangles",
        "groups",
    )
    if (
        c["kind"] not in OPERATIONS
        or any(type(c[n]) is not int or not 0 <= c[n] < 2**24 for n in dimensions)
        or c["count"] > c["storage"]
        or c["groups"] * 64 < c["count"]
    ):
        raise ValueError("Invalid query dimensions")
    if (
        set(b) != set(BUFFERS)
        or set(c["buffer_words"]) != set(BUFFERS)
        or any(
            a.dtype != np.uint32 or a.shape != (c["buffer_words"][n],)
            for n, a in b.items()
        )
    ):
        raise ValueError("Invalid query buffers")
    if (
        len(b["Queries"]) != c["storage"] * l["query_words"]
        or len(b["Results"]) != c["storage"] * l["result_words"]
        or len(b["Count"]) != 2
        or b["Count"][0] != c["count"]
    ):
        raise ValueError("Invalid query storage")
    v, idx = b["Vertices"].view(np.float32).reshape(-1, l["vertex_words"]), b["Indices"]
    last = (
        c["vertex_root"] + c["nodes"] * 2 + c["vertices_count"] * 2 + c["triangles"] * 3
    )
    end = c["index_root"] + c["nodes"] * 3 + c["triangles"] * 3
    if last > len(v) or end > len(idx) or not np.isfinite(v[:last, :3]).all():
        raise ValueError("Invalid mesh storage")
    tree = idx[c["index_root"] : c["index_root"] + c["nodes"] * 3].reshape(-1, 3)
    for i, (entry, exit_, triangle) in enumerate(tree):
        if (
            not i < int(exit_) <= c["nodes"]
            or (entry != 0xFFFFFFFF and not i < int(entry) < int(exit_))
            or (entry == 0xFFFFFFFF and triangle >= c["triangles"])
        ):
            raise ValueError("Invalid forward mesh traversal")
        box = v[c["vertex_root"] + i * 2 : c["vertex_root"] + i * 2 + 2, :3]
        if np.any(box[0] > box[1]):
            raise ValueError("Invalid mesh bounds")
    ids = idx[c["index_root"] + c["nodes"] * 3 : end]
    if np.any(ids >= c["vertices_count"]):
        raise ValueError("Invalid mesh vertex index")
    q = l["query"]
    for row in b["Queries"].reshape(-1, l["query_words"])[: c["count"]]:
        floats = row.view(np.float32)
        if (
            not np.isfinite(floats[q["point"] : q["point"] + 3]).all()
            or not np.isfinite(floats[q["direction"] : q["direction"] + 3]).all()
            or not np.isfinite(floats[q["parameter"]])
        ):
            raise ValueError("Nonfinite query")
        if c["kind"] == "mesh_projection" and floats[q["parameter"]] < 0:
            raise ValueError("Negative distance bound")
        if c["kind"] == "mesh_pseudo_normal" and (
            row[q["triangle"]] >= c["triangles"]
            or (row[q["feature"]] == 1 and row[q["feature_id"]] >= 3)
        ):
            raise ValueError("Invalid pseudo-normal index")


def independent(l, c, b):
    q = l["query"]
    vertices = (
        b["Vertices"]
        .view(np.float32)
        .reshape(-1, l["vertex_words"])[:, :3]
        .astype(float)
    )
    start = c["vertex_root"] + c["nodes"] * 2
    indices = b["Indices"][c["index_root"] + c["nodes"] * 3 :][
        : c["triangles"] * 3
    ].reshape(-1, 3)
    if c["kind"] == "mesh_projection":
        # This fixture is the union of two coplanar triangles forming a unit square.
        # Its exact closest-point geometry is independent of triangle Voronoi code.
        np.testing.assert_array_equal(
            vertices[start : start + 4], [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
        )
        np.testing.assert_array_equal(indices, [[0, 1, 2], [1, 3, 2]])
    points = []
    flags = []
    for row in b["Queries"].reshape(-1, l["query_words"])[: c["count"]]:
        f = row.view(np.float32).astype(float)
        pt = f[q["point"] : q["point"] + 3]
        parameter = f[q["parameter"]]
        if c["kind"] == "ray_point":
            point = pt + parameter * f[q["direction"] : q["direction"] + 3]
            found, inside = True, False
        elif c["kind"] == "mesh_projection":
            point = np.array([np.clip(pt[0], 0, 1), np.clip(pt[1], 0, 1), 0.0])
            distance = np.linalg.norm(point - pt)
            found = distance <= parameter
            inside = found and pt[2] <= 0
            if not found:
                point[:] = 0
        else:
            tri, feature, fid = (
                int(row[q[n]]) for n in ("triangle", "feature", "feature_id")
            )
            ids = indices[tri]
            found, inside = True, False
            if feature == 0:
                point = vertices[start + c["vertices_count"] + ids[min(fid, 2)]]
            elif feature == 1:
                point = vertices[start + 2 * c["vertices_count"] + tri * 3 + fid]
            elif feature in (2, 3):
                a, bb, cc = vertices[start + ids]
                point = np.cross(bb - a, cc - a)
            else:
                point = np.zeros(3)
        points.append(point)
        flags.append([found, inside])
    return np.asarray(points).reshape(-1, 3), np.asarray(
        flags, dtype=np.uint32
    ).reshape(-1, 2)


def check_outputs(l, c, b, a, e, k):
    validate_case(l, c, b)
    for result in (a, e):
        for name in BUFFERS:
            if result[name].dtype != np.uint32 or result[name].shape != b[name].shape:
                raise ValueError("Invalid query result storage")
            if name != "Results":
                np.testing.assert_array_equal(
                    result[name], b[name], err_msg="Readonly query input"
                )
        rows = result["Results"].reshape(-1, l["result_words"])
        before = b["Results"].reshape(-1, l["result_words"])
        mask = np.zeros(rows.shape, bool)
        for name, width in (("point", 3), ("found", 1), ("inside", 1)):
            offset = l["result"][name]
            mask[: c["count"], offset : offset + width] = True
        np.testing.assert_array_equal(
            rows[~mask], before[~mask], err_msg="Exact query guard"
        )
    points, flags = independent(l, c, b)
    offset = l["result"]["point"]
    error = 0.0
    for result in (a, e):
        rows = result["Results"].reshape(-1, l["result_words"])[: c["count"]]
        actual = rows.view(np.float32)[:, offset : offset + 3]
        if not np.isfinite(actual).all():
            raise AssertionError("Nonfinite query result")
        np.testing.assert_allclose(
            actual,
            points,
            atol=k["atol"],
            rtol=k["rtol"],
            err_msg="Independent query geometry",
        )
        for j, key in enumerate(("found", "inside")):
            np.testing.assert_array_equal(
                rows[:, l["result"][key]], flags[:, j], err_msg="Exact query flags"
            )
    af = (
        a["Results"]
        .reshape(-1, l["result_words"])[: c["count"]]
        .view(np.float32)[:, offset : offset + 3]
    )
    ef = (
        e["Results"]
        .reshape(-1, l["result_words"])[: c["count"]]
        .view(np.float32)[:, offset : offset + 3]
    )
    np.testing.assert_allclose(af, ef, atol=k["atol"], rtol=k["rtol"])
    if c["count"]:
        error = float(np.max(np.abs(af - ef)))
    return {"point": error}


def main():
    from point_query_kernels import make_kernel

    run_stage(
        parse_args(OPERATIONS, "mesh_projection"),
        oracle="point-queries",
        buffers=BUFFERS,
        make_kernel=make_kernel,
        validate_case=validate_case,
        check_outputs=check_outputs,
        launch=lambda c: (c["groups"], 64, c["iterations"]),
    )


if __name__ == "__main__":
    main()
