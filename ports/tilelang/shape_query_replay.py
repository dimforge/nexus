"""Native shape wrappers with independent distance/membership and singularity checks."""

import numpy as np
from stage_replay import parse_args, run_stage

BUFFERS = ("Shapes", "Poses", "Queries", "Count", "Results")
OPERATIONS = (
    "shape_project_local",
    "shape_project_local_boundary",
    "shape_project_world",
    "shape_project_world_boundary",
)


def validate_case(l, c, b):
    if (
        c["kind"] not in OPERATIONS
        or any(
            type(c[n]) is not int or not 0 <= c[n] < 2**24
            for n in ("count", "storage", "shapes", "groups")
        )
        or c["count"] > c["storage"]
        or c["groups"] * 64 < c["count"]
    ):
        raise ValueError("Invalid shape query dimensions")
    sizes = {
        "Shapes": c["shapes"] * l["shape_words"],
        "Poses": c["storage"] * l["pose_words"],
        "Queries": c["storage"] * l["query_words"],
        "Count": 2,
        "Results": c["storage"] * l["result_words"],
    }
    if (
        c["buffer_words"] != sizes
        or set(b) != set(BUFFERS)
        or any(a.dtype != np.uint32 or a.shape != (sizes[n],) for n, a in b.items())
        or b["Count"][0] != c["count"]
    ):
        raise ValueError("Invalid shape query storage")
    for i in range(c["count"]):
        row = b["Queries"][i * l["query_words"] : (i + 1) * l["query_words"]]
        if (
            row[l["query"]["shape_index"]] >= c["shapes"]
            or not np.isfinite(
                row.view(np.float32)[l["query"]["point"] : l["query"]["point"] + 3]
            ).all()
        ):
            raise ValueError("Invalid shape query index or point")
        pose = b["Poses"][i * l["pose_words"] : (i + 1) * l["pose_words"]].view(
            np.float32
        )
        q = pose[l["pose_rotation"] : l["pose_rotation"] + 4]
        if (
            not np.isfinite(pose).all()
            or abs(np.linalg.norm(q.astype(float)) - 1) > 2e-6
        ):
            raise ValueError("Invalid shape query pose")
    for row in b["Shapes"].reshape(-1, l["shape_words"]):
        f = row.view(np.float32)
        s = l["shape"]
        ty = int(row[s["a"] + 3])
        types = l["shape_types"]
        if ty not in types.values():
            continue
        a = f[s["a"] : s["a"] + 3]
        bb = f[s["b"] : s["b"] + 3]
        if not np.isfinite(a).all() or not np.isfinite(bb).all():
            raise ValueError("Nonfinite shape")
        dims = (
            a[:1]
            if ty == types["ball"]
            else a
            if ty == types["cuboid"]
            else a[:2]
            if ty in (types["cone"], types["cylinder"])
            else [f[s["b"] + 3]]
        )
        if not np.isfinite(dims).all() or np.any(np.asarray(dims) < 0):
            raise ValueError("Invalid shape dimensions")
        if ty == types["cone"] and a[0] <= 0:
            raise ValueError("Degenerate cone outside the shape-query contract")


def rotate(q, v):
    return (
        v * (q[3] * q[3] - np.dot(q[:3], q[:3]))
        + q[:3] * (2 * np.dot(q[:3], v))
        + np.cross(q[:3], v) * (2 * q[3])
    )


def local_data(l, c, b, i):
    row = b["Queries"].reshape(-1, l["query_words"])[i]
    s = l["shape"]
    shape = b["Shapes"].reshape(-1, l["shape_words"])[
        int(row[l["query"]["shape_index"]])
    ]
    sf = shape.view(np.float32).astype(float)
    ty = int(shape[s["a"] + 3])
    a = sf[s["a"] : s["a"] + 3]
    bb = sf[s["b"] : s["b"] + 3]
    point = row.view(np.float32)[l["query"]["point"] : l["query"]["point"] + 3].astype(
        float
    )
    pose = b["Poses"].reshape(-1, l["pose_words"])[i].view(np.float32).astype(float)
    q = pose[l["pose_rotation"] : l["pose_rotation"] + 4]
    translation = pose[l["pose_translation"] : l["pose_translation"] + 3]
    if "world" in c["kind"]:
        point = rotate(q * np.array([-1, -1, -1, 1]), point - translation)
    return ty, a, bb, sf[s["b"] + 3], point, q, translation


def distance_to_surface(types, ty, a, b, radius, p):
    if ty == types["ball"]:
        distance = np.linalg.norm(p)
        return distance <= a[0], abs(distance - a[0])
    if ty == types["cuboid"]:
        delta = np.maximum(np.abs(p) - a, 0)
        inside = bool(np.all(np.abs(p) <= a))
        return inside, float(np.min(a - np.abs(p))) if inside else float(
            np.linalg.norm(delta)
        )
    if ty == types["capsule"]:
        axis = b - a
        den = np.dot(axis, axis)
        t = np.clip(np.dot(p - a, axis) / den, 0, 1) if den else 0
        distance = np.linalg.norm(p - (a + t * axis))
        return distance <= radius, abs(distance - radius)
    height, rad = a[:2]
    rho = np.linalg.norm(p[[0, 2]])
    if ty == types["cylinder"]:
        inside = abs(p[1]) <= height and rho <= rad
        distance = (
            min(height - abs(p[1]), rad - rho)
            if inside
            else np.linalg.norm([max(abs(p[1]) - height, 0), max(rho - rad, 0)])
        )
        return inside, float(distance)
    if ty == types["cone"]:
        side = np.array([rad, -2 * height])
        from_apex = np.array([rho, p[1] - height])
        t = np.clip(np.dot(from_apex, side) / np.dot(side, side), 0, 1)
        side_distance = np.linalg.norm(from_apex - side * t)
        base_distance = np.linalg.norm([max(rho - rad, 0), p[1] + height])
        inside = -height <= p[1] <= height and rho <= rad * (height - p[1]) / (
            2 * height
        )
        return inside, float(min(side_distance, base_distance))
    return False, 0.0


def check_outputs(l, c, b, a, e, k):
    validate_case(l, c, b)
    types = l["shape_types"]
    boundary = c["kind"].endswith("boundary")
    offset = l["result"]["point"]
    max_error = 0.0
    for result in (a, e):
        for name in BUFFERS:
            if result[name].dtype != np.uint32 or result[name].shape != b[name].shape:
                raise ValueError("Invalid shape query result storage")
            if name != "Results":
                np.testing.assert_array_equal(
                    result[name], b[name], err_msg="Readonly shape query"
                )
        rows = result["Results"].reshape(-1, l["result_words"])
        before = b["Results"].reshape(rows.shape)
        mask = np.zeros(rows.shape, bool)
        mask[: c["count"], offset : offset + 3] = True
        mask[: c["count"], l["result"]["inside"]] = True
        np.testing.assert_array_equal(
            rows[~mask], before[~mask], err_msg="Exact shape query guard"
        )
    for i in range(c["count"]):
        ty, aa, bb, radius, point, q, translation = local_data(l, c, b, i)
        singular = boundary and ty == types["ball"] and np.dot(point, point) == 0.0
        native_inside, distance = distance_to_surface(types, ty, aa, bb, radius, point)
        for result in (a, e):
            row = result["Results"].reshape(-1, l["result_words"])[i]
            output = row.view(np.float32)[offset : offset + 3].astype(float)
            if singular:
                if not np.isnan(output).all():
                    raise AssertionError("Native ball-center boundary singularity")
            else:
                if not np.isfinite(output).all():
                    raise AssertionError("Unexpected nonfinite shape query")
                if "world" in c["kind"] and ty in types.values():
                    output = rotate(q * np.array([-1, -1, -1, 1]), output - translation)
                if ty not in types.values():
                    original = b["Queries"].reshape(-1, l["query_words"])[
                        i, l["query"]["point"] : l["query"]["point"] + 3
                    ]
                    np.testing.assert_array_equal(
                        row[offset : offset + 3],
                        original,
                        err_msg="Native unsupported shape fallback",
                    )
                else:
                    expected_distance = (
                        distance if boundary or not native_inside else 0.0
                    )
                    np.testing.assert_allclose(
                        np.linalg.norm(output - point),
                        expected_distance,
                        atol=k["geometry_atol"],
                        rtol=k["rtol"],
                        err_msg="Independent shape closest distance",
                    )
                    if boundary or not native_inside:
                        _, surface = distance_to_surface(
                            types, ty, aa, bb, radius, output
                        )
                        np.testing.assert_allclose(
                            surface,
                            0.0,
                            atol=k["geometry_atol"],
                            rtol=0,
                            err_msg="Independent shape surface",
                        )
            if int(row[l["result"]["inside"]]) != int(boundary and native_inside):
                raise AssertionError("Exact native inside flag")
        ar = (
            a["Results"]
            .reshape(-1, l["result_words"])[i, offset : offset + 3]
            .view(np.float32)
        )
        er = (
            e["Results"]
            .reshape(-1, l["result_words"])[i, offset : offset + 3]
            .view(np.float32)
        )
        if not singular:
            np.testing.assert_allclose(ar, er, atol=k["atol"], rtol=k["rtol"])
            max_error = max(max_error, float(np.max(np.abs(ar - er))))
    return {"point": max_error}


def main():
    from shape_query_kernels import make_kernel

    run_stage(
        parse_args(OPERATIONS, "shape_project_local"),
        oracle="shape-queries",
        buffers=BUFFERS,
        make_kernel=make_kernel,
        validate_case=validate_case,
        check_outputs=check_outputs,
        launch=lambda c: (c["groups"], 64, c["iterations"]),
    )


if __name__ == "__main__":
    main()
