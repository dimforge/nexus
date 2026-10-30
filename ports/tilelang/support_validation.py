"""Native-derived support/GJK ABI and exact state/witness comparisons."""

import numpy as np
from support_geometry import check as check_geometry
from support_geometry import contains, inverse, support_value

OPERATIONS = ("support_local", "support_world", "cso_support", "gjk_closest")
BUFFERS = ("Shapes", "Poses", "Vertices", "Inputs", "Output", "State")


def validate_case(l, c, b):
    if (
        l.get("wire_format") != "explicit_geometry_diagnostics_v1"
        or c.get("kind") not in OPERATIONS
    ):
        raise ValueError("Invalid support operation or wire format")
    n = c["count"]
    storage = c["storage"]
    if (
        type(n) is not int
        or n < 0
        or storage != n + 3
        or c["groups"] != 1
        or c["iterations"] != 1
    ):
        raise ValueError("Invalid support capacity or dispatch")
    sizes = {
        "Shapes": 2 * storage * l["shape_words"],
        "Poses": storage * l["pose_words"],
        "Inputs": storage * l["support_input_words"],
        "Output": storage * l["support_output_words"],
        "State": storage * l["simplex_words"],
        "Vertices": c["buffer_words"]["Vertices"],
    }
    if (
        set(b) != set(BUFFERS)
        or sizes != c["buffer_words"]
        or any(
            b[key].dtype != np.uint32 or b[key].shape != (size,)
            for key, size in sizes.items()
        )
        or sizes["Vertices"] % l["vertex_words"]
    ):
        raise ValueError("Invalid support storage ABI")
    vertices = (
        b["Vertices"]
        .view(np.float32)
        .reshape(-1, l["vertex_words"])[:, l["vertex_vector"] : l["vertex_vector"] + 3]
        .astype(np.float64)
    )
    if not np.isfinite(vertices).all() or np.any(np.abs(vertices) > 1e6):
        raise ValueError("Invalid support vertices")
    sf = l["shape"]
    types = l["shape_types"]
    shapes = b["Shapes"].reshape(-1, l["shape_words"])
    for shape in shapes[: 2 * n]:
        ty = int(shape[sf["a"] + 3])
        a = shape.view(np.float32)[sf["a"] : sf["a"] + 3]
        bvec = shape.view(np.float32)[sf["b"] : sf["b"] + 4]
        if ty not in [
            types[name]
            for name in (
                "ball",
                "cuboid",
                "capsule",
                "cone",
                "cylinder",
                "convex_poly",
                "triangle",
            )
        ]:
            raise ValueError("Unsupported convex shape")
        if not np.isfinite(shape.view(np.float32)).all() or np.any(
            np.abs(shape.view(np.float32)) > 1e6
        ):
            raise ValueError("Invalid shape floats")
        if (
            (ty == types["ball"] and a[0] < 0)
            or (ty == types["cuboid"] and np.any(a < 0))
            or (ty == types["capsule"] and bvec[3] < 0)
            or (ty in (types["cone"], types["cylinder"]) and np.any(a[:2] <= 0))
        ):
            raise ValueError("Invalid convex dimensions")
        if ty == types["convex_poly"] and not 0 <= int(shape[sf["a"]]) < int(
            shape[sf["a"] + 1]
        ) <= len(vertices):
            raise ValueError("Invalid convex vertex range")
    poses = b["Poses"].view(np.float32).reshape(-1, l["pose_words"])[:n]
    q = poses[:, l["pose_rotation"] : l["pose_rotation"] + 4]
    t = poses[:, l["pose_translation"] : l["pose_translation"] + 3]
    if (
        not np.isfinite(q).all()
        or not np.isfinite(t).all()
        or np.any(np.abs(t) > 1e6)
        or not np.allclose(
            np.sum(q.astype(np.float64) ** 2, axis=1), 1, atol=2e-6, rtol=0
        )
    ):
        raise ValueError("Invalid support pose")
    si = l["support_input"]
    rows = b["Inputs"].reshape(-1, l["support_input_words"])[:n]
    directions = rows[:, si["direction"] : si["direction"] + 3].view(np.float32)
    distance = rows[:, si["max_distance"]].copy().view(np.float32)
    if (
        not np.isfinite(directions).all()
        or np.any(np.linalg.norm(directions.astype(np.float64), axis=1) < 1e-12)
        or np.any(np.abs(directions) > 1e6)
        or not np.isfinite(distance).all()
        or np.any(distance < 0)
        or np.any(rows[:, si["exact"]] > 1)
    ):
        raise ValueError("Invalid support direction or distance policy")
    state = b["State"].reshape(-1, l["simplex_words"])[:n]
    ss = l["simplex"]
    for i, row in enumerate(state):
        if (
            row[ss["dim"]] != 0
            or row[ss["previous_dim"]] != 0
            or not np.array_equal(
                row[ss["previous_vertices"] : ss["previous_vertices"] + 4],
                np.arange(4, dtype=np.uint32),
            )
        ):
            raise ValueError("Cold GJK initial simplex required")
        if np.any(row[ss["vertices"] + l["cso_words"] : ss["dim"]]):
            raise ValueError("Invalid initial unused simplex state")
        first = (
            row[ss["vertices"] : ss["vertices"] + l["cso_words"]]
            .view(np.float32)
            .reshape(3, 3)
            .astype(np.float64)
        )
        if (
            not np.isfinite(first).all()
            or np.max(np.abs(first[0] - (first[1] - first[2]))) > 3e-6
        ):
            raise ValueError("Invalid initial CSO decomposition")
        if not contains(l, shapes[2 * i], vertices, first[1], True) or abs(
            np.dot(first[1], directions[i])
            - support_value(l, shapes[2 * i], vertices, directions[i])
        ) > 2e-6 * max(1, np.linalg.norm(directions[i])):
            raise ValueError("Invalid initial support witness")
        local_b = inverse(q[i].astype(np.float64), first[2] - t[i].astype(np.float64))
        axis_b = inverse(q[i].astype(np.float64), -directions[i].astype(np.float64))
        if not contains(l, shapes[2 * i + 1], vertices, local_b, True) or abs(
            np.dot(local_b, axis_b)
            - support_value(l, shapes[2 * i + 1], vertices, axis_b)
        ) > 2e-6 * max(1, np.linalg.norm(axis_b)):
            raise ValueError("Invalid initial transformed support witness")


def check_outputs(l, c, b, a, e, k):
    validate_case(l, c, b)
    for r in (a, e):
        if set(r) != set(b) or any(
            r[name].dtype != np.uint32 or r[name].shape != b[name].shape for name in b
        ):
            raise ValueError("Invalid returned support storage")
    masks = {name: np.zeros(value.size, np.uint8) for name, value in b.items()}
    so = l["support_output"]
    sf = l["simplex"]
    ow = l["support_output_words"]
    sw = l["simplex_words"]
    for i in range(c["count"]):
        width = 3 if c["kind"].startswith("support_") else 9
        masks["Output"][i * ow + so["a"] : i * ow + so["a"] + width] = 1
        if c["kind"] == "gjk_closest":
            masks["Output"][i * ow + so["status"]] = 2
            masks["State"][i * sw : i * sw + sf["dim"]] = 1
            masks["State"][i * sw + sf["dim"] : i * sw + sf["new_point"]] = 2
            if (
                a["Output"][i * ow + so["status"]] == 0
                and e["Output"][i * ow + so["status"]] == 0
            ):
                dim = int(a["State"][i * sw + sf["dim"]])
                if dim > 3:
                    raise AssertionError("Invalid GJK simplex dimension")
                masks["State"][
                    i * sw + sf["vertices"] : i * sw
                    + sf["vertices"]
                    + (dim + 1) * l["cso_words"]
                ] = 3
    errors = {}
    for name, m in masks.items():
        for r in (a, e):
            np.testing.assert_array_equal(
                r[name][m == 0],
                b[name][m == 0],
                err_msg=f"Support guard/readonly {name}",
            )
            if not np.isfinite(r[name].view(np.float32)[(m == 1) | (m == 3)]).all():
                raise AssertionError("Nonfinite support result")
        np.testing.assert_array_equal(
            a[name][m == 2], e[name][m == 2], err_msg=f"Exact GJK identity {name}"
        )
        af, ef = (r[name].view(np.float32)[m == 1] for r in (a, e))
        np.testing.assert_allclose(af, ef, atol=k["atol"], rtol=k["rtol"], err_msg=name)
        af, ef = (r[name].view(np.float32)[(m == 1) | (m == 3)] for r in (a, e))
        errors[name] = float(np.max(np.abs(af - ef))) if af.size else 0.0
    for r in (a, e):
        check_geometry(l, c, b, r)
    return errors
