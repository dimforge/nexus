"""Native polygonal feature ABI, clipping parity and independent hull checks."""

import numpy as np
from geometry_validation import closest_on_hull
from support_geometry import contains, inverse, rotate
from support_validation import validate_shapes

OPERATIONS = ("support_face", "polygonal_contacts")
FACE_BUFFERS = ("Shapes", "Vertices", "Indices", "Inputs", "Output")
CLIP_BUFFERS = ("Features", "Poses", "Inputs", "Output")


def validate_case(l, c, b):
    n = c["count"]
    storage = c["storage"]
    kind = c["kind"]
    if (
        kind not in OPERATIONS
        or type(n) is not int
        or n < 0
        or storage != n + 3
        or c["groups"] != 1
        or c["iterations"] != 1
    ):
        raise ValueError("Invalid feature dispatch")
    if kind == "support_face":
        sizes = {
            "Shapes": storage * l["shape_words"],
            "Vertices": c["buffer_words"]["Vertices"],
            "Indices": c["buffer_words"]["Indices"],
            "Inputs": storage * l["feature_input_words"],
            "Output": storage * l["feature_words"],
        }
    else:
        sizes = {
            "Features": storage * 2 * l["feature_words"],
            "Poses": storage * l["pose_words"],
            "Inputs": storage * l["polygonal_input_words"],
            "Output": storage * l["feature_manifold_words"],
        }
    if (
        set(b) != set(sizes)
        or sizes != c["buffer_words"]
        or any(
            v.dtype != np.uint32 or v.shape != (sizes[name],) for name, v in b.items()
        )
    ):
        raise ValueError("Invalid feature storage ABI")
    if kind == "support_face":
        if sizes["Vertices"] % l["vertex_words"]:
            raise ValueError("Invalid feature vertex stride")
        vertices = (
            b["Vertices"]
            .view(np.float32)
            .reshape(-1, l["vertex_words"])[
                :, l["vertex_vector"] : l["vertex_vector"] + 3
            ]
            .astype(float)
        )
        if not np.isfinite(vertices).all() or np.any(np.abs(vertices) > 1e6):
            raise ValueError("Invalid feature vertices")
        shapes = b["Shapes"].reshape(-1, l["shape_words"])[:n]
        validate_shapes(l, shapes, vertices)
        for shape in shapes:
            sf = l["shape"]
            if shape[sf["a"] + 3] == l["shape_types"]["convex_poly"]:
                first, end = map(int, shape[sf["b"] : sf["b"] + 2])
                base, limit = map(int, shape[sf["a"] : sf["a"] + 2])
                if not (0 <= first < end <= len(b["Indices"])) or (end - first) % 3:
                    raise ValueError("Invalid feature face range")
                ids = b["Indices"][first:end].astype(np.int64)
                if np.any(ids >= limit - base):
                    raise ValueError("Invalid feature vertex indices")
                tri = vertices[base + ids].reshape(-1, 3, 3)
                if not np.any(
                    np.linalg.norm(
                        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
                    )
                    > 1e-12
                ):
                    raise ValueError("Feature hull requires a nondegenerate face")
        rows = b["Inputs"].reshape(-1, l["feature_input_words"])[:n]
        direction = rows[
            :, l["feature_input"]["direction"] : l["feature_input"]["direction"] + 3
        ].view(np.float32)
        if not np.isfinite(direction).all() or np.any(np.abs(direction) > 1e6):
            raise ValueError("Invalid support face direction")
    else:
        features = b["Features"].reshape(-1, l["feature_words"])[: 2 * n]
        ff = l["feature"]
        coords = features[:, ff["vertices"] : ff["count"]].view(np.float32)
        if (
            np.any(features[:, ff["count"]] > 4)
            or not np.isfinite(coords).all()
            or np.any(np.abs(coords) > 1e6)
        ):
            raise ValueError("Invalid polygonal feature")
        rows = b["Inputs"].reshape(-1, l["polygonal_input_words"])[:n]
        pi = l["polygonal_input"]
        normal = rows[:, pi["normal"] : pi["normal"] + 3].view(np.float32).astype(float)
        prediction = rows[:, pi["prediction"]].copy().view(np.float32)
        poses = b["Poses"].view(np.float32).reshape(-1, l["pose_words"])[:n]
        q = poses[:, l["pose_rotation"] : l["pose_rotation"] + 4].astype(float)
        t = poses[:, l["pose_translation"] : l["pose_translation"] + 3]
        if (
            not np.isfinite(q).all()
            or not np.isfinite(t).all()
            or np.any(np.abs(t) > 1e6)
            or not np.allclose(np.sum(q * q, axis=1), 1, atol=2e-6, rtol=0)
        ):
            raise ValueError("Invalid polygonal pose")
        if (
            not np.isfinite(normal).all()
            or not np.allclose(np.linalg.norm(normal, axis=1), 1, atol=2e-6, rtol=0)
            or not np.isfinite(prediction).all()
            or np.any(prediction < 0)
            or np.any(prediction > 1e6)
            or np.any(rows[:, pi["flipped"]] > 1)
        ):
            raise ValueError("Invalid polygonal clipping parameters")


def check_outputs(l, c, b, a, e, k):
    validate_case(l, c, b)
    for r in (a, e):
        if set(r) != set(b) or any(
            r[n].dtype != np.uint32 or r[n].shape != b[n].shape for n in b
        ):
            raise ValueError("Invalid returned feature storage")
    face = c["kind"] == "support_face"
    width = l["feature_words"] if face else l["feature_manifold_words"]
    fields = l["feature"] if face else l["feature_manifold"]
    count = fields["count"]
    mask = np.zeros(b["Output"].size, np.uint8)
    for i in range(c["count"]):
        mask[i * width : i * width + count] = 1
        mask[i * width + count] = 2
    for r in (a, e):
        for name in b:
            if name != "Output":
                np.testing.assert_array_equal(
                    r[name], b[name], err_msg="Feature readonly " + name
                )
        np.testing.assert_array_equal(
            r["Output"][mask == 0],
            b["Output"][mask == 0],
            err_msg="Feature output guard",
        )
        if not np.isfinite(r["Output"].view(np.float32)[mask == 1]).all():
            raise AssertionError("Nonfinite feature output")
    np.testing.assert_array_equal(
        a["Output"][mask == 2], e["Output"][mask == 2], err_msg="Exact feature count"
    )
    af, ef = [r["Output"].view(np.float32)[mask == 1] for r in (a, e)]
    np.testing.assert_allclose(af, ef, atol=k["atol"], rtol=k["rtol"])
    if face:
        shapes = b["Shapes"].reshape(-1, l["shape_words"])
        vertices = (
            b["Vertices"]
            .view(np.float32)
            .reshape(-1, l["vertex_words"])[
                :, l["vertex_vector"] : l["vertex_vector"] + 3
            ]
            .astype(float)
        )
        for i in range(c["count"]):
            rows = [r["Output"].reshape(-1, width)[i] for r in (a, e)]
            ty = int(shapes[i, l["shape"]["a"] + 3])
            types = l["shape_types"]
            if ty in (
                types["ball"],
                types["cuboid"],
                types["convex_poly"],
                types["triangle"],
            ) or (ty == types["capsule"] and rows[0][count] == 2):
                np.testing.assert_array_equal(
                    rows[0], rows[1], err_msg="Exact copied feature vertices"
                )
            for row in rows:
                if row[count] > 4:
                    raise AssertionError("Invalid returned feature count")
                for point in (
                    row[fields["vertices"] : count]
                    .view(np.float32)
                    .reshape(4, 3)[: row[count]]
                ):
                    if not contains(l, shapes[i], vertices, point.astype(float)):
                        raise AssertionError("Support feature point outside its shape")
    else:
        inputs = b["Inputs"].reshape(-1, l["polygonal_input_words"])
        features = b["Features"].reshape(-1, l["feature_words"])
        ff = l["feature"]
        poses = b["Poses"].view(np.float32).reshape(-1, l["pose_words"]).astype(float)
        for r in (a, e):
            rows = r["Output"].reshape(-1, width)
            for i in range(c["count"]):
                row = rows[i]
                size = int(row[count])
                if size > 4:
                    raise AssertionError("Invalid returned manifold count")
                np.testing.assert_array_equal(
                    row[fields["normal"] : count],
                    np.zeros(3, np.uint32),
                    err_msg="Clipping leaves normal unset",
                )
                side = int(inputs[i, l["polygonal_input"]["flipped"]])
                feature = features[i * 2 + side]
                hull = (
                    feature[ff["vertices"] : ff["count"]]
                    .view(np.float32)
                    .reshape(4, 3)[: feature[ff["count"]]]
                    .astype(float)
                )
                points = (
                    row[fields["points"] : fields["normal"]]
                    .view(np.float32)
                    .reshape(4, l["feature_point_words"])
                )
                prediction = float(
                    inputs[
                        i,
                        l["polygonal_input"]["prediction"] : l["polygonal_input"][
                            "prediction"
                        ]
                        + 1,
                    ].view(np.float32)[0]
                )
                for point in points[:size]:
                    position = point[
                        l["feature_point"]["pt"] : l["feature_point"]["pt"] + 3
                    ].astype(float)
                    if (
                        len(hull) == 0
                        or np.linalg.norm(closest_on_hull(hull, position) - position)
                        > 1e-5
                    ):
                        raise AssertionError(
                            "Clipped contact outside its output feature"
                        )
                    normal = (
                        inputs[
                            i,
                            l["polygonal_input"]["normal"] : l["polygonal_input"][
                                "normal"
                            ]
                            + 3,
                        ]
                        .copy()
                        .view(np.float32)
                        .astype(float)
                    )
                    q = poses[i, l["pose_rotation"] : l["pose_rotation"] + 4]
                    t = poses[i, l["pose_translation"] : l["pose_translation"] + 3]
                    distance = float(point[l["feature_point"]["dist"]])
                    secondary = (
                        rotate(q, position) + t - normal * distance
                        if side
                        else inverse(q, position + normal * distance - t)
                    )
                    other = features[i * 2 + 1 - side]
                    other_hull = (
                        other[ff["vertices"] : ff["count"]]
                        .copy()
                        .view(np.float32)
                        .reshape(4, 3)[: other[ff["count"]]]
                        .astype(float)
                    )
                    if (
                        len(other_hull) == 0
                        or np.linalg.norm(
                            closest_on_hull(other_hull, secondary) - secondary
                        )
                        > 1e-5
                    ):
                        raise AssertionError(
                            "Clipped secondary contact outside its feature"
                        )
                    if point[l["feature_point"]["dist"]] > prediction + 1e-6:
                        raise AssertionError("Clipped contact exceeds prediction")
    return {
        name: (float(np.max(np.abs(af - ef))) if af.size else 0.0)
        if name == "Output"
        else 0.0
        for name in b
    }
