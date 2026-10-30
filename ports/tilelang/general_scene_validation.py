"""General scene ABI, collider geometry, and full trajectory checks."""

import numpy as np
import scene_outputs as outputs
import scene_validation as base
from deferred_validation import mesh_indices, vertices
from general_scene_physics import check as check_physics
from general_scene_physics import primitives
from scene_trajectory import check_trajectory
from support_validation import validate_shapes

EXTRA = (
    "GeometryIndices",
    "PfmPairs",
    "PfmIdentity",
    "PfmOrder",
    "PfmKeys",
    "PfmSortedKeys",
)
BUFFERS = base.BUFFERS + EXTRA
SCENES = tuple(
    "general_" + s
    for s in (
        "capsule",
        "cone",
        "cylinder",
        "convex",
        "triangle",
        "mesh",
        "polyline",
        "mixed",
    )
)


def validate_case(l, c, b):
    base.validate_case(l, c, b)
    n = c["body_count"]
    mats = b["Materials"].reshape(-1, l["material_words"])
    for field in ("friction", "restitution"):
        values = mats[:, l["material"][field]].view(np.float32)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Invalid general scene material")
    for field in ("friction_combine_rule", "restitution_combine_rule"):
        if np.any(mats[:, l["material"][field]] > 3):
            raise ValueError("Invalid general material combine rule")
    local = b["LocalPoses"].view(np.float32).reshape(-1, l["pose_words"])[:n]
    if np.any(
        local[:, l["pose_translation"] : l["pose_translation"] + 3] != 0
    ) or not np.all(
        local[:, l["pose_rotation"] : l["pose_rotation"] + 4] == [0, 0, 0, 1]
    ):
        raise ValueError("General scene requires identity local poses")
    filters = b["Filters"].reshape(-1, 2)
    if not np.array_equal(filters[:, 0], b["Parents"]) or np.any(filters[:, 1]):
        raise ValueError("General scene filter mapping changed")
    if c["kind"] != "general_scene" or c["scene"] not in SCENES:
        raise ValueError("Invalid general scene")
    if (
        type(c["vertex_count"]) is not int
        or not 1 <= c["vertex_count"] < 2**20
        or type(c["geometry_indices"]) is not int
        or not 1 <= c["geometry_indices"] < 2**20
    ):
        raise ValueError("Invalid geometry capacity")
    if (
        set(b) != set(BUFFERS)
        or c["trace_stride"]
        != c["body_count"] * (l["pose_words"] + l["velocity_words"])
        or b["FrameIndex"][0] != 0
    ):
        raise ValueError("Invalid general scene trace")
    np.testing.assert_array_equal(
        b["PfmIdentity"], np.arange(c["contact_storage"], dtype=np.uint32)
    )
    shapes = b["Shapes"].reshape(-1, l["shape_words"])
    v = vertices(l, b)
    sf = l["shape"]
    types = l["shape_types"]
    tags = shapes[:, sf["a"] + 3]
    composite = (tags == types["trimesh"]) | (tags == types["polyline"])
    if not np.isfinite(v).all():
        raise ValueError("Nonfinite geometry")
    validate_shapes(l, shapes[~composite], v)
    for shape in shapes[composite]:
        mesh_indices(l, shape, v, b["GeometryIndices"])
    for shape in shapes[tags == types["convex_poly"]]:
        vb, ve = map(int, shape[sf["a"] : sf["a"] + 2])
        ib, ie = map(int, shape[sf["b"] : sf["b"] + 2])
        if (
            not 0 <= ib < ie <= len(b["GeometryIndices"])
            or (ie - ib) % 3
            or np.any(b["GeometryIndices"][ib:ie] >= ve - vb)
        ):
            raise ValueError("Invalid convex indices")
        expected = np.array(
            [
                [
                    (-0.5 if i & 1 == 0 else 0.5),
                    (-0.5 if i & 2 == 0 else 0.5),
                    (-0.5 if i & 4 == 0 else 0.5),
                ]
                for i in range(8)
            ]
        )
        if not np.array_equal(v[vb:ve], expected):
            raise ValueError("General fixture convex geometry changed")
    n, nb = c["body_count"], c["batches"]
    mass = b["LocalMass"].view(np.float32).reshape(-1, l["local_words"])
    if np.any(mass[nb:n, l["local_inv_mass"] : l["local_inv_mass"] + 3] != 1) or np.any(
        mass[nb:n, l["local_inv_inertia"] : l["local_inv_inertia"] + 3] != 6
    ):
        raise ValueError("General fixture mass changed")
    poses = b["Poses"].view(np.float32).reshape(-1, l["pose_words"])
    for i in range(n):
        shape = shapes[i]
        f = shape.view(np.float32)
        kind = int(shape[sf["a"] + 3])
        fixed = i < nb
        expected = (
            "trimesh"
            if fixed and c["scene"] == "general_mesh"
            else "polyline"
            if fixed and c["scene"] == "general_polyline"
            else "cuboid"
            if fixed
            else {
                "general_capsule": "capsule",
                "general_cone": "cone",
                "general_cylinder": "cylinder",
                "general_convex": "convex_poly",
                "general_triangle": "triangle",
                "general_mesh": "convex_poly",
                "general_polyline": "ball",
            }.get(
                c["scene"],
                (
                    "ball",
                    "cuboid",
                    "capsule",
                    "cone",
                    "cylinder",
                    "convex_poly",
                    "triangle",
                )[(i // nb - 1) % 7],
            )
        )
        if kind != types[expected]:
            raise ValueError("General fixture shape tag changed")
        if expected == "ball" and f[sf["a"]] != np.float32(0.5):
            raise ValueError("General fixture ball changed")
        if expected == "cuboid" and not np.array_equal(
            f[sf["a"] : sf["a"] + 3], [20, 0.5, 20] if fixed else [0.5, 0.5, 0.5]
        ):
            raise ValueError("General fixture box changed")
        if expected in ("cone", "cylinder") and not np.array_equal(
            f[sf["a"] : sf["a"] + 2], [0.5, 0.5]
        ):
            raise ValueError("General fixture axial shape changed")
        if expected == "capsule" and not (
            np.array_equal(f[sf["a"] : sf["a"] + 3], np.array([0, -0.3, 0], np.float32))
            and np.array_equal(
                f[sf["b"] : sf["b"] + 4], np.array([0, 0.3, 0, 0.2], np.float32)
            )
        ):
            raise ValueError("General fixture capsule changed")
        if expected == "triangle":
            points = np.array([f[sf[key] : sf[key] + 3] for key in ("a", "b", "c")])
            if not np.array_equal(
                points,
                np.array(
                    [[-0.5, -0.5, -0.4], [0.5, -0.5, -0.4], [0, 0.5, 0.4]], np.float32
                ),
            ):
                raise ValueError("General fixture triangle changed")
        if fixed:
            q = poses[i, l["pose_rotation"] : l["pose_rotation"] + 4]
            y = poses[i, l["pose_translation"] + 1]
            if not np.array_equal(q, [0, 0, 0, 1]) or y != np.float32(
                0 if expected == "trimesh" else -0.4 if expected == "polyline" else -0.5
            ):
                raise ValueError("General fixture ground pose changed")
            if expected in ("trimesh", "polyline"):
                ps = primitives(l, shape, v, b["GeometryIndices"])
                if len(ps) != 2:
                    raise ValueError("General fixture ground topology changed")
                for primitive in ps:
                    pf = primitive.view(np.float32)
                    fields = ("a", "b", "c") if expected == "trimesh" else ("a", "b")
                    for field in fields:
                        p = pf[sf[field] : sf[field] + 3]
                        if (
                            p[1] != 0
                            or abs(p[0]) != 20
                            or abs(p[2])
                            != np.float32(20 if expected == "trimesh" else 0.1)
                        ):
                            raise ValueError("General fixture ground geometry changed")


def check_checkpoint(l, c, b, a, e, k):
    from general_scene_outputs import check_collision

    for r in (a, e):
        if set(r) != set(b) or any(
            v.dtype != np.uint32 or v.shape != b[n].shape for n, v in r.items()
        ):
            raise ValueError("Invalid general scene returned storage")
    bc, ac, ec = ({n: r[n] for n in base.BUFFERS} for r in (b, a, e))
    outputs.check_guards(l, c, bc, ac, ec)
    for r in (a, e):
        for name in ("GeometryIndices", "PfmIdentity"):
            np.testing.assert_array_equal(
                r[name], b[name], err_msg=f"General readonly {name}"
            )
        for name, width, count in [
            ("PfmPairs", l["pfm_pair_words"], c["pair_capacity"]),
            *[
                (n, 1, c["pair_capacity"])
                for n in ("PfmOrder", "PfmKeys", "PfmSortedKeys")
            ],
        ]:
            np.testing.assert_array_equal(
                r[name][count * width :],
                b[name][count * width :],
                err_msg=f"General guard {name}",
            )
    errors = outputs.check_bodies(l, c, bc, ac, ec, k)
    errors.update(check_collision(l, c, b, a, e, k))
    count = outputs.check_graph(l, c, ac, ec)
    errors.update(outputs.check_constraints(l, c, ac, ec, k, count))
    errors.update(outputs.check_states(l, c, ac, ec, k, count))
    return errors


def check_outputs(l, c, b, a, e, k):
    validate_case(l, c, b)
    errors = check_checkpoint(l, c, b, a, e, k)
    errors["Trajectory"] = check_trajectory(l, c, b, a, e, k)
    for r in (a, e):
        check_physics(l, c, b, r, k)
    return errors
