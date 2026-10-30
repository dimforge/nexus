"""Checks for public Nexus scene imports, with separate body/collider index spaces."""

import numpy as np
from general_scene_validation import check_checkpoint as check_general_checkpoint
from joint_scene_validation import check_joint_rows
from lbvh_scene_validation import check_trees
from native_geometry import lbvh_vertices
from scan_layout import validate_scan
from scene_trajectory import check_trajectory


def validate_case(l, c, b):
    if (
        c["kind"] != "runtime_scene"
        or c["broadphase"] != "lbvh"
        or c.get("general_collision") is not True
    ):
        raise ValueError("Invalid native runtime scene configuration")
    if type(c.get("contact_reduction")) is not bool or c["rounds"] != max(
        c["color_stride"] - 3, 2
    ):
        raise ValueError("Invalid native pipeline configuration")
    nb = c["batches"]
    if (
        nb < 1
        or c["body_count"] != c["bodies_per_batch"] * nb
        or c["collider_count"] != c["colliders_per_batch"] * nb
        or c["body_storage"] != c["body_capacity_per_batch"] * nb
    ):
        raise ValueError("Inconsistent runtime body/collider counts")
    batch = b["BatchIds"]
    for field, value in {
        "num_batches": nb,
        "bodies_len": c["bodies_per_batch"],
        "colliders_len": c["colliders_per_batch"],
        "colliders_batch_capacity": c["body_capacity_per_batch"],
        "collision_pairs_capacity": c["pair_capacity"],
        "contacts_capacity": c["contact_capacity"],
        "impulse_joints_len": c["joints_per_batch"],
        "solver_color_buckets_stride": c["color_stride"],
    }.items():
        if int(batch[l["batch"][field]]) != value:
            raise ValueError(f"Runtime native uniform differs: {field}")
    if set(b) != set(c["buffer_words"]) or any(
        v.dtype != np.uint32 or v.shape != (c["buffer_words"][n],) for n, v in b.items()
    ):
        raise ValueError("Invalid runtime buffer shape/type")
    validate_scan(c, b, "Adj", c["body_storage"], "strides", "levels")
    validate_scan(
        c, b, "Bucket", c["color_stride"] * nb, "bucket_strides", "bucket_levels"
    )
    from scene_hub_validation import validate

    validate(l, c, b)
    lbvh_vertices(l, b)
    for name, count in (
        ("Poses", c["body_count"]),
        ("LocalPoses", c["collider_count"]),
    ):
        rows = b[name].view(np.float32).reshape(-1, l["pose_words"])[:count]
        q = rows[:, l["pose_rotation"] : l["pose_rotation"] + 4]
        if not np.isfinite(rows).all() or not np.allclose(
            np.linalg.norm(q.astype(float), axis=1), 1, atol=2e-6, rtol=0
        ):
            raise ValueError("Invalid runtime body/collider pose")
    for name, width in (
        ("Velocities", l["velocity_words"]),
        ("LocalMass", l["local_words"]),
    ):
        if not np.isfinite(
            b[name].view(np.float32).reshape(-1, width)[: c["body_count"]]
        ).all():
            raise ValueError("Nonfinite runtime dynamics input")
    mass = (
        b["LocalMass"].view(np.float32).reshape(-1, l["local_words"])[: c["body_count"]]
    )
    for offset in (l["local_inv_mass"], l["local_inv_inertia"]):
        if np.any(mass[:, offset : offset + 3] < 0):
            raise ValueError("Negative runtime inverse mass/inertia")
    materials = b["Materials"].reshape(-1, l["material_words"])[: c["collider_count"]]
    for field in ("friction", "restitution"):
        values = materials[:, l["material"][field]].view(np.float32)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Invalid runtime material")
    for field in ("friction_combine_rule", "restitution_combine_rule"):
        if np.any(materials[:, l["material"][field]] > 3):
            raise ValueError("Invalid runtime material combine rule")
    parents = b["Parents"][: c["collider_count"]]
    if np.any(parents >= c["body_count"]) or np.any(
        parents % nb != np.arange(len(parents)) % nb
    ):
        raise ValueError("Runtime collider refers to another batch or inactive body")
    if c["joint_count"]:
        rows = b["Joints"].reshape(-1, l["impulse_joint_words"])
        ends = b["JointColors"][: c["joint_color_count"]].astype(int)
        if (
            len(ends) == 0
            or ends[-1] != c["joints_per_batch"]
            or np.any(np.diff(np.r_[0, ends]) <= 0)
        ):
            raise ValueError("Invalid runtime joint color partition")
        for start, end in zip(np.r_[0, ends[:-1]], ends):
            used = set()
            for i in range(start * nb, end * nb):
                for field in ("body_a", "body_b"):
                    local = int(rows[i, l["impulse_joint"][field]])
                    slot = local * nb + i % nb
                    if slot >= c["body_count"] or slot in used:
                        raise ValueError(
                            "Runtime joint color aliases a velocity writer"
                        )
                    used.add(slot)
    rf = l["rbd_params"]
    if "friction_models" in l:
        twist = b["RbdParams"][rf["friction_model"]] == l["friction_models"]["twist"]
        if twist != (l.get("friction_model") == "twist"):
            raise ValueError("Runtime contact ABI differs from friction model")
    params = b["RbdParams"]
    if (
        params[rf["num_solver_iterations"]] != c["substeps"]
        or params[rf["num_internal_pgs_iterations"]] != c["pgs"]
    ):
        raise ValueError("Runtime iteration schedule differs from native parameters")
    if (
        bool(params.view(np.float32)[rf["warmstart_coefficient"]] != 1)
        != c["scale_warmstart"]
    ):
        raise ValueError("Runtime warmstart schedule differs from native parameters")


def check_runtime_checkpoint(l, c, b, a, e, k):
    errors = check_general_checkpoint(l, c, b, a, e, k)
    if c["joint_count"]:
        errors.update(check_joint_rows(l, c, b, a, e, k))
    else:
        for name in (
            "Joints",
            "Builders",
            "JointConstraints",
            "JointColors",
            "MotorParams",
            "JointBias",
            "JointUnbiased",
        ):
            for result in (a, e):
                np.testing.assert_array_equal(
                    result[name], b[name], err_msg=f"Inactive joints: {name}"
                )
    check_trees(l, c, b, a, e, k)
    for name in b:
        if name.startswith("Color") and name[5:].isdecimal():
            for result in (a, e):
                np.testing.assert_array_equal(
                    result[name], b[name], err_msg=f"Runtime color uniform {name}"
                )
    return errors


def check_states(l, c, b, a, e, frame_indices=None):
    frame_indices = range(c["frames"]) if frame_indices is None else frame_indices
    n, pw, vw = c["body_count"], l["pose_words"], l["velocity_words"]
    mass = b["LocalMass"].view(np.float32).reshape(-1, l["local_words"])
    initial_vel = b["Velocities"].view(np.float32).reshape(-1, vw)
    static = np.all(
        mass[:n, l["local_inv_mass"] : l["local_inv_mass"] + 3] == 0, axis=1
    )
    static &= np.all(
        mass[:n, l["local_inv_inertia"] : l["local_inv_inertia"] + 3] == 0, axis=1
    )
    static &= np.all(initial_vel[:n] == 0, axis=1)
    initial = b["Poses"].reshape(-1, pw)[:n]
    for result in (a, e):
        trace = result["Trajectory"][: c["frames"] * c["trace_stride"]].reshape(
            c["frames"], c["trace_stride"]
        )
        for index in frame_indices:
            poses = trace[index, : n * pw].reshape(n, pw)
            rotation = poses.view(np.float32)[
                :, l["pose_rotation"] : l["pose_rotation"] + 4
            ]
            np.testing.assert_allclose(
                np.linalg.norm(rotation.astype(float), axis=1),
                1,
                atol=2e-5,
                rtol=0,
                err_msg="Runtime unit rotation",
            )
            np.testing.assert_array_equal(
                poses[static], initial[static], err_msg="Runtime fixed body changed"
            )
        if c["joint_count"]:
            rows = result["JointConstraints"].reshape(-1, l["joint_constraint_words"])[
                : c["joint_count"]
            ]
            for field in ("solver_vel_a", "solver_vel_b"):
                if np.any(rows[:, l["joint_constraint"][field]] >= n):
                    raise AssertionError(
                        "Runtime joint references inactive body padding"
                    )


def check_outputs(l, c, b, a, e, k):
    validate_case(l, c, b)
    errors = check_runtime_checkpoint(l, c, b, a, e, k)
    errors["Trajectory"] = check_trajectory(l, c, b, a, e, k)
    check_states(l, c, b, a, e)
    from runtime_scene_physics import check

    for result in (a, e):
        check(l, c, b, result, k)
    return errors
