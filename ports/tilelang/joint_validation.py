"""Native 3D joint ABI, conflict-free scheduling and write-set validation."""

BUFFERS = (
    "Joints",
    "Builders",
    "Constraints",
    "LocalMass",
    "WorldMass",
    "Poses",
    "Velocities",
    "Params",
    "BatchIds",
    "ColorGroups",
    "Color",
    "UseBias",
    "MotorParams",
)
OPERATIONS = (
    "joint_motor_params",
    "joint_init",
    "joint_update",
    "joint_orthogonalize",
    "joint_solve",
    "joint_program",
)


def validate_case(l, c, a):
    import numpy as np

    if c["kind"] not in OPERATIONS or any(
        type(c[k]) is not int or not 0 <= c[k] < 2**24
        for k in (
            "count",
            "joints_per_batch",
            "batches",
            "storage",
            "bodies",
            "groups",
            "iterations",
        )
    ):
        raise ValueError("Invalid joint dimensions")
    n, nb = c["count"], c["batches"]
    if not (
        1 <= nb <= 64
        and n == nb * c["joints_per_batch"]
        and c["storage"] == n + 3
        and c["bodies"] >= 2 * n + 3
        and 1 <= c["groups"] <= 65535
        and c["iterations"] == 1
    ):
        raise ValueError("Invalid joint capacities or dispatch")
    sizes = {
        name: c["storage"] * l[key]
        for name, key in [
            ("Joints", "impulse_joint_words"),
            ("Builders", "builder_words"),
            ("Constraints", "joint_constraint_words"),
        ]
    }
    sizes.update(
        {
            name: c["bodies"] * l[key]
            for name, key in [
                ("LocalMass", "local_words"),
                ("WorldMass", "world_words"),
                ("Poses", "pose_words"),
                ("Velocities", "velocity_words"),
            ]
        }
    )
    sizes.update(
        Params=l["rbd_params_words"],
        BatchIds=l["batch_words"],
        ColorGroups=4,
        Color=2,
        UseBias=2,
        MotorParams=c["storage"] * 36,
    )
    if (
        set(a) != set(BUFFERS)
        or c["buffer_words"] != sizes
        or any(
            v.dtype != np.uint32 or v.shape != (sizes[name],) for name, v in a.items()
        )
    ):
        raise ValueError("Invalid joint storage manifest")
    if (
        a["BatchIds"][l["batch"]["num_batches"]] != nb
        or a["BatchIds"][l["batch"]["impulse_joints_len"]] != c["joints_per_batch"]
    ):
        raise ValueError("Joint specialization differs from batch uniforms")
    colors = a["ColorGroups"][:3].astype(np.int64)
    if (
        np.any(colors[1:] < colors[:-1])
        or colors[-1] != c["joints_per_batch"]
        or a["Color"][0] >= 3
        or a["UseBias"][0] > 1
    ):
        raise ValueError("Invalid joint color range")
    velocities = a["Velocities"].view(np.float32).reshape(-1, l["velocity_words"])
    if any(
        not np.isfinite(velocities[:, l[field] : l[field] + 3]).all()
        for field in ("velocity_linear", "velocity_angular")
    ):
        raise ValueError("Nonfinite joint input velocity")
    poses = a["Poses"].view(np.float32).reshape(-1, l["pose_words"])
    local = a["LocalMass"].view(np.float32).reshape(-1, l["local_words"])
    world = a["WorldMass"].view(np.float32).reshape(-1, l["world_words"])
    q = poses[:, l["pose_rotation"] : l["pose_rotation"] + 4]
    if (
        not np.isfinite(q).all()
        or not np.isfinite(
            poses[:, l["pose_translation"] : l["pose_translation"] + 3]
        ).all()
        or np.any(np.abs(np.linalg.norm(q, axis=1) - 1) > 2e-6)
        or any(
            not np.isfinite(data[:, l[field] : l[field] + width]).all()
            for data, field, width in [
                (local, "local_com", 3),
                (local, "local_inertia_frame", 4),
                (local, "local_inv_inertia", 3),
                (local, "local_inv_mass", 3),
                (world, "world_inertia", 16),
                (world, "world_com", 3),
                (world, "world_inv_mass", 3),
            ]
        )
    ):
        raise ValueError("Invalid joint body pose or mass")
    if np.any(local[:, l["local_inv_mass"] : l["local_inv_mass"] + 3] < 0) or np.any(
        local[:, l["local_inv_inertia"] : l["local_inv_inertia"] + 3] < 0
    ):
        raise ValueError("Negative joint inverse mass")
    p = a["Params"].view(np.float32)
    pf = l["rbd_params"]
    if (
        not np.isfinite(p).all()
        or p[pf["dt"]] < 0
        or p[pf["joint_natural_frequency"]] < 0
        or p[pf["joint_damping_ratio"]] <= 0
    ):
        raise ValueError("Invalid joint softness or timestep")
    joints = a["Joints"].reshape(-1, l["impulse_joint_words"])
    jf, gf, mf = l["impulse_joint"], l["generic_joint"], l["motor"]

    def validate_data(j):
        f = j.view(np.float32)
        for field in ("locked_axes", "limit_axes", "motor_axes", "coupled_axes"):
            if j[gf[field]] >= 64:
                raise ValueError("Invalid 3D joint axis mask")
        for frame in ("local_frame_a", "local_frame_b"):
            off = gf[frame]
            q = f[off + l["pose_rotation"] : off + l["pose_rotation"] + 4]
            t = f[off + l["pose_translation"] : off + l["pose_translation"] + 3]
            if (
                not np.isfinite(q).all()
                or not np.isfinite(t).all()
                or abs(float(np.linalg.norm(q)) - 1) > 2e-6
            ):
                raise ValueError("Invalid joint anchor frame")
        for axis in range(6):
            start = gf["limits"] + axis * l["limits_words"]
            lim = f[start : start + l["limits_words"]]
            start = gf["motors"] + axis * l["motor_words"]
            motor = j[start : start + l["motor_words"]]
            values = motor.view(np.float32)
            if (
                not np.isfinite(lim).all()
                or lim[l["limits"]["min"]] > lim[l["limits"]["max"]]
            ):
                raise ValueError("Invalid joint limits")
            if (
                motor[mf["model"]] > 1
                or not np.isfinite(np.delete(values, mf["model"])).all()
                or np.any(values[[mf["stiffness"], mf["damping"], mf["max_force"]]] < 0)
            ):
                raise ValueError("Invalid joint motor")

    for i, row in enumerate(joints[:n]):
        if (
            row[jf["body_a"]] >= 2 * c["joints_per_batch"]
            or row[jf["body_b"]] >= 2 * c["joints_per_batch"]
            or row[jf["body_a"]] == row[jf["body_b"]]
        ):
            raise ValueError("Invalid native joint body IDs")
        validate_data(row[jf["data"] : jf["data"] + l["generic_joint_words"]])
    if c["kind"] in ("joint_update", "joint_orthogonalize", "joint_solve"):
        builders = a["Builders"].reshape(-1, l["builder_words"])
        bf = l["builder"]
        for i, row in enumerate(builders[:n]):
            if (
                row[bf["body1"]] >= c["bodies"]
                or row[bf["body2"]] >= c["bodies"]
                or row[bf["constraint_id"]] != i
            ):
                raise ValueError("Invalid joint builder identity")
            validate_data(row[bf["joint"] : bf["joint"] + l["generic_joint_words"]])
    if c["kind"] == "joint_program":
        starts = [0] + list(map(int, a["ColorGroups"][:2]))
        ends = list(map(int, a["ColorGroups"][:3]))
        for start, end in zip(starts, ends):
            used = set()
            for i in range(start * nb, end * nb):
                for field in ("body_a", "body_b"):
                    slot = int(joints[i, jf[field]]) * nb + i % nb
                    if slot in used:
                        raise ValueError("Joint color has overlapping velocity writers")
                    used.add(slot)
    if c["kind"] in ("joint_solve", "joint_orthogonalize"):
        validate_constraints(l, c, a)
    return sizes


def validate_constraints(l, c, a):
    import numpy as np

    con = a["Constraints"].reshape(-1, l["joint_constraint_words"])
    cf, ef = l["joint_constraint"], l["joint_element"]
    for row in con[: c["count"]]:
        length = int(row[cf["len"]])
        if (
            length > 12
            or row[cf["solver_vel_a"]] >= c["bodies"]
            or row[cf["solver_vel_b"]] >= c["bodies"]
            or row[cf["solver_vel_a"]] == row[cf["solver_vel_b"]]
        ):
            raise ValueError("Invalid prepared joint row count or body ID")
        for k in range(length):
            start = cf["elements"] + k * l["joint_element_words"]
            el = row[start : start + l["joint_element_words"]]
            f = el.view(np.float32)
            if (
                not np.isfinite(np.delete(f, ef["joint_id"])).all()
                or el[ef["joint_id"]] >= c["count"]
                or f[ef["impulse_bounds"]] > f[ef["impulse_bounds"] + 1]
            ):
                raise ValueError("Invalid prepared joint element")
    # Rust's host coloring treats both static and dynamic endpoints as writers.
    # Its solver stores both slots even when no active row changes them.
    starts = [0] + list(map(int, a["ColorGroups"][:2]))
    ends = list(map(int, a["ColorGroups"][:3]))
    nb = c["batches"]
    for start, end in zip(starts, ends):
        used = set()
        for row in con[start * nb : end * nb]:
            for field in ("solver_vel_a", "solver_vel_b"):
                slot = int(row[cf[field]])
                if slot in used:
                    raise ValueError("Joint color has overlapping velocity writers")
                used.add(slot)


def output_masks(l, c, b, e):
    import numpy as np

    masks = {n: np.zeros(v.size, np.uint8) for n, v in b.items()}
    op = c["kind"]
    n = c["count"]
    cf, ef = l["joint_constraint"], l["joint_element"]
    if op in ("joint_init", "joint_program"):
        masks["Builders"][: n * l["builder_words"]] = 2
        for i in range(n):
            for f in ("local_frame_a", "local_frame_b"):
                off = (
                    i * l["builder_words"]
                    + l["builder"]["joint"]
                    + l["generic_joint"][f]
                    + l["pose_translation"]
                )
                masks["Builders"][off : off + 3] = 1
            base = i * l["joint_constraint_words"]
            for field, width in (
                ("im_a", 3),
                ("im_b", 3),
                ("solver_vel_a", 1),
                ("solver_vel_b", 1),
                ("len", 1),
            ):
                masks["Constraints"][base + cf[field] : base + cf[field] + width] = 2
    if op == "joint_motor_params":
        masks["MotorParams"][: n * 36] = 1
        for i in range(n * 6):
            masks["MotorParams"][i * 6 + 3 : i * 6 + 5] = 2
    if op in ("joint_update", "joint_orthogonalize", "joint_program"):
        for i in range(n):
            base = i * l["joint_constraint_words"]
            length = int(e["Constraints"][base + cf["len"]])
            if not 0 <= length <= 12:
                raise ValueError("Invalid expected native joint length")
            if op != "joint_orthogonalize":
                masks["Constraints"][base + cf["len"]] = 2
            for k in range(length):
                eb = base + cf["elements"] + k * l["joint_element_words"]
                if op == "joint_orthogonalize":
                    fields = [
                        ("inv_lhs", 1),
                        ("cfm_gain", 1),
                        ("lin_jac", 3),
                        ("ang_jac_a", 3),
                        ("ang_jac_b", 3),
                        ("ii_ang_jac_a", 3),
                        ("ii_ang_jac_b", 3),
                        ("rhs_wo_bias", 1),
                        ("rhs", 1),
                    ]
                    for f, w in fields:
                        masks["Constraints"][eb + ef[f] : eb + ef[f] + w] = 1
                else:
                    masks["Constraints"][eb : eb + l["joint_element_words"]] = 1
                    masks["Constraints"][eb + ef["joint_id"]] = 2
                    for d in (0, 1):
                        idx = eb + ef["impulse_bounds"] + d
                        if e["Constraints"][idx] in (
                            0,
                            l["constants"]["max_flt_bits"],
                            l["constants"]["max_flt_bits"] ^ 0x80000000,
                        ):
                            masks["Constraints"][idx] = 2
    if op in ("joint_solve", "joint_program"):
        color = int(b["Color"][0])
        nb = c["batches"]
        start = int(b["ColorGroups"][color - 1]) * nb if color else 0
        end = int(b["ColorGroups"][color]) * nb
        for i in range(start, end):
            base = i * l["joint_constraint_words"]
            row = e["Constraints"][base : base + l["joint_constraint_words"]]
            for k in range(int(row[cf["len"]])):
                masks["Constraints"][
                    base + cf["elements"] + k * l["joint_element_words"] + ef["impulse"]
                ] = 1
            for field in ("solver_vel_a", "solver_vel_b"):
                slot = int(row[cf[field]])
                for f in ("velocity_linear", "velocity_angular"):
                    off = slot * l["velocity_words"] + l[f]
                    masks["Velocities"][off : off + 3] = 1
    return masks


def check_outputs(l, c, b, a, e, k):
    import numpy as np

    validate_case(l, c, b)
    for r in (a, e):
        if set(r) != set(b) or any(
            v.dtype != np.uint32 or v.shape != b[n].shape for n, v in r.items()
        ):
            raise ValueError("Invalid returned joint buffers")
    if c["kind"] in (
        "joint_update",
        "joint_orthogonalize",
        "joint_solve",
        "joint_program",
    ):
        for r in (a, e):
            validate_constraints(l, c, r)
    if c["kind"] in ("joint_solve", "joint_program"):
        color = int(b["Color"][0])
        start = int(b["ColorGroups"][color - 1]) if color else 0
        end = int(b["ColorGroups"][color])
        cf, ef = l["joint_constraint"], l["joint_element"]
        for result in (a, e):
            rows = result["Constraints"].reshape(-1, l["joint_constraint_words"])
            for row in rows[start * c["batches"] : end * c["batches"]]:
                for i in range(int(row[cf["len"]])):
                    off = cf["elements"] + i * l["joint_element_words"]
                    el = row[off : off + l["joint_element_words"]].view(np.float32)
                    if (
                        not el[ef["impulse_bounds"]]
                        <= el[ef["impulse"]]
                        <= el[ef["impulse_bounds"] + 1]
                    ):
                        raise AssertionError("Solved joint impulse outside bounds")
    masks = output_masks(l, c, b, e)
    errors = {}
    for name, m in masks.items():
        for r in (a, e):
            np.testing.assert_array_equal(
                r[name][m == 0], b[name][m == 0], err_msg=f"Joint guard/readonly {name}"
            )
        np.testing.assert_array_equal(
            a[name][m == 2],
            e[name][m == 2],
            err_msg=f"Joint integer/copied field {name}",
        )
        af, ef = a[name].view(np.float32)[m == 1], e[name].view(np.float32)[m == 1]
        if not np.isfinite(af).all() or not np.isfinite(ef).all():
            raise AssertionError("Nonfinite joint arithmetic")
        np.testing.assert_allclose(
            af, ef, atol=k["atol"], rtol=k["rtol"], err_msg=f"Joint floats {name}"
        )
        errors[name] = float(np.max(np.abs(af - ef))) if af.size else 0.0
    return errors
