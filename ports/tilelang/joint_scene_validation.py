"""Composition checks for connected joint scenes and device trajectory storage."""

import joint_validation as joints
import numpy as np
import scene_numeric as numeric_policy
import scene_outputs as outputs
import scene_validation as base
from scene_trajectory import check_trajectory

EXTRA = (
    "Joints",
    "Builders",
    "JointConstraints",
    "JointColors",
    "MotorParams",
    "JointBias",
    "JointUnbiased",
)
BUFFERS = base.BUFFERS + EXTRA
SCENES = (
    "joint_fixed",
    "joint_ball",
    "joint_revolute",
    "joint_prismatic",
    "joint_motor",
    "joint_impact",
    "joint_chain",
)


def validate_case(l, c, b, *, capacity_variant=False):
    core = {n: b[n] for n in base.BUFFERS}
    bc = dict(
        c,
        kind="analytic_scene",
        scene="free",
        buffer_words={n: c["buffer_words"][n] for n in core},
    )
    base.validate_case(l, bc, core, capacity_variant=capacity_variant)
    if c["kind"] != "joint_scene" or c["scene"] not in SCENES:
        raise ValueError("Invalid joint scene")
    n = c["joint_count"]
    nb = c["batches"]
    js = c["joint_storage"]
    if (
        n != c["joints_per_batch"] * nb
        or js != n + 3
        or c["joint_color_count"] != 3
        or c["joints_per_batch"] != c["columns"] * c["levels_scene"]
    ):
        raise ValueError("Invalid scene joint capacity")
    sizes = {
        "Joints": js * l["impulse_joint_words"],
        "Builders": js * l["builder_words"],
        "JointConstraints": js * l["joint_constraint_words"],
        "JointColors": 4,
        "MotorParams": js * 36,
        "Trajectory": c["frames"] * c["trace_stride"] + 3,
        "FrameIndex": 2,
        "JointBias": 2,
        "JointUnbiased": 2,
    }
    if c["trace_stride"] != c["body_count"] * (l["pose_words"] + l["velocity_words"]):
        raise ValueError("Invalid trajectory stride")
    if set(b) != set(BUFFERS) or any(
        c["buffer_words"][name] != size
        or b[name].shape != (size,)
        or b[name].dtype != np.uint32
        for name, size in sizes.items()
    ):
        raise ValueError("Invalid joint scene buffer ABI")
    if b["FrameIndex"][0] != 0 or b["JointBias"][0] != 1 or b["JointUnbiased"][0] != 0:
        raise ValueError("Invalid joint frame/bias state")
    groups = b["JointColors"][:3].astype(np.int64)
    if np.any(groups[1:] < groups[:-1]) or groups[-1] != c["joints_per_batch"]:
        raise ValueError("Invalid joint color prefix ranges")
    row = b["Joints"].reshape(-1, l["impulse_joint_words"])
    jf, gf = l["impulse_joint"], l["generic_joint"]
    # Fixed fixture topology and masks make the independent physical checks
    # meaningful; bounds alone would accept a different experiment.
    locked = {
        "joint_fixed": 63,
        "joint_ball": 7,
        "joint_revolute": 31,
        "joint_prismatic": 62,
        "joint_motor": 62,
        "joint_impact": 63,
        "joint_chain": 7,
    }[c["scene"]]
    expected_pairs = {
        (
            body - 1
            if c["scene"] == "joint_chain" and (body - 1) % c["levels_scene"]
            else 0,
            body,
        )
        for body in range(1, c["colliders_per_batch"])
    }
    for batch in range(nb):
        observed = set()
        for i in range(batch, n, nb):
            a, bb = int(row[i, jf["body_a"]]), int(row[i, jf["body_b"]])
            observed.add((a, bb))
            data = row[i, jf["data"] : jf["data"] + l["generic_joint_words"]]
            if (
                data[gf["locked_axes"]] != locked
                or data[gf["limit_axes"]] != (c["scene"] == "joint_prismatic")
                or data[gf["motor_axes"]] != (c["scene"] == "joint_motor")
                or data[gf["coupled_axes"]] != 0
            ):
                raise ValueError("Joint fixture axes changed")
            f = data.view(np.float32)
            for field in ("local_frame_a", "local_frame_b"):
                start = gf[field]
                q = f[start + l["pose_rotation"] : start + l["pose_rotation"] + 4]
                if (
                    not np.array_equal(q, np.array([0, 0, 0, 1], np.float32))
                    or not np.isfinite(
                        f[
                            start + l["pose_translation"] : start
                            + l["pose_translation"]
                            + 3
                        ]
                    ).all()
                ):
                    raise ValueError("Invalid fixture joint anchor")
            if c["scene"] == "joint_motor":
                m = gf["motors"]
                mf = l["motor"]
                if data[m + mf["model"]] != 0 or any(
                    f[m + mf[key]] != np.float32(value)
                    for key, value in {
                        "target_pos": 0.25,
                        "target_vel": 0,
                        "stiffness": 40,
                        "damping": 8,
                        "max_force": 10,
                    }.items()
                ):
                    raise ValueError("Fixture motor changed")
            if c["scene"] == "joint_prismatic":
                off = gf["limits"]
                if f[off + l["limits"]["min"]] != np.float32(-0.4) or f[
                    off + l["limits"]["max"]
                ] != np.float32(0.4):
                    raise ValueError("Fixture limits changed")
        if observed != expected_pairs:
            raise ValueError("Fixture joint topology changed")
    for start, end in zip([0, *groups[:2]], groups):
        used = set()
        for i in range(int(start) * nb, int(end) * nb):
            for field in ("body_a", "body_b"):
                slot = int(row[i, jf[field]]) * nb + i % nb
                if slot in used:
                    raise ValueError("Joint scene color has overlapping writers")
                used.add(slot)
    p = b["RbdParams"].view(np.float32)
    pf = l["rbd_params"]
    if p[pf["joint_natural_frequency"]] < 0 or p[pf["joint_damping_ratio"]] <= 0:
        raise ValueError("Invalid joint scene softness")


def check_checkpoint(l, c, b, a, e, k):
    for r in (a, e):
        if set(r) != set(b) or any(
            v.dtype != np.uint32 or v.shape != b[n].shape for n, v in r.items()
        ):
            raise ValueError("Invalid returned scene storage")
    bc, ac, ec = ({n: r[n] for n in base.BUFFERS} for r in (b, a, e))
    outputs.check_guards(l, c, bc, ac, ec)
    errors = outputs.check_bodies(l, c, bc, ac, ec, k)
    errors.update(outputs.check_collision(l, c, bc, ac, ec, k))
    count = outputs.check_graph(l, c, ac, ec)
    errors.update(outputs.check_constraints(l, c, ac, ec, k, count))
    errors.update(outputs.check_states(l, c, ac, ec, k, count))
    errors.update(check_joint_rows(l, c, b, a, e, k))
    return errors


def check_joint_rows(l, c, b, a, e, k):
    """Prepared ordinary joint identities, row bounds, padding and numeric fields."""
    errors = {}
    for name in ("Joints", "JointColors", "MotorParams", "JointBias", "JointUnbiased"):
        for r in (a, e):
            np.testing.assert_array_equal(r[name], b[name], err_msg=f"Readonly {name}")
    jc = dict(c, kind="joint_program", count=c["joint_count"], bodies=c["body_storage"])

    def joint_arrays(arr):
        return {
            "Builders": arr["Builders"],
            "Constraints": arr["JointConstraints"],
            "Velocities": arr["SolverVelocities"],
            "Color": b["Color0"],
            "ColorGroups": b["JointColors"],
        }

    jb, ja, je = map(joint_arrays, (b, a, e))
    for r in (ja, je):
        joints.validate_constraints(l, jc, r)
    masks = joints.output_masks(l, jc, jb, je)
    for name in ("Builders", "Constraints"):
        m = masks[name]
        for r in (ja, je):
            np.testing.assert_array_equal(
                r[name][m == 0], jb[name][m == 0], err_msg=f"Joint scene guard {name}"
            )
        np.testing.assert_array_equal(
            ja[name][m == 2], je[name][m == 2], err_msg=f"Joint scene identity {name}"
        )
        af, ef = (r[name].view(np.float32)[m == 1] for r in (ja, je))
        if not np.isfinite(af).all() or not np.isfinite(ef).all():
            raise AssertionError("Nonfinite joint scene row")
        groups = {"Joint" + name: m == 1}
        if name == "Constraints":
            # Row impulses and velocity RHS can amplify tiny contact feedback;
            # a scoped allowance must not loosen Jacobians or effective masses.
            policies = numeric_policy.policy_for(c, k).get("fields", {})
            cf, element_fields = l["joint_constraint"], l["joint_element"]
            for label, fields in (
                ("JointImpulses", ("impulse",)),
                ("JointRhs", ("rhs", "rhs_wo_bias")),
            ):
                if label not in policies:
                    continue
                selection = np.zeros(m.size, bool)
                for i, row in enumerate(
                    je[name].reshape(-1, l["joint_constraint_words"])[
                        : c["joint_count"]
                    ]
                ):
                    for j in range(int(row[cf["len"]])):
                        start = (
                            i * l["joint_constraint_words"]
                            + cf["elements"]
                            + j * l["joint_element_words"]
                        )
                        for field in fields:
                            selection[start + element_fields[field]] = True
                selection &= m == 1
                groups["JointConstraints"] &= ~selection
                groups[label] = selection
        for label, selection in groups.items():
            np.testing.assert_allclose(
                ja[name].view(np.float32)[selection],
                je[name].view(np.float32)[selection],
                atol=numeric_policy.tolerances(c, k, label)[0],
                rtol=numeric_policy.tolerances(c, k, label)[1],
                err_msg=f"Joint scene floats {label}",
            )
        errors["Joint" + name] = float(np.max(np.abs(af - ef))) if len(af) else 0.0
    cf, ef = l["joint_constraint"], l["joint_element"]
    for r in (ja, je):
        for row in r["Constraints"].reshape(-1, l["joint_constraint_words"])[
            : c["joint_count"]
        ]:
            for i in range(int(row[cf["len"]])):
                offset = cf["elements"] + i * l["joint_element_words"]
                f = row[offset : offset + l["joint_element_words"]].view(np.float32)
                if (
                    not f[ef["impulse_bounds"]]
                    <= f[ef["impulse"]]
                    <= f[ef["impulse_bounds"] + 1]
                ):
                    raise AssertionError("Scene joint impulse outside bounds")
    return errors


def check_outputs(l, c, b, a, e, k):
    validate_case(l, c, b)
    errors = check_checkpoint(l, c, b, a, e, k)
    errors["Trajectory"] = check_trajectory(l, c, b, a, e, k)
    from joint_scene_physics import check

    for r in (a, e):
        check(l, c, b, r, k)
    return errors
