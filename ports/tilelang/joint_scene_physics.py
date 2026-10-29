"""Independent all-frame rigid-joint geometry and physical diagnostics."""

import numpy as np


def rotate(q, v):
    return v + 2 * np.cross(q[:3], np.cross(q[:3], v) + q[3] * v)


def multiply(a, b):
    return np.r_[
        a[3] * b[:3] + b[3] * a[:3] + np.cross(a[:3], b[:3]),
        a[3] * b[3] - np.dot(a[:3], b[:3]),
    ]


def check(l, c, b, r, k):
    from scene_physics import check_physics, state

    n, nb = c["body_count"], c["batches"]
    pw, vw = l["pose_words"], l["velocity_words"]
    stride = c["trace_stride"]
    rows = b["Joints"].reshape(-1, l["impulse_joint_words"])[: c["joint_count"]]
    jf, gf = l["impulse_joint"], l["generic_joint"]
    largest_anchor = 0.0
    largest_rotation = 0.0
    largest_limit = 0.0
    largest_energy = 0.0
    largest_penetration = 0.0
    work = 0.0
    prev_position = state(l, c, b)[0]
    for frame in range(c["frames"]):
        s = {name: value.copy() for name, value in b.items()}
        offset = frame * stride
        s["Poses"][: n * pw] = r["Trajectory"][offset : offset + n * pw]
        s["Velocities"][: n * vw] = r["Trajectory"][offset + n * pw : offset + stride]
        position, q, _velocity, _angular = state(l, c, s)
        # The motor can add mechanical energy. Bound work from its force cap
        # and measured travel, separately from passive-joint energy allowance.
        if c["scene"] == "joint_motor":
            work += 10 * float(np.sum(np.abs(position[nb:, 0] - prev_position[nb:, 0])))
        ck = dict(
            k,
            physics=dict(
                k["physics"],
                energy_per_body_allowance=k["physics"]["energy_per_body_allowance"]
                + work / max(n - nb, 1),
            ),
        )
        metrics = check_physics(l, dict(c, frames=frame + 1), b, s, ck)
        largest_energy = max(largest_energy, metrics["energy"])
        largest_penetration = max(largest_penetration, metrics["penetration"])
        for i, row in enumerate(rows):
            a = int(row[jf["body_a"]]) * nb + i % nb
            body_b = int(row[jf["body_b"]]) * nb + i % nb
            data = row[jf["data"] : jf["data"] + l["generic_joint_words"]]
            floats = data.view(np.float32).astype(np.float64)
            frames = []
            for name, body in (("local_frame_a", a), ("local_frame_b", body_b)):
                off = gf[name]
                frot = floats[off + l["pose_rotation"] : off + l["pose_rotation"] + 4]
                ft = floats[
                    off + l["pose_translation"] : off + l["pose_translation"] + 3
                ]
                frames.append(
                    (multiply(q[body], frot), position[body] + rotate(q[body], ft))
                )
            qa, pa = frames[0]
            qb, pb = frames[1]
            inverse = np.r_[-qa[:3], qa[3]]
            linear = rotate(inverse, pb - pa)
            rotation = multiply(inverse, qb)
            locked = int(data[gf["locked_axes"]])
            limited = int(data[gf["limit_axes"]])
            for axis in range(3):
                if locked & (1 << axis):
                    largest_anchor = max(largest_anchor, abs(float(linear[axis])))
                if locked & (1 << (axis + 3)):
                    largest_rotation = max(largest_rotation, abs(float(rotation[axis])))
                if limited & (1 << axis):
                    off = gf["limits"] + axis * l["limits_words"]
                    lo = floats[off + l["limits"]["min"]]
                    hi = floats[off + l["limits"]["max"]]
                    largest_limit = max(
                        largest_limit,
                        float(lo - linear[axis]),
                        float(linear[axis] - hi),
                    )
        if largest_anchor > k["joint_physics"]["max_anchor_error"]:
            raise AssertionError(f"Joint anchor drift {largest_anchor}")
        if largest_rotation > k["joint_physics"]["max_locked_rotation_component"]:
            raise AssertionError(f"Joint locked rotation drift {largest_rotation}")
        if largest_limit > k["joint_physics"]["max_limit_overshoot"]:
            raise AssertionError(f"Joint limit overshoot {largest_limit}")
        prev_position = position
    if (
        c["scene"] == "joint_motor"
        and c["frames"] == 120
        and np.max(np.abs(position[nb:, 0] - 0.25))
        > k["joint_physics"]["motor_final_position_error"]
    ):
        raise AssertionError("Motor did not reach target")
    return {
        "anchor_error": largest_anchor,
        "rotation_error": largest_rotation,
        "limit_error": largest_limit,
        "maximum_energy": largest_energy,
        "maximum_penetration": largest_penetration,
        "motor_work_bound": work,
    }
