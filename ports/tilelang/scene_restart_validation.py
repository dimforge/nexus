"""Identical-state single-frame checks for contact-sensitive native trajectories."""

import general_scene_validation as general
import joint_scene_validation as joint
import numpy as np
import scene_numeric as numeric_policy
import scene_outputs
import scene_validation as base


def validator(c):
    if c["kind"] == "general_scene":
        return general
    if c["kind"] == "joint_scene":
        return joint
    raise ValueError("Unsupported restart scene kind")


def validate_core(l, c, b):
    if (
        type(c.get("start_frame")) is not int
        or not 0 <= c["start_frame"] < c["frames"]
        or c["iterations"] != 1
    ):
        raise ValueError("Invalid restart frame")
    if int(b["FrameIndex"][0]) != c["start_frame"]:
        raise ValueError("Restart frame index mismatch")
    # These counters are cleared before use by the frame program. Normalize them
    # only for the shared fresh-scene input schema; the actual restart is untouched.
    normalized = dict(b)
    for name in ("Plan", "PairCount", "PfmCount", "FrameIndex"):
        normalized[name] = b[name].copy()
        normalized[name][0] = 0
    normalized["Plan"][:] = 0
    validator(c).validate_case(l, dict(c, iterations=c["frames"]), normalized)
    if c["start_frame"]:
        if (
            b["Plan"][l["plan"]["bound"]] > c["contact_capacity"]
            or b["PairCount"][0] > c["pair_capacity"]
            or b["PfmCount"][0] > c["pair_capacity"]
        ):
            raise ValueError("Restart published count exceeds capacity")
        core = {n: b[n] for n in base.BUFFERS}
        scene_outputs.check_graph(l, c, core, core)
        numeric_policy.constraint_geometry(l, c, b)


def check_core(l, c, b, a, e, k):
    validate_core(l, c, b)
    # Collision's broad-phase checker expects the reset value, not the preceding
    # frame's count. Preserve every guard word and all actual kernel inputs.
    before = dict(b)
    before["PairCount"] = b["PairCount"].copy()
    before["PairCount"][0] = 0
    errors = validator(c).check_checkpoint(l, c, before, a, e, k)
    frame = c["start_frame"]
    stride = c["trace_stride"]
    n = c["body_count"]
    pw = l["pose_words"]
    vw = l["velocity_words"]
    start = frame * stride
    end = start + stride
    for r in (a, e):
        np.testing.assert_array_equal(
            r["FrameIndex"], np.array([frame + 1, b["FrameIndex"][1]], np.uint32)
        )
        np.testing.assert_array_equal(r["Trajectory"][:start], b["Trajectory"][:start])
        np.testing.assert_array_equal(r["Trajectory"][end:], b["Trajectory"][end:])
        np.testing.assert_array_equal(
            r["Trajectory"][start:end],
            np.r_[r["Poses"][: n * pw], r["Velocities"][: n * vw]],
        )
        # The independent physics check sees just the replayed frame and its starting
        # state, so it cannot accidentally evaluate an unrecorded future trajectory.
        single = dict(
            r,
            Trajectory=np.r_[r["Trajectory"][start:end], b["Trajectory"][-3:]],
            FrameIndex=np.array([1, b["FrameIndex"][1]], np.uint32),
        )
        if c["kind"] == "general_scene":
            from general_scene_physics import check
        else:
            from joint_scene_physics import check
        check(l, dict(c, frames=1, iterations=1), b, single, k)
    errors["Trajectory"] = float(
        np.max(
            abs(
                a["Trajectory"][start:end].view(np.float32)
                - e["Trajectory"][start:end].view(np.float32)
            ),
            initial=0,
        )
    )
    return errors


def validate_case(l, c, b):
    if c.get("broadphase") != "lbvh":
        return validate_core(l, c, b)
    import lbvh_scene_validation as tree
    from scene_broadphase import EXTRA

    core = validator(c).BUFFERS
    if set(b) != set(core) | set(EXTRA):
        raise ValueError("Invalid LBVH restart buffers")
    validate_core(l, tree.base_case(c, core), {n: b[n] for n in core})
    tree.validate_tree(l, c, b, restarted=True)


def check_outputs(l, c, b, a, e, k):
    validate_case(l, c, b)
    if c.get("broadphase") != "lbvh":
        return check_core(l, c, b, a, e, k)
    import lbvh_scene_validation as tree

    core = validator(c).BUFFERS
    errors = check_core(
        l, tree.base_case(c, core), *({n: r[n] for n in core} for r in (b, a, e)), k
    )
    tree.check_trees(l, c, b, a, e, k, restarted=True)
    return errors
