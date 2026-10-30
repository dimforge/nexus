"""All-frame byte identity and numeric checks shared by scene families."""

import numpy as np
import scene_numeric as numeric_policy


def check_trajectory(l, c, b, a, e, k):
    overrides = numeric_policy.policy_for(c, k).get("trajectory_frames", {})
    for frame, fields in overrides.items():
        if (
            not frame.isdecimal()
            or str(int(frame)) != frame
            or not 1 <= int(frame) <= c["frames"]
            or not set(fields) <= {"TrajectoryPoses", "TrajectoryVelocities"}
        ):
            raise ValueError("Invalid scoped trajectory frame")
    n = c["body_count"]
    pw = l["pose_words"]
    vw = l["velocity_words"]
    stride = c["trace_stride"]
    mask = np.zeros(a["Trajectory"].size, bool)
    pose_mask = mask.copy()
    for frame in range(c["frames"]):
        for i in range(n):
            for field, width in (("pose_translation", 3), ("pose_rotation", 4)):
                off = frame * stride + i * pw + l[field]
                mask[off : off + width] = True
                pose_mask[off : off + width] = True
            for field in ("velocity_linear", "velocity_angular"):
                off = frame * stride + n * pw + i * vw + l[field]
                mask[off : off + 3] = True
    for r in (a, e):
        np.testing.assert_array_equal(
            r["FrameIndex"], np.array([c["frames"], b["FrameIndex"][1]], np.uint32)
        )
        np.testing.assert_array_equal(r["Trajectory"][-3:], b["Trajectory"][-3:])
        # The last recorded state must be exactly the published current state.
        np.testing.assert_array_equal(
            r["Trajectory"][(c["frames"] - 1) * stride : c["frames"] * stride],
            np.r_[r["Poses"][: n * pw], r["Velocities"][: n * vw]],
        )
    np.testing.assert_array_equal(a["Trajectory"][~mask], e["Trajectory"][~mask])
    af, ef = (r["Trajectory"].view(np.float32)[mask] for r in (a, e))
    for field, selection in [
        ("TrajectoryPoses", pose_mask),
        ("TrajectoryVelocities", mask & ~pose_mask),
    ]:
        for frame in range(c["frames"]):
            start, end = frame * stride, (frame + 1) * stride
            limits = overrides.get(str(frame + 1), {}).get(field)
            atol, rtol = (
                numeric_policy.tolerances(c, k, field)
                if limits is None
                else (limits["atol"], limits["rtol"])
            )
            chosen = selection[start:end]
            np.testing.assert_allclose(
                a["Trajectory"].view(np.float32)[start:end][chosen],
                e["Trajectory"].view(np.float32)[start:end][chosen],
                atol=atol,
                rtol=rtol,
                err_msg=f"{field} frame {frame + 1}",
            )
    return float(np.max(np.abs(af - ef))) if len(af) else 0.0
