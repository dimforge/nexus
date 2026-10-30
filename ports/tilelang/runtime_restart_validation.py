"""One complete frame from a live native runtime snapshot, with strict history guards."""

import numpy as np
import runtime_scene_validation as runtime
import scene_numeric


def validate_case(l, c, b):
    frame = c.get("start_frame")
    if type(frame) is not int or not 0 <= frame < c["frames"] or c["iterations"] != 1:
        raise ValueError("Invalid native runtime restart frame")
    if int(b["FrameIndex"][0]) != frame:
        raise ValueError("Native runtime restart index mismatch")
    normalized = dict(b, HubCounts=b["HubCounts"].copy())
    for field in ("hubs", "slots"):
        normalized["HubCounts"][l["hub"][field]] = 0
    runtime.validate_case(l, c, normalized)
    if frame:
        from scene_hub_validation import check
        from scene_outputs import check_graph

        if (
            b["PairCount"][0] > c["pair_capacity"]
            or b["PfmCount"][0] > c["pair_capacity"]
        ):
            raise ValueError("Restart pair count exceeds capacity")
        if b["Plan"][l["plan"]["bound"]] > c["contact_capacity"]:
            raise ValueError("Restart contact count exceeds capacity")
        check_graph(l, c, b, b)
        check(l, c, b)


def check_outputs(l, c, b, a, e, k):
    validate_case(l, c, b)
    # Broad phase resets its counter before use. Preserve the actual kernel input.
    normalized = dict(b, PairCount=b["PairCount"].copy())
    normalized["PairCount"][0] = 0
    errors = runtime.check_runtime_checkpoint(l, c, normalized, a, e, k)
    frame, stride, n = c["start_frame"], c["trace_stride"], c["body_count"]
    pw, vw = l["pose_words"], l["velocity_words"]
    start, end = frame * stride, (frame + 1) * stride
    for result in (a, e):
        np.testing.assert_array_equal(
            result["FrameIndex"], [frame + 1, b["FrameIndex"][1]]
        )
        np.testing.assert_array_equal(
            result["Trajectory"][:start], b["Trajectory"][:start]
        )
        np.testing.assert_array_equal(result["Trajectory"][end:], b["Trajectory"][end:])
        np.testing.assert_array_equal(
            result["Trajectory"][start:end],
            np.r_[result["Poses"][: n * pw], result["Velocities"][: n * vw]],
        )
    mask = np.zeros(stride, bool)
    for body in range(n):
        for field, width in (("pose_translation", 3), ("pose_rotation", 4)):
            off = body * pw + l[field]
            mask[off : off + width] = True
        for field in ("velocity_linear", "velocity_angular"):
            off = n * pw + body * vw + l[field]
            mask[off : off + 3] = True
    aa, ee = (result["Trajectory"][start:end] for result in (a, e))
    np.testing.assert_array_equal(aa[~mask], ee[~mask])
    scene_numeric.check(
        c, k, "RestartTrajectory", aa.view(np.float32)[mask], ee.view(np.float32)[mask]
    )
    runtime.check_states(l, c, b, a, e, [frame])
    from runtime_scene_physics import check

    for result in (a, e):
        check(l, c, b, result, k, [frame])
    errors["RestartTrajectory"] = float(
        np.max(abs(aa.view(np.float32)[mask] - ee.view(np.float32)[mask]), initial=0)
    )
    return errors
