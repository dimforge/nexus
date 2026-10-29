#!/usr/bin/env python3
"""Collider pose propagation ABI, guards and live Rust replay."""

from stage_replay import parse_args, run_stage

BUFFERS = ("BodyPoses", "LocalPoses", "WorldPoses", "Parents")


def validate_case(layout, case, arrays):
    import numpy as np

    for field in (
        "count",
        "batches",
        "colliders_per_batch",
        "body_count",
        "storage",
        "body_storage",
        "iterations",
    ):
        if type(case[field]) is not int or not 0 <= case[field] <= 2**31 - 1:
            raise ValueError("Invalid pose dimensions")
    stride = layout["pose_words"]
    if type(stride) is not int or not 1 <= stride <= 1024:
        raise ValueError("Invalid pose stride")
    used = set()
    for field, width in (
        ("pose_rotation", 4),
        ("pose_translation", 3),
        ("pose_padding", 1),
    ):
        offset = layout[field]
        words = set(range(offset, offset + width))
        if not 0 <= offset <= stride - width or words & used:
            raise ValueError("Invalid pose layout")
        used |= words
    if (
        case["count"] != case["batches"] * case["colliders_per_batch"]
        or not 0 <= case["count"] <= case["storage"]
        or not 0 <= case["body_count"] <= case["body_storage"]
        or not 1 <= case["iterations"] <= 1024
        or not min(case["storage"], case["body_storage"])
    ):
        raise ValueError("Invalid pose dimensions")
    sizes = {
        "BodyPoses": case["body_storage"] * stride,
        "LocalPoses": case["storage"] * stride,
        "WorldPoses": case["storage"] * stride,
        "Parents": case["storage"],
    }
    if set(arrays) != set(BUFFERS) or any(
        a.dtype != np.uint32 or a.shape != (sizes[name],) or a.size > 2**31 - 1
        for name, a in arrays.items()
    ):
        raise ValueError("Invalid pose buffer sizes")
    if np.any(arrays["Parents"][: case["count"]] >= case["body_count"]):
        raise ValueError("Out-of-range collider parent")
    for name in ("BodyPoses", "LocalPoses"):
        a = arrays[name].view(np.float32).reshape(-1, stride)
        q = a[:, layout["pose_rotation"] : layout["pose_rotation"] + 4]
        t = a[:, layout["pose_translation"] : layout["pose_translation"] + 3]
        if (
            not np.isfinite(q).all()
            or not np.isfinite(t).all()
            or np.any(np.abs(np.linalg.norm(q, axis=1) - 1) > 2e-6)
        ):
            raise ValueError("Invalid input pose")


def check_outputs(layout, case, before, actual, expected, contract):
    import numpy as np

    validate_case(layout, case, before)
    n = case["count"]
    stride = layout["pose_words"]
    errors = {}
    for name in BUFFERS:
        if actual[name].dtype != np.uint32 or actual[name].shape != before[name].shape:
            raise ValueError("Invalid pose output size")
        if name != "WorldPoses":
            np.testing.assert_array_equal(actual[name], before[name])
            errors[name] = 0.0
        else:
            a = actual[name].reshape(-1, stride)
            e = expected[name].reshape(-1, stride)
            np.testing.assert_array_equal(a[n:], before[name].reshape(-1, stride)[n:])
            np.testing.assert_array_equal(
                a[:n, layout["pose_padding"]], e[:n, layout["pose_padding"]]
            )
            mask = np.ones(stride, dtype=bool)
            mask[layout["pose_padding"]] = False
            af = a[:n].view(np.float32)[:, mask]
            ef = e[:n].view(np.float32)[:, mask]
            if not np.isfinite(af).all() or not np.isfinite(ef).all():
                raise AssertionError("Non-finite collider pose")
            np.testing.assert_allclose(
                af, ef, atol=contract["atol"], rtol=contract["rtol"]
            )
            errors[name] = float(np.max(np.abs(af - ef))) if af.size else 0.0
    return errors


def main():
    from pose_kernels import make_kernel

    run_stage(
        parse_args(["sync_colliders"], "sync_colliders"),
        oracle="poses",
        buffers=BUFFERS,
        make_kernel=make_kernel,
        validate_case=validate_case,
        check_outputs=check_outputs,
        launch=lambda case: (
            max((case["count"] + 63) // 64, 1),
            64,
            case["iterations"],
        ),
        extra_files=("BatchIndices.bin",),
    )


if __name__ == "__main__":
    main()
