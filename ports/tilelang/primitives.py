#!/usr/bin/env python3
"""Replay irregular integer operations against the actual Rust GPU entry function."""

from replay import HERE as HERE
from stage_replay import parse_args, run_stage

BUFFERS = ("Counts", "Ids", "Sorted")


def validate_case(case, arrays):
    import numpy as np

    rows, storage = case["rows"], case["storage"]
    if not 0 <= rows <= 65535 or not 1 <= storage <= 2**31 - 64:
        raise ValueError("Invalid primitive dimensions")
    if arrays["Counts"].size != max(rows, 1) or any(
        arrays[n].size != storage for n in ("Ids", "Sorted")
    ):
        raise ValueError("Invalid primitive buffer sizes")
    ends = arrays["Counts"][:rows].astype(np.uint64)
    if (
        np.any(ends[1:] < ends[:-1])
        or (ends.size and ends[-1] != case["count"])
        or not 0 <= case["count"] <= storage
        or (not rows and case["count"])
    ):
        raise ValueError("Invalid cumulative row ends")


def check_outputs(layout, case, before, actual, expected, contract):
    import numpy as np

    validate_case(case, before)
    errors = {}
    for name in BUFFERS:
        reference = expected[name] if name in contract["outputs"] else before[name]
        np.testing.assert_array_equal(
            actual[name], reference, err_msg=f"{case['name']}: {name}"
        )
        errors[name] = 0.0
    # Sentinel output words past the active prefix must remain exactly unchanged.
    np.testing.assert_array_equal(
        actual["Sorted"][case["count"] :], before["Sorted"][case["count"] :]
    )
    return errors


def main():
    from primitive_kernels import make_kernel

    run_stage(
        parse_args(["stabilize"], "stabilize"),
        oracle="primitives",
        buffers=BUFFERS,
        make_kernel=lambda layout, case, target: make_kernel(case, target),
        validate_case=lambda layout, case, arrays: validate_case(case, arrays),
        check_outputs=check_outputs,
        launch=lambda case: (max(case["rows"], 1), 64, 1),
    )


if __name__ == "__main__":
    main()
