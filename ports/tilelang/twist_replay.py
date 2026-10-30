#!/usr/bin/env python3
"""Replay native compact twist arithmetic with exact identity/guard checks."""

from stage_replay import parse_args, run_stage
from twist_validation import BUFFERS, OPERATIONS, check_outputs, validate_case


def main():
    from shaders.dynamics.twist_tiles import make_kernel

    run_stage(
        parse_args(OPERATIONS, "twist_solve"),
        oracle="twist",
        buffers=BUFFERS,
        make_kernel=make_kernel,
        validate_case=validate_case,
        check_outputs=check_outputs,
        launch=lambda case: (
            max((case["count"] + 63) // 64, 1),
            64,
            case["iterations"],
        ),
        extra_files=("RbdParams.bin",),
    )


if __name__ == "__main__":
    main()
