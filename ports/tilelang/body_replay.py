#!/usr/bin/env python3
"""Replay body stages and no-contact composition against live Rust entry points."""

from body_validation import (
    BUFFERS,
    OPERATIONS,
    dispatch_count,
    validate_case,
    check_outputs,
)
from stage_replay import parse_args, run_stage


def main():
    from body_kernels import make_kernel

    run_stage(
        parse_args(OPERATIONS, "free_step"),
        oracle="bodies",
        buffers=BUFFERS,
        make_kernel=make_kernel,
        validate_case=validate_case,
        check_outputs=check_outputs,
        launch=lambda case: (
            max((dispatch_count(case) + 63) // 64, 1),
            64,
            case["iterations"],
        ),
        extra_files=("RbdParams.bin", "BatchIndices.bin", "Gravity.bin"),
    )


if __name__ == "__main__":
    main()
