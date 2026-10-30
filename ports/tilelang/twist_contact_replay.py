#!/usr/bin/env python3
"""Replay fresh/recycled twist contact preparation against native Rust kernels."""

from contact_replay import BUFFERS, check_outputs, validate_case
from stage_replay import parse_args, run_stage


def main():
    from contact_kernels import make_kernel

    run_stage(
        parse_args(["twist_prepare"], "twist_prepare"),
        oracle="twist-contacts",
        buffers=BUFFERS,
        make_kernel=make_kernel,
        validate_case=validate_case,
        check_outputs=check_outputs,
        launch=lambda c: (
            max((c.get("launch_capacity", c["count"]) + 63) // 64, 1),
            64,
            c["iterations"],
        ),
        extra_files=("ContactPlan.bin", "RecycleOffsets.bin"),
    )


if __name__ == "__main__":
    main()
