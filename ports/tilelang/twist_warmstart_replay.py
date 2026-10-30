#!/usr/bin/env python3
"""Replay compact twist cache scaling, ordered gather and independent hub slots."""

from stage_replay import parse_args, run_stage
from warmstart_replay import BUFFERS, OPERATIONS, check_outputs, validate_case


def main():
    from warmstart_kernels import make_kernel

    run_stage(
        parse_args(["twist_" + op for op in OPERATIONS], "twist_gather_warmstart"),
        oracle="twist-warmstarts",
        buffers=BUFFERS,
        make_kernel=make_kernel,
        validate_case=validate_case,
        check_outputs=check_outputs,
        launch=lambda case: (
            max((case["dispatch"] + 63) // 64, 1),
            64,
            case["iterations"],
        ),
    )


if __name__ == "__main__":
    main()
