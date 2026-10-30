#!/usr/bin/env python3
"""Native twist color passes, shared static bodies and cached warmstart replay."""

from colored_replay import BUFFERS, OPERATIONS, check_outputs, validate_case
from stage_replay import parse_args, run_stage


def main():
    from colored_kernels import make_program

    run_stage(
        parse_args(["twist_" + op for op in OPERATIONS], "twist_colored_program"),
        oracle="twist-colored",
        buffers=BUFFERS,
        make_kernel=None,
        make_program=make_program,
        validate_case=validate_case,
        check_outputs=check_outputs,
        launch=lambda case: (case["groups"], 64, case["iterations"]),
    )


if __name__ == "__main__":
    main()
