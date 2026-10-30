"""Replay the full resident LBVH broad phase against native Rust kernels."""

from lbvh_program_validation import BUFFERS, check_outputs, validate_case


def main():
    from lbvh_program_kernels import make_program
    from stage_replay import parse_args, run_stage

    run_stage(
        parse_args(["lbvh_program"], "lbvh_program"),
        oracle="lbvh",
        buffers=BUFFERS,
        make_kernel=None,
        make_program=make_program,
        validate_case=validate_case,
        check_outputs=check_outputs,
        launch=None,
    )


if __name__ == "__main__":
    main()
