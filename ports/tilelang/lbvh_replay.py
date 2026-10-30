"""Replay native LBVH stages and verify tree geometry and pair completeness."""

from lbvh_validation import BUFFERS, OPERATIONS, check_outputs, validate_case


def main():
    from lbvh_kernels import dimensions, make_kernel
    from stage_replay import parse_args, run_stage

    run_stage(
        parse_args(OPERATIONS, "lbvh_domain"),
        oracle="lbvh",
        buffers=BUFFERS,
        make_kernel=make_kernel,
        validate_case=validate_case,
        check_outputs=check_outputs,
        launch=lambda c: (*dimensions(c)[:2], 1),
    )


if __name__ == "__main__":
    main()
