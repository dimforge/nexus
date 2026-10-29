"""Live native 3D rigid-joint construction, orthogonalization and solve replay."""

from joint_validation import BUFFERS, OPERATIONS, check_outputs, validate_case


def main():
    from joint_kernels import make_program
    from stage_replay import parse_args, run_stage

    run_stage(
        parse_args(OPERATIONS, "joint_program"),
        oracle="joints",
        buffers=BUFFERS,
        make_kernel=None,
        make_program=make_program,
        validate_case=validate_case,
        check_outputs=check_outputs,
        launch=None,
    )


if __name__ == "__main__":
    main()
