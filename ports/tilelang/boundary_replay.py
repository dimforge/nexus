"""Retained zero-distance stress with independently bounded CFM branch choices."""

from boundary_validation import BUFFERS, check_outputs, classify, validate_case


def main():
    from kernels import make_kernel
    from replay import BUFFERS as SOLVER
    from stage_replay import parse_args, run_stage

    args = parse_args(["solve_boundary"], "solve_boundary")

    def make_program(layout, case, target):
        return [
            {
                "name": "solve",
                "kernel": make_kernel(layout, case, target),
                "buffer_map": {n: n for n in SOLVER},
                "blocks": max((case["count"] + 63) // 64, 1),
                "threads": 64,
            }
        ]

    run_stage(
        args,
        oracle="boundary",
        buffers=BUFFERS,
        make_kernel=None,
        make_program=make_program,
        validate_case=validate_case,
        check_outputs=check_outputs,
        diagnostics=classify,
        extra_files=["RbdParams.bin"],
        launch=None,
    )


if __name__ == "__main__":
    main()
