"""Live Rust projection and simplex replay with an explicit diagnostic ABI."""

from geometry_validation import OPERATIONS, buffers, check_outputs, validate_case


def main():
    from stage_replay import parse_args, run_stage

    args = parse_args(OPERATIONS, "simplex_project")
    if args.operation.startswith("simplex_"):
        from simplex_kernels import make_kernel
    else:
        from projection_kernels import make_kernel

    def make_program(l, c, target):
        return [
            {
                "name": c["kind"],
                "kernel": make_kernel(l, c, target),
                "buffer_map": {n: n for n in buffers(c["kind"])},
                "blocks": c["groups"],
                "threads": 64,
            }
        ]

    run_stage(
        args,
        oracle="geometry",
        buffers=buffers(args.operation),
        make_kernel=None,
        make_program=make_program,
        validate_case=validate_case,
        check_outputs=check_outputs,
        launch=None,
    )


if __name__ == "__main__":
    main()
