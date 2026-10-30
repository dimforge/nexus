"""Live native support-map and GJK parity, including geometric witnesses."""

from support_validation import BUFFERS, OPERATIONS, check_outputs, validate_case


def main():
    from stage_replay import parse_args, run_stage
    from support_kernels import make_kernel

    def make_program(l, c, target):
        return [
            {
                "name": c["kind"],
                "kernel": make_kernel(l, c, target),
                "buffer_map": {n: n for n in BUFFERS},
                "blocks": c["groups"],
                "threads": 64,
            }
        ]

    run_stage(
        parse_args(OPERATIONS, "gjk_closest"),
        oracle="geometry",
        buffers=BUFFERS,
        make_kernel=None,
        make_program=make_program,
        validate_case=validate_case,
        check_outputs=check_outputs,
        launch=None,
    )


if __name__ == "__main__":
    main()
