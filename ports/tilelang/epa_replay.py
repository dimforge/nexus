"""Live native EPA and GJK/EPA parity, including geometric witnesses."""

from epa_validation import BUFFERS, OPERATIONS, check_outputs, validate_case


def main():
    from epa_kernels import make_kernel
    from stage_replay import parse_args, run_stage

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
        parse_args(OPERATIONS, "gjk_epa"),
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
