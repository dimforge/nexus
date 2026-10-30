"""Live native deferred collision stages, including composite BVH traversal."""

from deferred_validation import (
    BUFFERS,
    OPERATIONS,
    check_outputs,
    diagnostics,
    validate_case,
)


def main():
    from deferred_kernels import BUFFERS as KERNEL_BUFFERS
    from deferred_kernels import make_kernel
    from stage_replay import parse_args, run_stage

    args = parse_args(OPERATIONS, "deferred_manifolds")

    def make_program(l, c, target):
        return [
            {
                "name": c["kind"],
                "kernel": make_kernel(l, c, target, c["kind"]),
                "buffer_map": {n: n for n in KERNEL_BUFFERS},
                "blocks": c["groups"],
                "threads": 64,
            }
        ]

    run_stage(
        args,
        oracle="deferred",
        buffers=BUFFERS,
        make_kernel=None,
        make_program=make_program,
        validate_case=validate_case,
        check_outputs=check_outputs,
        launch=None,
        diagnostics=diagnostics,
    )


if __name__ == "__main__":
    main()
