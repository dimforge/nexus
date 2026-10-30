"""Live Rust PFM subshape conversion and convex manifold replay."""

from pfm_validation import (
    BUFFERS,
    OPERATIONS,
    check_outputs,
    diagnostics,
    validate_case,
)


def main():
    from pfm_kernels import make_kernel
    from stage_replay import parse_args, run_stage

    args = parse_args(OPERATIONS, "pfm_manifold")

    def make_program(l, c, target):
        stages = (
            ["subshape"] if c["kind"] == "pfm_subshape" else ["subshape", "manifold"]
        )
        return [
            {
                "name": stage,
                "kernel": make_kernel(l, c, target, stage),
                "buffer_map": {n: n for n in BUFFERS},
                "blocks": c["groups"],
                "threads": 64,
            }
            for stage in stages
        ]

    run_stage(
        args,
        oracle="geometry",
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
