"""Live native support faces and generic polygonal clipping replay."""

from feature_validation import (
    CLIP_BUFFERS,
    FACE_BUFFERS,
    OPERATIONS,
    check_outputs,
    validate_case,
)


def main():
    from feature_kernels import make_kernel as face_kernel
    from polygonal_kernels import make_kernel as clip_kernel
    from stage_replay import parse_args, run_stage

    args = parse_args(OPERATIONS, "polygonal_contacts")
    buffers = FACE_BUFFERS if args.operation == "support_face" else CLIP_BUFFERS
    kernel = face_kernel if args.operation == "support_face" else clip_kernel

    def make_program(l, c, target):
        return [
            {
                "name": c["kind"],
                "kernel": kernel(l, c, target),
                "buffer_map": {n: n for n in buffers},
                "blocks": c["groups"],
                "threads": 64,
            }
        ]

    run_stage(
        args,
        oracle="geometry",
        buffers=buffers,
        make_kernel=None,
        make_program=make_program,
        validate_case=validate_case,
        check_outputs=check_outputs,
        launch=None,
    )


if __name__ == "__main__":
    main()
