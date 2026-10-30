"""Replay one frame from freshly exported native triangle/chain checkpoints."""

from scene_restart_validation import check_outputs, validate_case


def diagnostics(layout, case, before, actual, expected, contract):
    return {
        "start_frame": case["start_frame"],
        "numerical_policy": contract.get("numerical_cases", {}).get(case["name"], {}),
    }


def main():
    from general_scene_validation import BUFFERS as GENERAL
    from joint_scene_validation import BUFFERS as JOINT
    from scene_kernels import make_program
    from stage_replay import parse_args, run_stage

    args = parse_args(
        [
            "general_restart",
            "joint_restart",
            "general_lbvh_restart",
            "joint_lbvh_restart",
        ],
        "general_restart",
    )
    from scene_broadphase import EXTRA

    buffers = JOINT if args.operation.startswith("joint_") else GENERAL
    if "_lbvh_" in args.operation:
        buffers += EXTRA
    run_stage(
        args,
        oracle="scene-restarts",
        buffers=buffers,
        oracle_args=(args.operation,),
        make_kernel=None,
        make_program=make_program,
        validate_case=validate_case,
        check_outputs=check_outputs,
        diagnostics=diagnostics,
        launch=None,
    )


if __name__ == "__main__":
    main()
