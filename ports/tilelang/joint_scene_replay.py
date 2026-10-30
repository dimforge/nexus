"""Replay continuous awake ordinary-joint scenes against live native stages."""

from joint_scene_validation import BUFFERS, check_outputs, validate_case


def diagnostics(layout, case, before, actual, expected, contract):
    from joint_scene_physics import check

    result = {
        name: check(layout, case, before, result, contract)
        for name, result in (("actual", actual), ("rust", expected))
    }

    result["numerical_policy"] = contract.get("numerical_cases", {}).get(case["name"])
    return result


def main():
    from scene_kernels import make_program
    from stage_replay import parse_args, run_stage

    run_stage(
        parse_args(["joint_scene"], "joint_scene"),
        oracle="joint-scenes",
        buffers=BUFFERS,
        make_kernel=None,
        make_program=make_program,
        validate_case=validate_case,
        check_outputs=check_outputs,
        diagnostics=diagnostics,
        launch=None,
    )


if __name__ == "__main__":
    main()
