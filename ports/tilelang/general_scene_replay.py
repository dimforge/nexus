"""Replay continuous awake general-collision scenes against live native stages."""

from general_scene_validation import BUFFERS, check_outputs, validate_case


def diagnostics(layout, case, before, actual, expected, contract):
    from general_scene_outputs import witness_diagnostics
    from general_scene_physics import check

    result = {
        name: dict(
            check(layout, case, before, result, contract),
            secondary_witness_misses=witness_diagnostics(layout, case, before, result),
        )
        for name, result in (("actual", actual), ("rust", expected))
    }

    result["numerical_policy"] = contract.get("numerical_cases", {}).get(case["name"])
    return result


def main():
    from scene_kernels import make_program
    from stage_replay import parse_args, run_stage

    run_stage(
        parse_args(["general_scene"], "general_scene"),
        oracle="general-scenes",
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
