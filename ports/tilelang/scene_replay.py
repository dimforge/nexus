"""Replay connected awake ball/box scenes against the live native Rust stages."""

from scene_validation import BUFFERS, validate_case


def check_outputs(layout, case, before, actual, expected, contract):
    import numpy as np
    from scene_outputs import (
        check_bodies,
        check_collision,
        check_constraints,
        check_graph,
        check_guards,
        check_states,
    )
    from scene_physics import check_physics

    validate_case(layout, case, before)
    for result in (actual, expected):
        if set(result) != set(before) or any(
            result[n].dtype != np.uint32 or result[n].shape != v.shape
            for n, v in before.items()
        ):
            raise ValueError("Invalid returned scene buffers")
    check_guards(layout, case, before, actual, expected)
    errors = check_bodies(layout, case, before, actual, expected, contract)
    errors.update(check_collision(layout, case, before, actual, expected, contract))
    count = check_graph(layout, case, actual, expected)
    errors.update(check_constraints(layout, case, actual, expected, contract, count))
    errors.update(check_states(layout, case, actual, expected, contract, count))
    for result in (actual, expected):
        check_physics(layout, case, before, result, contract)
    return errors


def diagnostics(layout, case, before, actual, expected, contract):
    from scene_physics import check_physics

    return {
        label: check_physics(layout, case, before, result, contract)
        for label, result in (("actual", actual), ("rust", expected))
    }


def main():
    from scene_kernels import make_program
    from stage_replay import parse_args, run_stage

    run_stage(
        parse_args(["analytic_scene"], "analytic_scene"),
        oracle="scenes",
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
