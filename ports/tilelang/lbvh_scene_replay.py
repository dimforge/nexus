"""Execute complete scene frames using the resident native LBVH broad phase."""

from lbvh_scene_validation import check_outputs, validate_case
from scene_broadphase import EXTRA


def main():
    import general_scene_validation as general
    import joint_scene_validation as joint
    import scene_validation as analytic
    from scene_kernels import make_program
    from stage_replay import parse_args, run_stage

    operations = {
        name + "_lbvh_scene": module
        for name, module in (
            ("analytic", analytic),
            ("general", general),
            ("joint", joint),
        )
    }
    args = parse_args(operations, "general_lbvh_scene")
    run_stage(
        args,
        oracle="lbvh-scenes",
        oracle_args=(args.operation,),
        buffers=operations[args.operation].BUFFERS + EXTRA,
        make_kernel=None,
        make_program=make_program,
        validate_case=validate_case,
        check_outputs=check_outputs,
        launch=None,
    )


if __name__ == "__main__":
    main()
