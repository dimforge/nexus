# ruff: noqa: SIM117
"""Launch adapters for native joint stages."""

import tilelang.language as T
from shaders.dynamics.joint import motor_params
from shaders.dynamics.joint_constraint import (
    gpu_init_joint_constraints,
    gpu_solve_joint_constraints,
    gpu_update_joint_constraints,
)
from shaders.dynamics.joint_constraint_builder import orthogonalize_constraints

BUFFERS = (
    "Joints",
    "Builders",
    "Constraints",
    "LocalMass",
    "WorldMass",
    "Poses",
    "Velocities",
    "Params",
    "BatchIds",
    "ColorGroups",
    "Color",
    "UseBias",
    "MotorParams",
)


def make_kernel(l, c, target):
    kind = c["kind"]
    sizes = c["buffer_words"]
    groups = c["groups"]
    update = gpu_update_joint_constraints(l)
    init = gpu_init_joint_constraints(l)
    solve = gpu_solve_joint_constraints(l)
    orthogonalize = orthogonalize_constraints(l)
    motor = motor_params(l)

    @T.macro
    def one(
        Joints,
        Builders,
        Constraints,
        LocalMass,
        WorldMass,
        Poses,
        Velocities,
        Params,
        BatchIds,
        ColorGroups,
        Color,
        UseBias,
        MotorParams,
        gid,
    ):
        if kind == "joint_init":
            init(Joints, Builders, Constraints, LocalMass, BatchIds, gid, groups)
        if kind == "joint_update":
            update(
                Builders, Constraints, Poses, WorldMass, Params, BatchIds, gid, groups
            )
        if kind == "joint_solve":
            solve(
                Constraints,
                Velocities,
                ColorGroups,
                Color,
                BatchIds,
                UseBias,
                gid,
                groups,
            )
        if kind == "joint_orthogonalize" and gid < c["count"]:
            for i in T.serial(gid, c["count"], groups * 64):
                orthogonalize(
                    Constraints,
                    i,
                    0,
                    T.cast(
                        Constraints[
                            i * l["joint_constraint_words"]
                            + l["joint_constraint"]["len"]
                        ],
                        "int32",
                    ),
                )
        if kind == "joint_motor_params" and gid < c["count"]:
            for i in T.serial(gid, c["count"], groups * 64):
                for axis in T.serial(6):
                    values = T.alloc_local((6,), "float32")
                    motor(
                        Joints,
                        i * l["impulse_joint_words"]
                        + l["impulse_joint"]["data"]
                        + l["generic_joint"]["motors"]
                        + axis * l["motor_words"],
                        Params,
                        values,
                    )
                    for d in T.serial(6):
                        MotorParams[(i * 6 + axis) * 6 + d] = T.reinterpret(
                            values[d], "uint32"
                        )

    if target == "c":

        @T.prim_func
        def kernel(
            Joints: T.Tensor((sizes["Joints"],), "uint32"),
            Builders: T.Tensor((sizes["Builders"],), "uint32"),
            Constraints: T.Tensor((sizes["Constraints"],), "uint32"),
            LocalMass: T.Tensor((sizes["LocalMass"],), "uint32"),
            WorldMass: T.Tensor((sizes["WorldMass"],), "uint32"),
            Poses: T.Tensor((sizes["Poses"],), "uint32"),
            Velocities: T.Tensor((sizes["Velocities"],), "uint32"),
            Params: T.Tensor((sizes["Params"],), "uint32"),
            BatchIds: T.Tensor((sizes["BatchIds"],), "uint32"),
            ColorGroups: T.Tensor((sizes["ColorGroups"],), "uint32"),
            Color: T.Tensor((sizes["Color"],), "uint32"),
            UseBias: T.Tensor((sizes["UseBias"],), "uint32"),
            MotorParams: T.Tensor((sizes["MotorParams"],), "uint32"),
        ):
            with T.Kernel(1):
                for gid in T.serial(groups * 64):
                    one(
                        Joints,
                        Builders,
                        Constraints,
                        LocalMass,
                        WorldMass,
                        Poses,
                        Velocities,
                        Params,
                        BatchIds,
                        ColorGroups,
                        Color,
                        UseBias,
                        MotorParams,
                        gid,
                    )
    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            Joints: A.Tensor((sizes["Joints"],), "uint32"),
            Builders: A.Tensor((sizes["Builders"],), "uint32"),
            Constraints: A.Tensor((sizes["Constraints"],), "uint32"),
            LocalMass: A.Tensor((sizes["LocalMass"],), "uint32"),
            WorldMass: A.Tensor((sizes["WorldMass"],), "uint32"),
            Poses: A.Tensor((sizes["Poses"],), "uint32"),
            Velocities: A.Tensor((sizes["Velocities"],), "uint32"),
            Params: A.Tensor((sizes["Params"],), "uint32"),
            BatchIds: A.Tensor((sizes["BatchIds"],), "uint32"),
            ColorGroups: A.Tensor((sizes["ColorGroups"],), "uint32"),
            Color: A.Tensor((sizes["Color"],), "uint32"),
            UseBias: A.Tensor((sizes["UseBias"],), "uint32"),
            MotorParams: A.Tensor((sizes["MotorParams"],), "uint32"),
        ):
            with A.Kernel(groups) as group:
                with A.SimtVF(threads=64):
                    one(
                        Joints,
                        Builders,
                        Constraints,
                        LocalMass,
                        WorldMass,
                        Poses,
                        Velocities,
                        Params,
                        BatchIds,
                        ColorGroups,
                        Color,
                        UseBias,
                        MotorParams,
                        group * 64 + A.get_thread_binding(),
                    )
    else:

        @T.prim_func
        def kernel(
            Joints: T.Tensor((sizes["Joints"],), "uint32"),
            Builders: T.Tensor((sizes["Builders"],), "uint32"),
            Constraints: T.Tensor((sizes["Constraints"],), "uint32"),
            LocalMass: T.Tensor((sizes["LocalMass"],), "uint32"),
            WorldMass: T.Tensor((sizes["WorldMass"],), "uint32"),
            Poses: T.Tensor((sizes["Poses"],), "uint32"),
            Velocities: T.Tensor((sizes["Velocities"],), "uint32"),
            Params: T.Tensor((sizes["Params"],), "uint32"),
            BatchIds: T.Tensor((sizes["BatchIds"],), "uint32"),
            ColorGroups: T.Tensor((sizes["ColorGroups"],), "uint32"),
            Color: T.Tensor((sizes["Color"],), "uint32"),
            UseBias: T.Tensor((sizes["UseBias"],), "uint32"),
            MotorParams: T.Tensor((sizes["MotorParams"],), "uint32"),
        ):
            with T.Kernel(groups, threads=64) as group:
                one(
                    Joints,
                    Builders,
                    Constraints,
                    LocalMass,
                    WorldMass,
                    Poses,
                    Velocities,
                    Params,
                    BatchIds,
                    ColorGroups,
                    Color,
                    UseBias,
                    MotorParams,
                    group * 64 + T.get_thread_binding(),
                )

    return kernel


def make_program(layout, case, target):
    kinds = (
        ("joint_init", "joint_update", "joint_solve")
        if case["kind"] == "joint_program"
        else (case["kind"],)
    )
    return [
        {
            "name": kind,
            "kernel": make_kernel(layout, dict(case, kind=kind), target),
            "blocks": case["groups"],
            "threads": 64,
            "buffer_map": {name: name for name in BUFFERS},
        }
        for kind in kinds
    ]
