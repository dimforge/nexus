"""Counterpart of dynamics/joint_constraint.rs entry points."""

import tilelang.language as T
from kernel_support import f32


def gpu_init_joint_constraints(l):
    j, g, b, c = (
        l["impulse_joint"],
        l["generic_joint"],
        l["builder"],
        l["joint_constraint"],
    )

    @T.macro
    def one(Joints, Builders, Constraints, LocalMass, BatchIds, gid, groups):
        nb = T.cast(BatchIds[l["batch"]["num_batches"]], "int32")
        total = T.cast(BatchIds[l["batch"]["impulse_joints_len"]], "int32") * nb
        if gid < total:
            for i in T.serial(gid, total, groups * 64):
                src = i * l["impulse_joint_words"]
                dst = i * l["builder_words"]
                con = i * l["joint_constraint_words"]
                ba = T.cast(Joints[src + j["body_a"]], "int32") * nb + i % nb
                bb = T.cast(Joints[src + j["body_b"]], "int32") * nb + i % nb
                Builders[dst + b["body1"]] = T.cast(ba, "uint32")
                Builders[dst + b["body2"]] = T.cast(bb, "uint32")
                Builders[dst + b["joint_id"]] = T.cast(i, "uint32")
                Builders[dst + b["constraint_id"]] = T.cast(i, "uint32")
                for d in T.serial(l["generic_joint_words"]):
                    Builders[dst + b["joint"] + d] = Joints[src + j["data"] + d]
                for d in T.serial(3):
                    fa = b["joint"] + g["local_frame_a"] + l["pose_translation"] + d
                    fb = b["joint"] + g["local_frame_b"] + l["pose_translation"] + d
                    Builders[dst + fa] = T.reinterpret(
                        f32(Builders, dst + fa)
                        - f32(LocalMass, ba * l["local_words"] + l["local_com"] + d),
                        "uint32",
                    )
                    Builders[dst + fb] = T.reinterpret(
                        f32(Builders, dst + fb)
                        - f32(LocalMass, bb * l["local_words"] + l["local_com"] + d),
                        "uint32",
                    )
                    Constraints[con + c["im_a"] + d] = LocalMass[
                        ba * l["local_words"] + l["local_inv_mass"] + d
                    ]
                    Constraints[con + c["im_b"] + d] = LocalMass[
                        bb * l["local_words"] + l["local_inv_mass"] + d
                    ]
                Constraints[con + c["solver_vel_a"]] = T.cast(ba, "uint32")
                Constraints[con + c["solver_vel_b"]] = T.cast(bb, "uint32")
                Constraints[con + c["len"]] = T.uint32(0)

    return one


def gpu_solve_joint_constraints(l):
    from .joint_constraint_builder import solve_joint_constraint

    solve = solve_joint_constraint(l)

    @T.macro
    def one(
        Constraints, Velocities, ColorGroups, Color, BatchIds, UseBias, gid, groups
    ):
        nb = T.cast(BatchIds[l["batch"]["num_batches"]], "int32")
        color = T.cast(Color[0], "int32")
        start = T.alloc_local((1,), "int32")
        start[0] = 0
        if color > 0:
            start[0] = T.cast(ColorGroups[color - 1], "int32")
        end = T.cast(ColorGroups[color], "int32")
        if start[0] * nb + gid < end * nb:
            for i in T.serial(start[0] * nb + gid, end * nb, groups * 64):
                solve(Constraints, Velocities, i, UseBias[0] != 0)

    return one


def gpu_update_joint_constraints(l):
    from .joint_constraint_builder import update_constraint

    update = update_constraint(l)

    @T.macro
    def one(Builders, Constraints, Poses, WorldMass, Params, BatchIds, gid, groups):
        total = T.cast(
            BatchIds[l["batch"]["impulse_joints_len"]]
            * BatchIds[l["batch"]["num_batches"]],
            "int32",
        )
        if gid < total:
            for i in T.serial(gid, total, groups * 64):
                update(Builders, Constraints, Poses, WorldMass, Params, i)

    return one


def ii_mul(l):
    """JointSolverBody::ii_mul, including the fourth zero component of Mat4."""

    @T.macro
    def one(WorldMass, body, Vector, Result):
        for d in T.serial(3):
            base = body * l["world_words"] + l["world_inertia"] + d
            Result[d] = (
                (
                    f32(WorldMass, base) * Vector[0]
                    + f32(WorldMass, base + 4) * Vector[1]
                )
                + f32(WorldMass, base + 8) * Vector[2]
            ) + f32(WorldMass, base + 12) * 0.0

    return one
