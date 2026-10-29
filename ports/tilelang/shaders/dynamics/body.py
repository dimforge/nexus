"""Counterpart of dynamics/body.rs::Velocity::integrate_linearized (3D)."""

import tilelang.language as T
from kernel_support import f32


def integrate_linearized(layout):
    @T.macro
    def integrate(Poses, i, v, w, dt):
        pp = i * layout["pose_words"]
        q = T.alloc_local((4,), "float32")
        r = T.alloc_local((4,), "float32")
        half_angular = T.alloc_local((3,), "float32")
        for d in T.serial(3):
            Poses[pp + layout["pose_translation"] + d] = T.reinterpret(
                f32(Poses, pp + layout["pose_translation"] + d) + v[d] * dt,
                "uint32",
            )
            half_angular[d] = w[d] * (dt * 0.5)
        for d in T.serial(4):
            q[d] = f32(Poses, pp + layout["pose_rotation"] + d)
        r[0] = (
            q[0]
            + half_angular[0] * q[3]
            + half_angular[1] * q[2]
            - half_angular[2] * q[1]
        )
        r[1] = (
            q[1]
            - half_angular[0] * q[2]
            + half_angular[1] * q[3]
            + half_angular[2] * q[0]
        )
        r[2] = (
            q[2]
            + half_angular[0] * q[1]
            - half_angular[1] * q[0]
            + half_angular[2] * q[3]
        )
        r[3] = (
            q[3]
            - half_angular[0] * q[0]
            - half_angular[1] * q[1]
            - half_angular[2] * q[2]
        )
        inv_len = 1.0 / T.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3])
        for d in T.serial(4):
            Poses[pp + layout["pose_rotation"] + d] = T.reinterpret(
                r[d] * inv_len, "uint32"
            )

    return integrate


def to_world(layout):
    """LocalMassProperties::to_world, including full Mat4 and explicit padding."""
    from kernel_support import quat_mul, quat_rotate, quat_to_mat3

    @T.macro
    def one(LocalMass, Poses, WorldMass, i):
        lp = i * layout["local_words"]
        pp = i * layout["pose_words"]
        wp = i * layout["world_words"]
        q = T.alloc_local((4,), "float32")
        frame = T.alloc_local((4,), "float32")
        world_frame = T.alloc_local((4,), "float32")
        com = T.alloc_local((3,), "float32")
        rotated_com = T.alloc_local((3,), "float32")
        inertia = T.alloc_local((3,), "float32")
        rotation = T.alloc_local((3, 3), "float32")
        scaled = T.alloc_local((3, 3), "float32")
        for d in T.serial(4):
            q[d] = f32(Poses, pp + layout["pose_rotation"] + d)
            frame[d] = f32(LocalMass, lp + layout["local_inertia_frame"] + d)
        for d in T.serial(3):
            com[d] = f32(LocalMass, lp + layout["local_com"] + d)
            inertia[d] = f32(LocalMass, lp + layout["local_inv_inertia"] + d)
        quat_rotate(q, com, rotated_com)
        quat_mul(q, frame, world_frame)
        quat_to_mat3(world_frame, rotation)
        for c in T.serial(3):
            for r in T.serial(3):
                scaled[c, r] = rotation[c, r] * inertia[c]
        for c in T.serial(4):
            for r in T.serial(4):
                if c < 3 and r < 3:
                    value = (
                        scaled[0, r] * rotation[0, c] + scaled[1, r] * rotation[1, c]
                    ) + scaled[2, r] * rotation[2, c]
                    WorldMass[wp + layout["world_inertia"] + c * 4 + r] = T.reinterpret(
                        value, "uint32"
                    )
                else:
                    WorldMass[wp + layout["world_inertia"] + c * 4 + r] = T.reinterpret(
                        T.if_then_else(c == 3 and r == 3, T.float32(1), T.float32(0)),
                        "uint32",
                    )
        for d in T.serial(3):
            WorldMass[wp + layout["world_inv_mass"] + d] = LocalMass[
                lp + layout["local_inv_mass"] + d
            ]
            WorldMass[wp + layout["world_com"] + d] = T.reinterpret(
                f32(Poses, pp + layout["pose_translation"] + d) + rotated_com[d],
                "uint32",
            )
        WorldMass[wp + layout["world_padding0"]] = T.uint32(0)
        WorldMass[wp + layout["world_padding1"]] = T.uint32(0)

    return one
