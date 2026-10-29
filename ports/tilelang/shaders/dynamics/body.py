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
