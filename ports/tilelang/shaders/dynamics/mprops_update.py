"""Counterpart of dynamics/mprops_update.rs; world mass-property update."""

import tilelang.language as T
from .body import to_world


def gpu_update_mprops(layout, case):
    update = to_world(layout)

    @T.macro
    def one(WorldMass, LocalMass, Poses, i):
        # The launch covers the full count, specializing the Rust grid-stride loop.
        if i < case["count"]:
            update(LocalMass, Poses, WorldMass, i)

    return one


def gpu_sync_collider_poses(layout, case):
    from kernel_support import f32, quat_mul, quat_rotate

    @T.macro
    def one(BodyPoses, LocalPoses, WorldPoses, Parents, i):
        if i < case["count"]:
            bp = Parents[i] * layout["pose_words"]
            cp = i * layout["pose_words"]
            body_q = T.alloc_local((4,), "float32")
            local_q = T.alloc_local((4,), "float32")
            world_q = T.alloc_local((4,), "float32")
            translation = T.alloc_local((3,), "float32")
            rotated = T.alloc_local((3,), "float32")
            for d in T.serial(4):
                body_q[d] = f32(BodyPoses, bp + layout["pose_rotation"] + d)
                local_q[d] = f32(LocalPoses, cp + layout["pose_rotation"] + d)
            for d in T.serial(3):
                translation[d] = f32(LocalPoses, cp + layout["pose_translation"] + d)
            quat_mul(body_q, local_q, world_q)
            quat_rotate(body_q, translation, rotated)
            for d in T.serial(4):
                WorldPoses[cp + layout["pose_rotation"] + d] = T.reinterpret(
                    world_q[d], "uint32"
                )
            for d in T.serial(3):
                WorldPoses[cp + layout["pose_translation"] + d] = T.reinterpret(
                    f32(BodyPoses, bp + layout["pose_translation"] + d) + rotated[d],
                    "uint32",
                )
            WorldPoses[cp + layout["pose_padding"]] = T.uint32(0)

    return one
