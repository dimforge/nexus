"""Counterpart of dynamics/prep_render.rs::gpu_rbd_prep_render (3D)."""

import tilelang.language as T
from kernel_support import f32, quat_mul, quat_rotate, quat_to_mat3


def gpu_rbd_prep_render(l):
    d = l["descriptor"]

    @T.macro
    def one(Poses, Descriptors, Count, Positions, Deformations, Colors, i):
        if i < T.cast(Count[0], "int32"):
            desc = i * l["descriptor_words"]
            body = (
                T.cast(Descriptors[desc + d["pose_index"]], "int32") * l["pose_words"]
            )
            local = desc + d["local_pose"]
            q = T.alloc_local((4,), "float32")
            local_q = T.alloc_local((4,), "float32")
            rotation = T.alloc_local((4,), "float32")
            offset = T.alloc_local((3,), "float32")
            translated = T.alloc_local((3,), "float32")
            matrix = T.alloc_local((3, 3), "float32")
            for j in T.serial(4):
                q[j] = f32(Poses, body + l["pose_rotation"] + j)
                local_q[j] = f32(Descriptors, local + l["pose_rotation"] + j)
                Colors[i * 4 + j] = Descriptors[desc + d["color"] + j]
            for j in T.serial(3):
                offset[j] = f32(Descriptors, local + l["pose_translation"] + j)
            quat_rotate(q, offset, translated)
            quat_mul(q, local_q, rotation)
            quat_to_mat3(rotation, matrix)
            for j in T.serial(3):
                Positions[i * 3 + j] = T.reinterpret(
                    translated[j] + f32(Poses, body + l["pose_translation"] + j),
                    "uint32",
                )
                for k in T.serial(3):
                    Deformations[i * 9 + j * 3 + k] = T.reinterpret(
                        matrix[j, k] * f32(Descriptors, desc + d["scale"] + j), "uint32"
                    )

    return one
