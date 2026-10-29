"""Counterpart of bounding_volumes/aabb.rs."""

import tilelang.language as T
from kernel_support import quat_rotate, quat_to_mat3


def absolute(value):
    # IEEE abs clears the sign, including negative zero; CPU TileLang cannot
    # lower tirx.fabs, so express the same f32 bit operation on every target.
    return T.reinterpret(
        T.reinterpret(value, "uint32") & T.uint32(0x7FFFFFFF), "float32"
    )


@T.macro
def transform_by(mins, maxs, q, translation, out_mins, out_maxs):
    center = T.alloc_local((3,), "float32")
    half_extents = T.alloc_local((3,), "float32")
    world_center = T.alloc_local((3,), "float32")
    rot = T.alloc_local((3, 3), "float32")
    for d in T.serial(3):
        center[d] = (mins[d] + maxs[d]) * T.float32(0.5)
        half_extents[d] = (maxs[d] - mins[d]) * T.float32(0.5)
    quat_rotate(q, center, world_center)
    quat_to_mat3(q, rot)
    for d in T.serial(3):
        extent = (
            absolute(rot[0, d]) * half_extents[0]
            + absolute(rot[1, d]) * half_extents[1]
        ) + absolute(rot[2, d]) * half_extents[2]
        out_mins[d] = (world_center[d] + translation[d]) - extent
        out_maxs[d] = (world_center[d] + translation[d]) + extent


def intersects(a_min, a_max, b_min, b_max):
    return (
        (a_min[0] <= b_max[0])
        & (a_max[0] >= b_min[0])
        & (a_min[1] <= b_max[1])
        & (a_max[1] >= b_min[1])
        & (a_min[2] <= b_max[2])
        & (a_max[2] >= b_min[2])
    )
