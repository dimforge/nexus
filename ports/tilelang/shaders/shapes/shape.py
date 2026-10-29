"""Counterpart of shapes/shape.rs; all native 3D compute_aabb branches."""

import tilelang.language as T
from kernel_support import f32, quat_rotate
from ..bounding_volumes.aabb import transform_by
from .triangle import aabb as triangle_aabb
from .convex_polyhedron import aabb as convex_aabb


def compute_aabb(layout):
    sf = layout["shape"]
    types = layout["shape_types"]
    convex = convex_aabb(layout)

    @T.macro
    def one(Shapes, Poses, Vertices, index, mins, maxs):
        base = index * layout["shape_words"]
        pose = index * layout["pose_words"]
        ty = Shapes[base + sf["a"] + 3]
        a = T.alloc_local((3,), "float32")
        b = T.alloc_local((3,), "float32")
        c = T.alloc_local((3,), "float32")
        q = T.alloc_local((4,), "float32")
        translation = T.alloc_local((3,), "float32")
        local_min = T.alloc_local((3,), "float32")
        local_max = T.alloc_local((3,), "float32")
        for d in T.serial(3):
            a[d] = f32(Shapes, base + sf["a"] + d)
            b[d] = f32(Shapes, base + sf["b"] + d)
            c[d] = f32(Shapes, base + sf["c"] + d)
            translation[d] = f32(Poses, pose + layout["pose_translation"] + d)
            mins[d] = T.float32(0)
            maxs[d] = T.float32(0)
            local_min[d] = T.float32(0)
            local_max[d] = T.float32(0)
        for d in T.serial(4):
            q[d] = f32(Poses, pose + layout["pose_rotation"] + d)
        if ty == types["ball"]:
            for d in T.serial(3):
                mins[d] = translation[d] - a[0]
                maxs[d] = translation[d] + a[0]
        elif ty == types["capsule"]:
            aa = T.alloc_local((3,), "float32")
            bb = T.alloc_local((3,), "float32")
            quat_rotate(q, a, aa)
            quat_rotate(q, b, bb)
            radius = f32(Shapes, base + sf["b"] + 3)
            for d in T.serial(3):
                va = aa[d] + translation[d]
                vb = bb[d] + translation[d]
                mins[d] = T.min(va, vb) - radius
                maxs[d] = T.max(va, vb) + radius
        else:
            valid = T.alloc_local((1,), "bool")
            valid[0] = True
            if ty == types["cuboid"]:
                for d in T.serial(3):
                    local_min[d] = -a[d]
                    local_max[d] = a[d]
            elif ty == types["cone"] or ty == types["cylinder"]:
                for d in T.serial(3):
                    extent = T.if_then_else(d == 1, a[0], a[1])
                    local_min[d] = -extent
                    local_max[d] = extent
            elif ty == types["triangle"]:
                triangle_aabb(a, b, c, local_min, local_max)
            elif ty == types["convex_poly"]:
                first = T.cast(Shapes[base + sf["a"]], "int32")
                end = T.cast(Shapes[base + sf["a"] + 1], "int32")
                convex(Vertices, first, end, local_min, local_max)
            elif ty == types["trimesh"] or ty == types["polyline"]:
                thickness = T.if_then_else(
                    ty == types["polyline"],
                    T.reinterpret(
                        T.uint32(layout["polyline_thickness_bits"]), "float32"
                    ),
                    T.float32(0),
                )
                for d in T.serial(3):
                    local_min[d] = b[d] - thickness
                    local_max[d] = c[d] + thickness
            else:
                valid[0] = False
            if valid[0]:
                transform_by(local_min, local_max, q, translation, mins, maxs)

    return one
