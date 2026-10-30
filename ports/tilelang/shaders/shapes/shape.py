"""Counterpart of shapes/shape.rs; bounds and analytic primitive operations."""

import tilelang.language as T
from kernel_support import dot, f32, quat_rotate

from ..bounding_volumes.aabb import absolute, transform_by
from . import capsule, cone, cylinder
from .convex_polyhedron import aabb as convex_aabb
from .triangle import aabb as triangle_aabb


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


@T.macro
def project_cuboid_boundary(extents, pt, out, inside):
    # Parry 0.31.1 Cuboid -> Aabb point query, reexported by shapes/mod.rs.
    # Parry's copy_sign_to uses copysign, preserving signed zero.
    shift = T.alloc_local((3,), "float32")
    for d in T.serial(3):
        shift[d] = T.max(-extents[d] - pt[d], T.float32(0)) - T.max(
            pt[d] - extents[d], T.float32(0)
        )
        out[d] = pt[d] + shift[d]
    inside[0] = (
        shift[0] == T.float32(0)
        and shift[1] == T.float32(0)
        and shift[2] == T.float32(0)
    )
    if inside[0]:
        sign = T.alloc_local((3,), "float32")
        diff = T.alloc_local((3,), "float32")
        for d in T.serial(3):
            sign[d] = T.reinterpret(
                (T.reinterpret(pt[d], "uint32") & T.uint32(0x80000000))
                | T.uint32(0x3F800000),
                "float32",
            )
            diff[d] = extents[d] - sign[d] * pt[d]
            out[d] = pt[d]
        if diff[0] <= diff[1] and diff[0] <= diff[2]:
            out[0] = pt[0] + diff[0] * sign[0]
        elif diff[1] <= diff[0] and diff[1] <= diff[2]:
            out[1] = pt[1] + diff[1] * sign[1]
        else:
            out[2] = pt[2] + diff[2] * sign[2]


def project_local_point_on_boundary(layout):
    sf, types = layout["shape"], layout["shape_types"]

    @T.macro
    def one(Shapes, index, pt, out, inside):
        base = index * layout["shape_words"]
        ty = T.cast(Shapes[base + sf["a"] + 3], "int32")
        a = T.alloc_local((3,), "float32")
        b = T.alloc_local((3,), "float32")
        for d in T.serial(3):
            a[d] = f32(Shapes, base + sf["a"] + d)
            b[d] = f32(Shapes, base + sf["b"] + d)
        if ty == types["cuboid"]:
            project_cuboid_boundary(a, pt, out, inside)
        elif ty == types["capsule"]:
            capsule.project_local_point_on_boundary(
                a, b, f32(Shapes, base + sf["b"] + 3), pt, out, inside
            )
        elif ty == types["cone"]:
            cone.project_local_point_on_boundary(a[0], a[1], pt, out, inside)
        elif ty == types["cylinder"]:
            cylinder.project_local_point_on_boundary(a[0], a[1], pt, out, inside)

    return one


def primitive_local_extent(layout):
    sf, types = layout["shape"], layout["shape_types"]

    @T.macro
    def one(Shapes, index, out):
        base = index * layout["shape_words"]
        ty = T.cast(Shapes[base + sf["a"] + 3], "int32")
        a = T.alloc_local((3,), "float32")
        b = T.alloc_local((3,), "float32")
        for d in T.serial(3):
            a[d] = f32(Shapes, base + sf["a"] + d)
            b[d] = f32(Shapes, base + sf["b"] + d)
        out[0] = T.float32(0)
        if ty == types["ball"]:
            out[0] = a[0] * T.float32(1.7320508)
        elif ty == types["cuboid"]:
            out[0] = T.sqrt(dot(a, a))
        elif ty == types["capsule"]:
            radius = f32(Shapes, base + sf["b"] + 3)
            lo = T.alloc_local((3,), "float32")
            hi = T.alloc_local((3,), "float32")
            for d in T.serial(3):
                lo[d] = T.min(a[d], b[d]) - radius
                hi[d] = T.max(a[d], b[d]) + radius
            out[0] = T.max(T.sqrt(dot(lo, lo)), T.sqrt(dot(hi, hi)))
        elif ty == types["cone"] or ty == types["cylinder"]:
            out[0] = T.sqrt((a[1] * a[1] + a[0] * a[0]) + a[1] * a[1])

    return one


@T.macro
def cuboid_support_point(half_extents, direction, point):
    # Parry's reexported Cuboid support map uses copysign, including signed zero.
    for d in T.serial(3):
        point[d] = T.reinterpret(
            (T.reinterpret(direction[d], "uint32") & T.uint32(0x80000000))
            | (T.reinterpret(half_extents[d], "uint32") & T.uint32(0x7FFFFFFF)),
            "float32",
        )


@T.macro
def cuboid_support_face(he, direction, vertices):
    # Parry 0.31.1's Cuboid::support_face (reexported through shapes/mod.rs).
    axis = T.alloc_local((1,), "int32")
    axis[0] = 0
    maximum = T.alloc_local((1,), "float32")
    maximum[0] = absolute(direction[0])
    if absolute(direction[1]) > maximum[0]:
        axis[0] = 1
        maximum[0] = absolute(direction[1])
    if absolute(direction[2]) > maximum[0]:
        axis[0] = 2
    sign = T.reinterpret(
        (T.reinterpret(direction[axis[0]], "uint32") & T.uint32(0x80000000))
        | T.uint32(0x3F800000),
        "float32",
    )
    if axis[0] == 0:
        for i in T.serial(4):
            vertices[i, 0] = he[0] * sign
            vertices[i, 1] = T.if_then_else(i == 0 or i == 3, he[1], -he[1])
            vertices[i, 2] = T.if_then_else(i < 2, he[2], -he[2])
    elif axis[0] == 1:
        for i in T.serial(4):
            vertices[i, 0] = T.if_then_else(i == 0 or i == 3, he[0], -he[0])
            vertices[i, 1] = he[1] * sign
            vertices[i, 2] = T.if_then_else(i < 2, he[2], -he[2])
    else:
        for i in T.serial(4):
            vertices[i, 0] = T.if_then_else(i < 2, he[0], -he[0])
            vertices[i, 1] = T.if_then_else(i == 0 or i == 3, he[1], -he[1])
            vertices[i, 2] = he[2] * sign


def local_support_point(l):
    from . import ball_local_support_point, cuboid_local_support_point
    from .convex_polyhedron import local_support_point as convex_support
    from .triangle import local_support_point as triangle_support

    convex = convex_support(l)
    sf = l["shape"]
    types = l["shape_types"]

    @T.macro
    def one(Shapes, index, Vertices, direction, Output):
        base = index * l["shape_words"]
        ty = T.cast(Shapes[base + sf["a"] + 3], "int32")
        a = T.alloc_local((3,), "float32")
        b = T.alloc_local((3,), "float32")
        c = T.alloc_local((3,), "float32")
        for d in T.serial(3):
            a[d] = f32(Shapes, base + sf["a"] + d)
            b[d] = f32(Shapes, base + sf["b"] + d)
            c[d] = f32(Shapes, base + sf["c"] + d)
            Output[d] = 0.0
        if ty == types["ball"]:
            ball_local_support_point(a[0], direction, Output)
        elif ty == types["cuboid"]:
            cuboid_local_support_point(a, direction, Output)
        elif ty == types["triangle"]:
            triangle_support(a, b, c, direction, Output)
        elif ty == types["capsule"]:
            capsule.local_support_point(
                a, b, f32(Shapes, base + sf["b"] + 3), direction, Output
            )
        elif ty == types["cone"]:
            cone.local_support_point(a[0], a[1], direction, Output)
        elif ty == types["cylinder"]:
            cylinder.local_support_point(a[0], a[1], direction, Output)
        elif ty == types["convex_poly"]:
            convex(
                Vertices,
                T.cast(Shapes[base + sf["a"]], "int32"),
                T.cast(Shapes[base + sf["a"] + 1], "int32"),
                direction,
                Output,
            )

    return one


def support_point(l):
    support = local_support_point(l)

    @T.macro
    def one(Shapes, index, Vertices, q, translation, direction, Output):
        inverse = T.alloc_local((4,), "float32")
        axis = T.alloc_local((3,), "float32")
        point = T.alloc_local((3,), "float32")
        for d in T.serial(3):
            inverse[d] = -q[d]
        inverse[3] = q[3]
        quat_rotate(inverse, direction, axis)
        support(Shapes, index, Vertices, axis, point)
        quat_rotate(q, point, Output)
        for d in T.serial(3):
            Output[d] = Output[d] + translation[d]

    return one


def support_face(l):
    from .capsule import support_face as capsule_face
    from .cone import support_face as cone_face
    from .convex_polyhedron import support_face as poly_face
    from .cylinder import support_face as cylinder_face
    from .triangle import support_face as triangle_face

    poly = poly_face(l)
    sf = l["shape"]
    types = l["shape_types"]

    @T.macro
    def one(Shapes, index, Vertices, Indices, direction, Output, Count):
        base = index * l["shape_words"]
        ty = T.cast(Shapes[base + sf["a"] + 3], "int32")
        a = T.alloc_local((3,), "float32")
        b = T.alloc_local((3,), "float32")
        c = T.alloc_local((3,), "float32")
        for d in T.serial(3):
            a[d] = f32(Shapes, base + sf["a"] + d)
            b[d] = f32(Shapes, base + sf["b"] + d)
            c[d] = f32(Shapes, base + sf["c"] + d)
        Count[0] = 0
        for v in T.serial(4):
            for d in T.serial(3):
                Output[v, d] = 0.0
        if ty == types["cuboid"]:
            cuboid_support_face(a, direction, Output)
            Count[0] = 4
        elif ty == types["capsule"]:
            capsule_face(
                a, b, f32(Shapes, base + sf["b"] + 3), direction, Output, Count
            )
        elif ty == types["cone"]:
            cone_face(a[0], a[1], direction, Output, Count)
        elif ty == types["cylinder"]:
            cylinder_face(a[0], a[1], direction, Output, Count)
        elif ty == types["triangle"]:
            triangle_face(a, b, c, Output, Count)
        elif ty == types["convex_poly"]:
            poly(
                Vertices,
                Indices,
                T.cast(Shapes[base + sf["a"]], "int32"),
                T.cast(Shapes[base + sf["b"]], "int32"),
                T.cast(Shapes[base + sf["b"] + 1], "int32"),
                direction,
                Output,
                Count,
            )

    return one


def pfm_subshape(l):
    sf = l["shape"]
    types = l["shape_types"]

    @T.macro
    def one(Shapes, index, Output, Thickness, Valid):
        base = index * l["shape_words"]
        ty = T.cast(Shapes[base + sf["a"] + 3], "int32")
        for word in T.serial(l["shape_words"]):
            Output[word] = Shapes[base + word]
        Thickness[0] = 0.0
        Valid[0] = (
            (ty == types["cuboid"])
            | (ty == types["cone"])
            | (ty == types["cylinder"])
            | (ty == types["convex_poly"])
            | (ty == types["triangle"])
        )
        if (ty == types["ball"]) | (ty == types["capsule"]):
            for word in T.serial(l["shape_words"]):
                Output[word] = T.uint32(0)
            Output[sf["a"] + 3] = T.uint32(types["capsule"])
            Valid[0] = True
            if ty == types["ball"]:
                Thickness[0] = f32(Shapes, base + sf["a"])
            else:
                for d in T.serial(3):
                    Output[sf["a"] + d] = Shapes[base + sf["a"] + d]
                    Output[sf["b"] + d] = Shapes[base + sf["b"] + d]
                Thickness[0] = f32(Shapes, base + sf["b"] + 3)

    return one
