"""Counterpart of src_rbd_shaders/dynamics/solver_utils.rs.

solve_points preserves normal-row order followed by friction-row order. The
factory captures Rust-derived layouts; its macro operates on native typed tiles.
"""

import tilelang.language as T
from kernel_support import dot, f32, relative_velocity


def solve_points(layout, case):
    n = case["count"]
    tile_words = layout["tile_words"]
    h, p = layout["header"], layout["point"]
    use_bias, solve_friction = case["use_bias"], case["solve_friction"]

    @T.macro
    def cross(a, b, out):
        out[0] = a[1] * b[2] - a[2] * b[1]
        out[1] = a[2] * b[0] - a[0] * b[2]
        out[2] = a[0] * b[1] - a[1] * b[0]

    @T.macro
    def inertia(diag, off, v, out):
        out[0] = diag[0] * v[0] + off[0] * v[1] + off[1] * v[2]
        out[1] = off[0] * v[0] + diag[1] * v[1] + off[2] * v[2]
        out[2] = off[1] * v[0] + off[2] * v[1] + diag[2] * v[2]

    @T.macro
    def transform_point(Poses, base, local, out):
        q = T.alloc_local((3,), "float32")
        crossed = T.alloc_local((3,), "float32")
        for d in T.serial(3):
            q[d] = f32(Poses, base + layout["pose_rotation"] + d)
        w = f32(Poses, base + layout["pose_rotation"] + 3)
        cross(q, local, crossed)
        scale = w * w - dot(q, q)
        along = dot(local, q) * 2.0
        for d in T.serial(3):
            out[d] = (
                local[d] * scale
                + q[d] * along
                + crossed[d] * (w * 2.0)
                + f32(Poses, base + layout["pose_translation"] + d)
            )

    @T.macro
    def solve_one(Tiles, Velocities, Poses, Params, index):
        if index < n:
            base = (index // 32) * tile_words
            lane = index % 32
            lane4 = lane * layout["vec_scalar_words"]
            count = Tiles[base + h["normal"] + lane4 + 3]
            body_a = T.cast(Tiles[base + h["body_a"] + lane4 + 3], "int32")
            body_b = T.cast(Tiles[base + h["body_b"] + lane4 + 3], "int32")
            slot_a = T.cast(Tiles[base + h["diag_a"] + lane4 + 3], "int32")
            slot_b = T.cast(Tiles[base + h["off_a"] + lane4 + 3], "int32")
            pose_a = body_a * layout["pose_words"]
            pose_b = body_b * layout["pose_words"]
            velocity_a = slot_a * layout["velocity_words"]
            velocity_b = slot_b * layout["velocity_words"]
            direction = T.alloc_local((3,), "float32")
            tangent0 = T.alloc_local((3,), "float32")
            tangent1 = T.alloc_local((3,), "float32")
            im_a = T.alloc_local((3,), "float32")
            im_b = T.alloc_local((3,), "float32")
            diag_a = T.alloc_local((3,), "float32")
            off_a = T.alloc_local((3,), "float32")
            diag_b = T.alloc_local((3,), "float32")
            off_b = T.alloc_local((3,), "float32")
            va = T.alloc_local((6,), "float32")
            vb = T.alloc_local((6,), "float32")
            for d in T.serial(3):
                direction[d] = f32(Tiles, base + h["normal"] + lane4 + d)
                tangent0[d] = f32(Tiles, base + h["tangent"] + lane4 + d)
                im_a[d] = f32(Tiles, base + h["body_a"] + lane4 + d)
                im_b[d] = f32(Tiles, base + h["body_b"] + lane4 + d)
                diag_a[d] = f32(Tiles, base + h["diag_a"] + lane4 + d)
                off_a[d] = f32(Tiles, base + h["off_a"] + lane4 + d)
                diag_b[d] = f32(Tiles, base + h["diag_b"] + lane4 + d)
                off_b[d] = f32(Tiles, base + h["off_b"] + lane4 + d)
                va[d] = f32(Velocities, velocity_a + layout["velocity_linear"] + d)
                va[d + 3] = f32(Velocities, velocity_a + layout["velocity_angular"] + d)
                vb[d] = f32(Velocities, velocity_b + layout["velocity_linear"] + d)
                vb[d + 3] = f32(Velocities, velocity_b + layout["velocity_angular"] + d)
            is_static = ((im_a[0] == 0) & (im_a[1] == 0) & (im_a[2] == 0)) | (
                (im_b[0] == 0) & (im_b[1] == 0) & (im_b[2] == 0)
            )
            cfm = T.if_then_else(is_static, Params[4], Params[2])
            erp = T.if_then_else(is_static, Params[5], Params[3])
            friction_coefficient = f32(Tiles, base + h["tangent"] + lane4 + 3)
            cross(direction, tangent0, tangent1)
            arm_a = T.alloc_local((3,), "float32")
            arm_b = T.alloc_local((3,), "float32")
            local_a = T.alloc_local((3,), "float32")
            local_b = T.alloc_local((3,), "float32")
            world_a = T.alloc_local((3,), "float32")
            world_b = T.alloc_local((3,), "float32")
            torque_a = T.alloc_local((3,), "float32")
            torque_b = T.alloc_local((3,), "float32")
            response_a = T.alloc_local((3,), "float32")
            response_b = T.alloc_local((3,), "float32")
            for k in T.serial(4):
                if k < count:
                    pb = base + layout["points"] + k * layout["point_words"] + lane4
                    for d in T.serial(3):
                        arm_a[d] = f32(Tiles, pb + p["arm_a"] + d)
                        arm_b[d] = f32(Tiles, pb + p["arm_b"] + d)
                        local_a[d] = f32(Tiles, pb + p["local_a"] + d)
                        local_b[d] = f32(Tiles, pb + p["local_b"] + d)
                    transform_point(Poses, pose_a, local_a, world_a)
                    transform_point(Poses, pose_b, local_b, world_b)
                    distance = f32(Tiles, pb + p["local_a"] + 3) + (
                        (world_a[0] - world_b[0]) * direction[0]
                        + (world_a[1] - world_b[1]) * direction[1]
                        + (world_a[2] - world_b[2]) * direction[2]
                    )
                    rhs = T.alloc_var(
                        "float32",
                        init=f32(Tiles, pb + p["local_b"] + 3)
                        + T.max(distance, 0.0) * Params[0],
                    )
                    cfm_row = T.alloc_var("float32", init=1.0)
                    if use_bias:
                        rhs += T.min(T.max(distance * erp, -Params[1]), 0.0)
                        # TileLang 0.1.15 emits invalid MSL for an if_then_else
                        # assigned to a mutable scalar. An explicit branch preserves semantics.
                        if distance <= 0.0:
                            cfm_row = cfm
                    cross(arm_a, direction, torque_a)
                    cross(arm_b, direction, torque_b)
                    for d in T.serial(3):
                        torque_b[d] = -torque_b[d]
                    dvel = (
                        relative_velocity(direction, torque_a, torque_b, va, vb) + rhs
                    )
                    old_impulse = f32(Tiles, pb + p["arm_a"] + 3)
                    normal_mass = f32(Tiles, pb + p["arm_b"] + 3)
                    updated_impulse = cfm_row * T.max(
                        old_impulse - normal_mass * dvel, 0.0
                    )
                    delta_impulse = updated_impulse - old_impulse
                    Tiles[pb + p["arm_a"] + 3] = T.reinterpret(
                        updated_impulse, "uint32"
                    )
                    inertia(diag_a, off_a, torque_a, response_a)
                    inertia(diag_b, off_b, torque_b, response_b)
                    for d in T.serial(3):
                        va[d] += direction[d] * im_a[d] * delta_impulse
                        va[d + 3] += response_a[d] * delta_impulse
                        vb[d] += direction[d] * im_b[d] * -delta_impulse
                        vb[d + 3] += response_b[d] * delta_impulse
            if solve_friction:
                torque_a1 = T.alloc_local((3,), "float32")
                torque_b1 = T.alloc_local((3,), "float32")
                for k in T.serial(4):
                    if k < count:
                        pb = base + layout["points"] + k * layout["point_words"] + lane4
                        for d in T.serial(3):
                            arm_a[d] = f32(Tiles, pb + p["arm_a"] + d)
                            arm_b[d] = f32(Tiles, pb + p["arm_b"] + d)
                        cross(arm_a, tangent0, torque_a)
                        cross(arm_b, tangent0, torque_b)
                        cross(arm_a, tangent1, torque_a1)
                        cross(arm_b, tangent1, torque_b1)
                        for d in T.serial(3):
                            torque_b[d] = -torque_b[d]
                            torque_b1[d] = -torque_b1[d]
                        dv0 = relative_velocity(tangent0, torque_a, torque_b, va, vb)
                        dv1 = relative_velocity(tangent1, torque_a1, torque_b1, va, vb)
                        k11 = f32(Tiles, pb + p["friction"] + 2)
                        k22 = f32(Tiles, pb + p["friction"] + 3)
                        k12 = f32(Tiles, base + layout["coupling"] + lane * 4 + k) * 0.5
                        determinant = k11 * k22 - k12 * k12
                        inverse = T.alloc_var("float32", init=0.0)
                        if (determinant < -1.0e-20) | (determinant > 1.0e-20):
                            inverse = 1.0 / determinant
                        old0 = f32(Tiles, pb + p["friction"])
                        old1 = f32(Tiles, pb + p["friction"] + 1)
                        candidate0 = old0 - (k22 * dv0 - k12 * dv1) * inverse
                        candidate1 = old1 - (k11 * dv1 - k12 * dv0) * inverse
                        magnitude = T.sqrt(
                            candidate0 * candidate0 + candidate1 * candidate1
                        )
                        limit = friction_coefficient * f32(Tiles, pb + p["arm_a"] + 3)
                        scale = T.alloc_var("float32", init=1.0)
                        if magnitude > limit:
                            scale = limit / magnitude
                        updated0 = candidate0 * scale
                        updated1 = candidate1 * scale
                        delta0 = updated0 - old0
                        delta1 = updated1 - old1
                        Tiles[pb + p["friction"]] = T.reinterpret(updated0, "uint32")
                        Tiles[pb + p["friction"] + 1] = T.reinterpret(
                            updated1, "uint32"
                        )
                        for d in T.serial(3):
                            torque_a[d] = torque_a[d] * delta0 + torque_a1[d] * delta1
                            torque_b[d] = torque_b[d] * delta0 + torque_b1[d] * delta1
                        inertia(diag_a, off_a, torque_a, response_a)
                        inertia(diag_b, off_b, torque_b, response_b)
                        for d in T.serial(3):
                            linear = tangent0[d] * delta0 + tangent1[d] * delta1
                            va[d] += linear * im_a[d]
                            va[d + 3] += response_a[d]
                            vb[d] -= linear * im_b[d]
                            vb[d + 3] += response_b[d]
            for d in T.serial(3):
                Velocities[velocity_a + layout["velocity_linear"] + d] = T.reinterpret(
                    va[d], "uint32"
                )
                Velocities[velocity_a + layout["velocity_angular"] + d] = T.reinterpret(
                    va[d + 3], "uint32"
                )
                Velocities[velocity_b + layout["velocity_linear"] + d] = T.reinterpret(
                    vb[d], "uint32"
                )
                Velocities[velocity_b + layout["velocity_angular"] + d] = T.reinterpret(
                    vb[d + 3], "uint32"
                )

    return solve_one


def warmstart_constraint(layout, case):
    n = case["count"]
    h, p = (layout["header"], layout["point"])

    @T.macro
    def cross(a, b, out):
        out[0] = a[1] * b[2] - a[2] * b[1]
        out[1] = a[2] * b[0] - a[0] * b[2]
        out[2] = a[0] * b[1] - a[1] * b[0]

    @T.macro
    def inertia(diag, off, v, out):
        out[0] = diag[0] * v[0] + off[0] * v[1] + off[1] * v[2]
        out[1] = off[0] * v[0] + diag[1] * v[1] + off[2] * v[2]
        out[2] = off[1] * v[0] + off[2] * v[1] + diag[2] * v[2]

    @T.macro
    def one(Tiles, Velocities, Poses, Params, i):
        if i < n:
            base = i // 32 * layout["tile_words"]
            lane = i % 32
            lane4 = lane * 4
            count = Tiles[base + h["normal"] + lane4 + 3]
            direction = T.alloc_local((3,), "float32")
            tangent0 = T.alloc_local((3,), "float32")
            tangent1 = T.alloc_local((3,), "float32")
            im_a = T.alloc_local((3,), "float32")
            im_b = T.alloc_local((3,), "float32")
            diag_a = T.alloc_local((3,), "float32")
            diag_b = T.alloc_local((3,), "float32")
            off_a = T.alloc_local((3,), "float32")
            off_b = T.alloc_local((3,), "float32")
            force = T.alloc_local((3,), "float32")
            torque_a = T.alloc_local((3,), "float32")
            torque_b = T.alloc_local((3,), "float32")
            for d in T.serial(3):
                direction[d] = f32(Tiles, base + h["normal"] + lane4 + d)
                tangent0[d] = f32(Tiles, base + h["tangent"] + lane4 + d)
                im_a[d] = f32(Tiles, base + h["body_a"] + lane4 + d)
                im_b[d] = f32(Tiles, base + h["body_b"] + lane4 + d)
                diag_a[d] = f32(Tiles, base + h["diag_a"] + lane4 + d)
                diag_b[d] = f32(Tiles, base + h["diag_b"] + lane4 + d)
                off_a[d] = f32(Tiles, base + h["off_a"] + lane4 + d)
                off_b[d] = f32(Tiles, base + h["off_b"] + lane4 + d)
                force[d] = 0.0
                torque_a[d] = 0.0
                torque_b[d] = 0.0
            cross(direction, tangent0, tangent1)
            for k in T.serial(4):
                if k < count:
                    pb = base + layout["points"] + k * layout["point_words"] + lane4
                    ra = T.alloc_local((3,), "float32")
                    rb = T.alloc_local((3,), "float32")
                    ta = T.alloc_local((3,), "float32")
                    tb = T.alloc_local((3,), "float32")
                    ia = T.alloc_local((3,), "float32")
                    ib = T.alloc_local((3,), "float32")
                    impulse = T.alloc_local((3,), "float32")
                    for d in T.serial(3):
                        ra[d] = f32(Tiles, pb + p["arm_a"] + d)
                        rb[d] = f32(Tiles, pb + p["arm_b"] + d)
                    ni = f32(Tiles, pb + p["arm_a"] + 3)
                    t0 = f32(Tiles, pb + p["friction"])
                    t1 = f32(Tiles, pb + p["friction"] + 1)
                    for d in T.serial(3):
                        impulse[d] = (
                            direction[d] * ni + tangent0[d] * t0 + tangent1[d] * t1
                        )
                        force[d] = force[d] + impulse[d]
                    cross(ra, impulse, ta)
                    cross(rb, impulse, tb)
                    for d in T.serial(3):
                        torque_a[d] = torque_a[d] + ta[d]
                        torque_b[d] = torque_b[d] + tb[d]
            ia = T.alloc_local((3,), "float32")
            ib = T.alloc_local((3,), "float32")
            inertia(diag_a, off_a, torque_a, ia)
            inertia(diag_b, off_b, torque_b, ib)
            va = (
                T.cast(Tiles[base + h["diag_a"] + lane4 + 3], "int32")
                * layout["velocity_words"]
            )
            vb = (
                T.cast(Tiles[base + h["off_a"] + lane4 + 3], "int32")
                * layout["velocity_words"]
            )
            for d in T.serial(3):
                Velocities[va + layout["velocity_linear"] + d] = T.reinterpret(
                    f32(Velocities, va + layout["velocity_linear"] + d)
                    + im_a[d] * force[d],
                    "uint32",
                )
                Velocities[vb + layout["velocity_linear"] + d] = T.reinterpret(
                    f32(Velocities, vb + layout["velocity_linear"] + d)
                    - im_b[d] * force[d],
                    "uint32",
                )
                Velocities[va + layout["velocity_angular"] + d] = T.reinterpret(
                    f32(Velocities, va + layout["velocity_angular"] + d) + ia[d],
                    "uint32",
                )
                Velocities[vb + layout["velocity_angular"] + d] = T.reinterpret(
                    f32(Velocities, vb + layout["velocity_angular"] + d) - ib[d],
                    "uint32",
                )

    return one
