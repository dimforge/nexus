"""rbd.contact.coulomb.solve3d: one conflict-free color of existing Nexus tiles.

Arithmetic follows ContactSolverData::solve_points: all normal rows, then all
friction rows, with sequential updates within each manifold. Integer fields
remain uint32 bits. ABI offsets come from the compiled Rust reference exporter.
Launch wrappers are deliberately separate from the shared arithmetic.
"""

import tilelang.language as T


def f32(buffer, index):
    return T.reinterpret(buffer[index], "float32")


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def relative_velocity(direction, torque_a, torque_b, va, vb):
    return (
        direction[0] * va[0]
        + direction[1] * va[1]
        + direction[2] * va[2]
        + (torque_a[0] * va[3] + torque_a[1] * va[4] + torque_a[2] * va[5])
        - (direction[0] * vb[0] + direction[1] * vb[1] + direction[2] * vb[2])
        + (torque_b[0] * vb[3] + torque_b[1] * vb[4] + torque_b[2] * vb[5])
    )


def make_kernel(layout, case, target):
    if (
        layout["version"] != 1
        or layout["tile_len"] != 32
        or layout["max_points"] != 4
        or layout["vec_scalar_words"] != 4
        or layout["friction_words"] != 4
    ):
        raise ValueError("Unsupported Rust replay ABI")
    n = case["count"]
    tile_words = layout["tile_words"]
    nt = case["tiles"] * tile_words
    nv = case["bodies"] * layout["velocity_words"]
    np = case["bodies"] * layout["pose_words"]
    h, p = layout["header"], layout["point"]
    use_bias, solve_friction = case["use_bias"], case["solve_friction"]
    threads = 64

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

    if target == "c":

        @T.prim_func
        def kernel(
            Tiles: T.Tensor((nt,), "uint32"),
            Velocities: T.Tensor((nv,), "uint32"),
            Poses: T.Tensor((np,), "uint32"),
            Params: T.Tensor((6,), "float32"),
        ):
            with T.Kernel(1):
                for index in T.serial(n):
                    solve_one(Tiles, Velocities, Poses, Params, index)
    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            Tiles: A.Tensor((nt,), "uint32"),
            Velocities: A.Tensor((nv,), "uint32"),
            Poses: A.Tensor((np,), "uint32"),
            Params: A.Tensor((6,), "float32"),
        ):
            with A.Kernel(max((n + threads - 1) // threads, 1)) as block:
                with A.SimtVF(threads=threads):
                    solve_one(
                        Tiles,
                        Velocities,
                        Poses,
                        Params,
                        block * threads + A.get_thread_binding(),
                    )
    elif target in ("metal", "cuda"):

        @T.prim_func
        def kernel(
            Tiles: T.Tensor((nt,), "uint32"),
            Velocities: T.Tensor((nv,), "uint32"),
            Poses: T.Tensor((np,), "uint32"),
            Params: T.Tensor((6,), "float32"),
        ):
            with T.Kernel(
                max((n + threads - 1) // threads, 1), threads=threads
            ) as block:
                solve_one(
                    Tiles,
                    Velocities,
                    Poses,
                    Params,
                    block * threads + T.get_thread_binding(),
                )
    else:
        raise ValueError(f"Unsupported target: {target}")
    return kernel
