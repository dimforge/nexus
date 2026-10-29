# ruff: noqa: SIM102
# Keep compile-time kind branches separate from runtime TIR predicates.
"""Counterpart of dynamics/joint_constraint_builder.rs; ordinary rigid joints."""

import tilelang.language as T
from kernel_support import dot, f32


def orthogonalize_constraints(l):
    c, e = l["joint_constraint"], l["joint_element"]

    @T.macro
    def one(Constraints, slot, start, end):
        base = slot * l["joint_constraint_words"]
        mass = T.alloc_local((3,), "float32")
        for d in T.serial(3):
            mass[d] = f32(Constraints, base + c["im_a"] + d) + f32(
                Constraints, base + c["im_b"] + d
            )
        for j in T.serial(start, end):
            jb = base + c["elements"] + j * l["joint_element_words"]
            lin = T.alloc_local((3,), "float32")
            angular_a = T.alloc_local((3,), "float32")
            angular_b = T.alloc_local((3,), "float32")
            ia = T.alloc_local((3,), "float32")
            ib = T.alloc_local((3,), "float32")
            ml = T.alloc_local((3,), "float32")
            for d in T.serial(3):
                lin[d] = f32(Constraints, jb + e["lin_jac"] + d)
                angular_a[d] = f32(Constraints, jb + e["ang_jac_a"] + d)
                angular_b[d] = f32(Constraints, jb + e["ang_jac_b"] + d)
                ia[d] = f32(Constraints, jb + e["ii_ang_jac_a"] + d)
                ib[d] = f32(Constraints, jb + e["ii_ang_jac_b"] + d)
                ml[d] = mass[d] * lin[d]
            dotjj = dot(lin, ml) + dot(ia, angular_a) + dot(ib, angular_b)
            gain = dotjj * f32(Constraints, jb + e["cfm_coeff"]) + f32(
                Constraints, jb + e["cfm_gain"]
            )
            inverse = T.if_then_else(dotjj == 0.0, 0.0, 1.0 / dotjj)
            lhs = dotjj + gain
            Constraints[jb + e["inv_lhs"]] = T.reinterpret(
                T.if_then_else(lhs == 0.0, 0.0, 1.0 / lhs), "uint32"
            )
            Constraints[jb + e["cfm_gain"]] = T.reinterpret(gain, "uint32")
            maximum = T.reinterpret(T.uint32(l["constants"]["max_flt_bits"]), "float32")
            if (f32(Constraints, jb + e["impulse_bounds"]) == -maximum) & (
                f32(Constraints, jb + e["impulse_bounds"] + 1) == maximum
            ):
                for i in T.serial(j + 1, end):
                    dest = base + c["elements"] + i * l["joint_element_words"]
                    dl = T.alloc_local((3,), "float32")
                    da = T.alloc_local((3,), "float32")
                    db = T.alloc_local((3,), "float32")
                    for d in T.serial(3):
                        dl[d] = f32(Constraints, dest + e["lin_jac"] + d)
                        da[d] = f32(Constraints, dest + e["ii_ang_jac_a"] + d)
                        db[d] = f32(Constraints, dest + e["ii_ang_jac_b"] + d)
                    coeff = (
                        dot(dl, ml) + dot(da, angular_a) + dot(db, angular_b)
                    ) * inverse
                    for d in T.serial(3):
                        Constraints[dest + e["lin_jac"] + d] = T.reinterpret(
                            f32(Constraints, dest + e["lin_jac"] + d) - lin[d] * coeff,
                            "uint32",
                        )
                        Constraints[dest + e["ang_jac_a"] + d] = T.reinterpret(
                            f32(Constraints, dest + e["ang_jac_a"] + d)
                            - angular_a[d] * coeff,
                            "uint32",
                        )
                        Constraints[dest + e["ang_jac_b"] + d] = T.reinterpret(
                            f32(Constraints, dest + e["ang_jac_b"] + d)
                            - angular_b[d] * coeff,
                            "uint32",
                        )
                        Constraints[dest + e["ii_ang_jac_a"] + d] = T.reinterpret(
                            f32(Constraints, dest + e["ii_ang_jac_a"] + d)
                            - ia[d] * coeff,
                            "uint32",
                        )
                        Constraints[dest + e["ii_ang_jac_b"] + d] = T.reinterpret(
                            f32(Constraints, dest + e["ii_ang_jac_b"] + d)
                            - ib[d] * coeff,
                            "uint32",
                        )
                    Constraints[dest + e["rhs_wo_bias"]] = T.reinterpret(
                        f32(Constraints, dest + e["rhs_wo_bias"])
                        - f32(Constraints, jb + e["rhs_wo_bias"]) * coeff,
                        "uint32",
                    )
                    Constraints[dest + e["rhs"]] = T.reinterpret(
                        f32(Constraints, dest + e["rhs"])
                        - f32(Constraints, jb + e["rhs"]) * coeff,
                        "uint32",
                    )

    return one


def solve_joint_constraint(l):
    c, e = l["joint_constraint"], l["joint_element"]

    @T.macro
    def one(Constraints, Velocities, slot, use_bias):
        base = slot * l["joint_constraint_words"]
        a = T.cast(Constraints[base + c["solver_vel_a"]], "int32") * l["velocity_words"]
        b = T.cast(Constraints[base + c["solver_vel_b"]], "int32") * l["velocity_words"]
        va = T.alloc_local((3,), "float32")
        wa = T.alloc_local((3,), "float32")
        vb = T.alloc_local((3,), "float32")
        wb = T.alloc_local((3,), "float32")
        for d in T.serial(3):
            va[d] = f32(Velocities, a + l["velocity_linear"] + d)
            wa[d] = f32(Velocities, a + l["velocity_angular"] + d)
            vb[d] = f32(Velocities, b + l["velocity_linear"] + d)
            wb[d] = f32(Velocities, b + l["velocity_angular"] + d)
        for i in T.serial(T.cast(Constraints[base + c["len"]], "int32")):
            eb = base + c["elements"] + i * l["joint_element_words"]
            lin = T.alloc_local((3,), "float32")
            ang_a = T.alloc_local((3,), "float32")
            ang_b = T.alloc_local((3,), "float32")
            dv = T.alloc_local((3,), "float32")
            for d in T.serial(3):
                lin[d] = f32(Constraints, eb + e["lin_jac"] + d)
                ang_a[d] = f32(Constraints, eb + e["ang_jac_a"] + d)
                ang_b[d] = f32(Constraints, eb + e["ang_jac_b"] + d)
                dv[d] = vb[d] - va[d]
            dlin = dot(lin, dv)
            dang = dot(ang_b, wb) - dot(ang_a, wa)
            rhs = (
                dlin
                + dang
                + T.if_then_else(
                    use_bias,
                    f32(Constraints, eb + e["rhs"]),
                    f32(Constraints, eb + e["rhs_wo_bias"]),
                )
            )
            impulse = f32(Constraints, eb + e["impulse"])
            total = T.min(
                T.max(
                    impulse
                    + f32(Constraints, eb + e["inv_lhs"])
                    * (rhs - f32(Constraints, eb + e["cfm_gain"]) * impulse),
                    f32(Constraints, eb + e["impulse_bounds"]),
                ),
                f32(Constraints, eb + e["impulse_bounds"] + 1),
            )
            delta = total - impulse
            Constraints[eb + e["impulse"]] = T.reinterpret(total, "uint32")
            for d in T.serial(3):
                linear_impulse = lin[d] * delta
                va[d] = va[d] + linear_impulse * f32(Constraints, base + c["im_a"] + d)
                wa[d] = wa[d] + f32(Constraints, eb + e["ii_ang_jac_a"] + d) * delta
                vb[d] = vb[d] - linear_impulse * f32(Constraints, base + c["im_b"] + d)
                wb[d] = wb[d] - f32(Constraints, eb + e["ii_ang_jac_b"] + d) * delta
        for d in T.serial(3):
            Velocities[a + l["velocity_linear"] + d] = T.reinterpret(va[d], "uint32")
            Velocities[a + l["velocity_angular"] + d] = T.reinterpret(wa[d], "uint32")
            Velocities[b + l["velocity_linear"] + d] = T.reinterpret(vb[d], "uint32")
            Velocities[b + l["velocity_angular"] + d] = T.reinterpret(wb[d], "uint32")

    return one


@T.macro
def gcross_matrix(r, Out):
    Out[0, 0] = 0.0
    Out[0, 1] = r[2]
    Out[0, 2] = -r[1]
    Out[1, 0] = -r[2]
    Out[1, 1] = 0.0
    Out[1, 2] = r[0]
    Out[2, 0] = r[1]
    Out[2, 1] = -r[0]
    Out[2, 2] = 0.0


@T.macro
def rotation_diff_conj1_2(Q1, Q2, Out):
    C1 = T.alloc_local((3, 3), "float32")
    C2 = T.alloc_local((3, 3), "float32")
    CS = T.alloc_local((3, 3), "float32")
    v1 = T.alloc_local((3,), "float32")
    v2 = T.alloc_local((3,), "float32")
    vs = T.alloc_local((3,), "float32")
    for d in T.serial(3):
        v1[d] = Q1[d]
        v2[d] = Q2[d]
        vs[d] = Q1[d] * Q2[3] + Q2[d] * Q1[3]
    gcross_matrix(v1, C1)
    gcross_matrix(v2, C2)
    gcross_matrix(vs, CS)
    for col in T.serial(3):
        for row in T.serial(3):
            product = (C1[0, row] * C2[col, 0] + C1[1, row] * C2[col, 1]) + C1[
                2, row
            ] * C2[col, 2]
            diag = T.if_then_else(col == row, Q1[3] * Q2[3], 0.0)
            Out[col, row] = (
                ((Q1[row] * Q2[col] + diag) - CS[col, row]) + product
            ) * 0.5


def new_helper(l):
    from kernel_support import quat_mul, quat_to_mat3

    @T.macro
    def one(Q1, Q2, P1, P2, COM1, COM2, locked, H, LinErr, AngErr):
        B1 = T.alloc_local((3, 3), "float32")
        B2 = T.alloc_local((3, 3), "float32")
        C1 = T.alloc_local((3, 3), "float32")
        C2 = T.alloc_local((3, 3), "float32")
        Angular = T.alloc_local((3, 3), "float32")
        center = T.alloc_local((3,), "float32")
        r1 = T.alloc_local((3,), "float32")
        r2 = T.alloc_local((3,), "float32")
        quat_to_mat3(Q1, B1)
        quat_to_mat3(Q2, B2)
        for d in T.serial(3):
            LinErr[d] = P2[d] - P1[d]
            center[d] = P2[d]
        for axis in T.serial(3):
            if (locked & T.shift_left(T.uint32(1), axis)) != 0:
                projection = (
                    B1[axis, 0] * LinErr[0]
                    + B1[axis, 1] * LinErr[1]
                    + B1[axis, 2] * LinErr[2]
                )
                for d in T.serial(3):
                    center[d] = center[d] - B1[axis, d] * projection
        for d in T.serial(3):
            r1[d] = center[d] - COM1[d]
            r2[d] = P2[d] - COM2[d]
        gcross_matrix(r1, C1)
        gcross_matrix(r2, C2)
        rotation_diff_conj1_2(Q1, Q2, Angular)
        inverse = T.alloc_local((4,), "float32")
        error = T.alloc_local((4,), "float32")
        for d in T.serial(4):
            inverse[d] = T.if_then_else(d == 3, Q1[d], -Q1[d])
        quat_mul(inverse, Q2, error)
        sign = T.if_then_else(
            ((Q1[0] * Q2[0] + Q1[1] * Q2[1]) + Q1[2] * Q2[2]) + Q1[3] * Q2[3] > 0.0,
            1.0,
            -1.0,
        )
        for col in T.serial(3):
            AngErr[col] = error[col] * sign
            for row in T.serial(3):
                H[0, col, row] = B1[col, row]
                H[1, col, row] = B2[col, row]
                H[2, col, row] = (
                    C1[0, row] * B1[col, 0] + C1[1, row] * B1[col, 1]
                ) + C1[2, row] * B1[col, 2]
                H[3, col, row] = (
                    C2[0, row] * B1[col, 0] + C2[1, row] * B1[col, 1]
                ) + C2[2, row] * B1[col, 2]
                H[4, col, row] = Angular[row, col] * sign

    return one


def world_frame(l):
    from kernel_support import quat_mul, quat_rotate

    @T.macro
    def one(Poses, body, Builders, frame, Q, Position, COM):
        bq = T.alloc_local((4,), "float32")
        fq = T.alloc_local((4,), "float32")
        ft = T.alloc_local((3,), "float32")
        rotated = T.alloc_local((3,), "float32")
        for d in T.serial(4):
            bq[d] = f32(Poses, body * l["pose_words"] + l["pose_rotation"] + d)
            fq[d] = f32(Builders, frame + l["pose_rotation"] + d)
        for d in T.serial(3):
            ft[d] = f32(Builders, frame + l["pose_translation"] + d)
            COM[d] = f32(Poses, body * l["pose_words"] + l["pose_translation"] + d)
        quat_mul(bq, fq, Q)
        quat_rotate(bq, ft, rotated)
        for d in T.serial(3):
            Position[d] = COM[d] + rotated[d]

    return one


def make_element(l, kind):
    from .joint_constraint import ii_mul

    inertia = ii_mul(l)
    c, e = l["joint_constraint"], l["joint_element"]

    @T.macro
    def one(
        Constraints,
        slot,
        index,
        joint_id,
        H,
        LinErr,
        AngErr,
        WorldMass,
        body_a,
        body_b,
        axis,
        coupled,
        lo,
        hi,
        Motor,
        Soft,
    ):
        lin = T.alloc_local((3,), "float32")
        aa = T.alloc_local((3,), "float32")
        ab = T.alloc_local((3,), "float32")
        ia = T.alloc_local((3,), "float32")
        ib = T.alloc_local((3,), "float32")
        dist = T.alloc_local((1,), "float32")
        dist[0] = 0.0
        if kind == "lock_angular" or kind == "limit_angular" or kind == "motor_angular":
            for d in T.serial(3):
                lin[d] = 0.0
                aa[d] = H[0 if kind == "motor_angular" else 4, axis, d]
                ab[d] = aa[d]
        elif kind == "limit_linear_coupled" or kind == "motor_linear_coupled":
            for d in T.serial(3):
                lin[d] = 0.0
                aa[d] = 0.0
                ab[d] = 0.0
            for ax in T.serial(3):
                if (coupled & T.shift_left(T.uint32(1), ax)) != 0:
                    coeff = (
                        H[0, ax, 0] * LinErr[0]
                        + H[0, ax, 1] * LinErr[1]
                        + H[0, ax, 2] * LinErr[2]
                    )
                    for d in T.serial(3):
                        lin[d] = lin[d] + H[0, ax, d] * coeff
                        aa[d] = aa[d] + H[2, ax, d] * coeff
                        ab[d] = ab[d] + H[3, ax, d] * coeff
            dist[0] = T.sqrt(dot(lin, lin))
            inv = T.if_then_else(dist[0] == 0.0, 0.0, 1.0 / dist[0])
            for d in T.serial(3):
                lin[d] = lin[d] * inv
                aa[d] = aa[d] * inv
                ab[d] = ab[d] * inv
        else:
            for d in T.serial(3):
                lin[d] = H[0, axis, d]
                aa[d] = H[2, axis, d]
                ab[d] = H[3, axis, d]
            dist[0] = dot(LinErr, lin)
        inertia(WorldMass, body_a, aa, ia)
        inertia(WorldMass, body_b, ab, ib)
        maximum = T.reinterpret(T.uint32(l["constants"]["max_flt_bits"]), "float32")
        rhs = T.alloc_local((1,), "float32")
        unbiased = T.alloc_local((1,), "float32")
        lower = T.alloc_local((1,), "float32")
        upper = T.alloc_local((1,), "float32")
        gain = T.alloc_local((1,), "float32")
        cfm = T.alloc_local((1,), "float32")
        rhs[0] = 0.0
        unbiased[0] = 0.0
        lower[0] = -maximum
        upper[0] = maximum
        gain[0] = 0.0
        cfm[0] = Soft[1]
        if kind == "lock_linear":
            rhs[0] = 0.0 + dist[0] * Soft[0]
        if kind == "lock_angular":
            rhs[0] = 0.0 + AngErr[axis] * Soft[0]
        if kind == "limit_linear":
            rhs[0] = (
                0.0 + (T.max(dist[0] - hi, 0.0) - T.max(lo - dist[0], 0.0)) * Soft[0]
            )
            lower[0] = T.if_then_else(dist[0] <= lo, -maximum, 0.0)
            upper[0] = T.if_then_else(hi <= dist[0], maximum, 0.0)
        if kind == "limit_linear_coupled":
            unbiased[0] = T.min(dist[0] - hi, 0.0) * Soft[2]
            rhs[0] = unbiased[0] + T.max(dist[0] - hi, 0.0) * Soft[0]
            lower[0] = 0.0
        if kind == "limit_angular":
            min_sin = T.sin(lo * 0.5)
            max_sin = T.sin(hi * 0.5)
            angle = AngErr[axis]
            lower[0] = T.if_then_else(angle <= min_sin, -maximum, 0.0)
            upper[0] = T.if_then_else(max_sin <= angle, maximum, 0.0)
            rhs[0] = (
                0.0
                + (T.max(angle - max_sin, 0.0) - T.max(min_sin - angle, 0.0)) * Soft[0]
            )
        if (
            kind == "motor_linear"
            or kind == "motor_linear_coupled"
            or kind == "motor_angular"
        ):
            cfm[0] = Motor[1]
            gain[0] = Motor[2]
            lower[0] = -Motor[5]
            upper[0] = Motor[5]
            if Motor[0] != 0.0:
                if kind == "motor_angular":
                    angle = T.asin(T.min(T.max(AngErr[axis], -1.0), 1.0)) * 2.0
                    error = angle - Motor[3]
                    sign = T.if_then_else(error < 0.0, -1.0, 1.0)
                    complement = error - sign * T.reinterpret(
                        T.uint32(l["constants"]["two_pi_bits"]), "float32"
                    )
                    delta = T.if_then_else(
                        T.if_then_else(error < 0.0, -error, error)
                        < T.if_then_else(complement < 0.0, -complement, complement),
                        error,
                        complement,
                    )
                    unbiased[0] = unbiased[0] + delta * Motor[0]
                else:
                    unbiased[0] = unbiased[0] + (dist[0] - Motor[3]) * Motor[0]
            target = T.alloc_local((1,), "float32")
            target[0] = Motor[4]
            if kind != "motor_angular":
                if (lo != -maximum) | (hi != maximum):
                    target[0] = T.min(
                        T.max(target[0], (lo - dist[0]) * Soft[2]),
                        (hi - dist[0]) * Soft[2],
                    )
            unbiased[0] = unbiased[0] + (-target[0])
            rhs[0] = unbiased[0]
        eb = (
            slot * l["joint_constraint_words"]
            + c["elements"]
            + index * l["joint_element_words"]
        )
        for d in T.serial(3):
            Constraints[eb + e["lin_jac"] + d] = T.reinterpret(lin[d], "uint32")
            Constraints[eb + e["ang_jac_a"] + d] = T.reinterpret(aa[d], "uint32")
            Constraints[eb + e["ang_jac_b"] + d] = T.reinterpret(ab[d], "uint32")
            Constraints[eb + e["ii_ang_jac_a"] + d] = T.reinterpret(ia[d], "uint32")
            Constraints[eb + e["ii_ang_jac_b"] + d] = T.reinterpret(ib[d], "uint32")
        Constraints[eb + e["joint_id"]] = joint_id
        Constraints[eb + e["impulse"]] = T.uint32(0)
        Constraints[eb + e["inv_lhs"]] = T.uint32(0)
        Constraints[eb + e["rhs"]] = T.reinterpret(rhs[0], "uint32")
        Constraints[eb + e["rhs_wo_bias"]] = T.reinterpret(unbiased[0], "uint32")
        Constraints[eb + e["cfm_gain"]] = T.reinterpret(gain[0], "uint32")
        Constraints[eb + e["cfm_coeff"]] = T.reinterpret(cfm[0], "uint32")
        Constraints[eb + e["impulse_bounds"]] = T.reinterpret(lower[0], "uint32")
        Constraints[eb + e["impulse_bounds"] + 1] = T.reinterpret(upper[0], "uint32")

    return one


def lock_linear(l):
    return make_element(l, "lock_linear")


def lock_angular(l):
    return make_element(l, "lock_angular")


def limit_linear(l):
    return make_element(l, "limit_linear")


def limit_linear_coupled(l):
    return make_element(l, "limit_linear_coupled")


def motor_linear(l):
    return make_element(l, "motor_linear")


def motor_linear_coupled(l):
    return make_element(l, "motor_linear_coupled")


def limit_angular(l):
    return make_element(l, "limit_angular")


def motor_angular(l):
    return make_element(l, "motor_angular")


def update_constraint(l):
    from .joint import motor_params
    from .sim_params import joint_coefficients

    helper = new_helper(l)
    frame = world_frame(l)
    softness = joint_coefficients(l)
    motor = motor_params(l)
    orthogonalize = orthogonalize_constraints(l)
    lock_lin = lock_linear(l)
    lock_ang = lock_angular(l)
    limit_lin = limit_linear(l)
    limit_ang = limit_angular(l)
    motor_lin = motor_linear(l)
    motor_ang = motor_angular(l)
    coupled_motor = motor_linear_coupled(l)
    coupled_limit = limit_linear_coupled(l)
    b, g, limit = l["builder"], l["generic_joint"], l["limits"]

    @T.macro
    def one(Builders, Constraints, Poses, WorldMass, Params, slot):
        base = slot * l["builder_words"]
        jb = base + b["joint"]
        joint_id = Builders[base + b["constraint_id"]]
        ba = T.cast(Builders[base + b["body1"]], "int32")
        bb = T.cast(Builders[base + b["body2"]], "int32")
        Q1 = T.alloc_local((4,), "float32")
        Q2 = T.alloc_local((4,), "float32")
        P1 = T.alloc_local((3,), "float32")
        P2 = T.alloc_local((3,), "float32")
        COM1 = T.alloc_local((3,), "float32")
        COM2 = T.alloc_local((3,), "float32")
        frame(Poses, ba, Builders, jb + g["local_frame_a"], Q1, P1, COM1)
        frame(Poses, bb, Builders, jb + g["local_frame_b"], Q2, P2, COM2)
        locked = Builders[jb + g["locked_axes"]]
        motor_axes = Builders[jb + g["motor_axes"]] & ~locked
        limit_axes = Builders[jb + g["limit_axes"]] & ~locked
        coupled = Builders[jb + g["coupled_axes"]]
        H = T.alloc_local((5, 3, 3), "float32")
        LinErr = T.alloc_local((3,), "float32")
        AngErr = T.alloc_local((3,), "float32")
        helper(Q1, Q2, P1, P2, COM1, COM2, locked, H, LinErr, AngErr)
        Soft = T.alloc_local((3,), "float32")
        Motor = T.alloc_local((6,), "float32")
        softness(Params, Soft)
        for d in T.serial(6):
            Motor[d] = 0.0
        length = T.alloc_local((1,), "int32")
        length[0] = 0
        first = T.alloc_local((1,), "int32")
        first[0] = 3
        for axis in T.serial(3):
            if (coupled & T.shift_left(T.uint32(1), axis)) != 0:
                if first[0] == 3:
                    first[0] = axis
        maximum = T.reinterpret(T.uint32(l["constants"]["max_flt_bits"]), "float32")
        for axis in T.serial(3, 6):
            if ((motor_axes & ~coupled) & T.shift_left(T.uint32(1), axis)) != 0:
                motor(
                    Builders, jb + g["motors"] + axis * l["motor_words"], Params, Motor
                )
                motor_ang(
                    Constraints,
                    slot,
                    length[0],
                    joint_id,
                    H,
                    LinErr,
                    AngErr,
                    WorldMass,
                    ba,
                    bb,
                    axis - 3,
                    coupled,
                    -maximum,
                    maximum,
                    Motor,
                    Soft,
                )
                length[0] = length[0] + 1
        for axis in T.serial(3):
            if ((motor_axes & ~coupled) & T.shift_left(T.uint32(1), axis)) != 0:
                low = T.alloc_local((1,), "float32")
                high = T.alloc_local((1,), "float32")
                low[0] = -maximum
                high[0] = maximum
                if (limit_axes & T.shift_left(T.uint32(1), axis)) != 0:
                    low[0] = f32(
                        Builders,
                        jb + g["limits"] + axis * l["limits_words"] + limit["min"],
                    )
                    high[0] = f32(
                        Builders,
                        jb + g["limits"] + axis * l["limits_words"] + limit["max"],
                    )
                motor(
                    Builders, jb + g["motors"] + axis * l["motor_words"], Params, Motor
                )
                motor_lin(
                    Constraints,
                    slot,
                    length[0],
                    joint_id,
                    H,
                    LinErr,
                    AngErr,
                    WorldMass,
                    ba,
                    bb,
                    axis,
                    coupled,
                    low[0],
                    high[0],
                    Motor,
                    Soft,
                )
                length[0] = length[0] + 1
        # Coupled angular motors are also unimplemented by the Rust reference.
        if (motor_axes & coupled & T.uint32(7)) != 0:
            low = T.alloc_local((1,), "float32")
            high = T.alloc_local((1,), "float32")
            low[0] = -maximum
            high[0] = maximum
            if (limit_axes & T.shift_left(T.uint32(1), first[0])) != 0:
                low[0] = f32(
                    Builders,
                    jb + g["limits"] + first[0] * l["limits_words"] + limit["min"],
                )
                high[0] = f32(
                    Builders,
                    jb + g["limits"] + first[0] * l["limits_words"] + limit["max"],
                )
            motor(
                Builders, jb + g["motors"] + first[0] * l["motor_words"], Params, Motor
            )
            coupled_motor(
                Constraints,
                slot,
                length[0],
                joint_id,
                H,
                LinErr,
                AngErr,
                WorldMass,
                ba,
                bb,
                first[0],
                coupled,
                low[0],
                high[0],
                Motor,
                Soft,
            )
            length[0] = length[0] + 1
        orthogonalize(Constraints, slot, 0, length[0])
        start = length[0]
        for axis in T.serial(3, 6):
            if (locked & T.shift_left(T.uint32(1), axis)) != 0:
                lock_ang(
                    Constraints,
                    slot,
                    length[0],
                    joint_id,
                    H,
                    LinErr,
                    AngErr,
                    WorldMass,
                    ba,
                    bb,
                    axis - 3,
                    coupled,
                    -maximum,
                    maximum,
                    Motor,
                    Soft,
                )
                length[0] = length[0] + 1
        for axis in T.serial(3):
            if (locked & T.shift_left(T.uint32(1), axis)) != 0:
                lock_lin(
                    Constraints,
                    slot,
                    length[0],
                    joint_id,
                    H,
                    LinErr,
                    AngErr,
                    WorldMass,
                    ba,
                    bb,
                    axis,
                    coupled,
                    -maximum,
                    maximum,
                    Motor,
                    Soft,
                )
                length[0] = length[0] + 1
        for axis in T.serial(3, 6):
            if ((limit_axes & ~coupled) & T.shift_left(T.uint32(1), axis)) != 0:
                lo = f32(
                    Builders, jb + g["limits"] + axis * l["limits_words"] + limit["min"]
                )
                hi = f32(
                    Builders, jb + g["limits"] + axis * l["limits_words"] + limit["max"]
                )
                limit_ang(
                    Constraints,
                    slot,
                    length[0],
                    joint_id,
                    H,
                    LinErr,
                    AngErr,
                    WorldMass,
                    ba,
                    bb,
                    axis - 3,
                    coupled,
                    lo,
                    hi,
                    Motor,
                    Soft,
                )
                length[0] = length[0] + 1
        for axis in T.serial(3):
            if ((limit_axes & ~coupled) & T.shift_left(T.uint32(1), axis)) != 0:
                lo = f32(
                    Builders, jb + g["limits"] + axis * l["limits_words"] + limit["min"]
                )
                hi = f32(
                    Builders, jb + g["limits"] + axis * l["limits_words"] + limit["max"]
                )
                limit_lin(
                    Constraints,
                    slot,
                    length[0],
                    joint_id,
                    H,
                    LinErr,
                    AngErr,
                    WorldMass,
                    ba,
                    bb,
                    axis,
                    coupled,
                    lo,
                    hi,
                    Motor,
                    Soft,
                )
                length[0] = length[0] + 1
        if (coupled & T.uint32(7)) != 0:
            if (limit_axes & T.shift_left(T.uint32(1), first[0])) != 0:
                lo = f32(
                    Builders,
                    jb + g["limits"] + first[0] * l["limits_words"] + limit["min"],
                )
                hi = f32(
                    Builders,
                    jb + g["limits"] + first[0] * l["limits_words"] + limit["max"],
                )
                coupled_limit(
                    Constraints,
                    slot,
                    length[0],
                    joint_id,
                    H,
                    LinErr,
                    AngErr,
                    WorldMass,
                    ba,
                    bb,
                    first[0],
                    coupled,
                    lo,
                    hi,
                    Motor,
                    Soft,
                )
                length[0] = length[0] + 1
        orthogonalize(Constraints, slot, start, length[0])
        Constraints[
            slot * l["joint_constraint_words"] + l["joint_constraint"]["len"]
        ] = T.cast(length[0], "uint32")

    return one
