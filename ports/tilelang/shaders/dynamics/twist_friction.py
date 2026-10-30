"""Counterpart of dynamics/twist_friction.rs on its native compact tile layout.

Normal row order, central sliding rows and manifold twist limits follow Rust.
Header and point offsets are supplied by the current Rust ABI exporter.
"""

import tilelang.language as T
from kernel_support import dot, f32, quat_rotate, relative_velocity

from .solver_utils import writes_velocity_a, writes_velocity_b


def compute_effective_masses(layout, case):
    return _operation(layout, case, "mass")


def warmstart(layout, case):
    return _operation(layout, case, "warmstart")


def solve(layout, case):
    return _operation(layout, case, "solve")


def scale_impulses(layout, case):
    return _operation(layout, case, "scale")


def _operation(l, c, operation):
    h, t = l["header"], l["twist"]
    writable_a, writable_b = writes_velocity_a(l), writes_velocity_b(l)

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
    def rotated(Tiles, base, offset, v, out, inverse):
        q = T.alloc_local((4,), "float32")
        for d in T.serial(4):
            q[d] = f32(Tiles, base + offset + d)
        if inverse:
            for d in T.serial(3):
                q[d] = -q[d]
        quat_rotate(q, v, out)

    @T.macro
    def one(Tiles, Velocities, Poses, Params, index):
        if index < c["count"]:
            base = index // 32 * l["tile_words"] + index % 32 * 4
            count = T.cast(Tiles[base + h["normal"] + 3], "int32")
            if operation == "scale":
                for k in T.serial(4):
                    if k < count:
                        Tiles[base + t["normal_impulse"] + k] = T.reinterpret(
                            f32(Tiles, base + t["normal_impulse"] + k) * c["scale"],
                            "uint32",
                        )
                for d in T.serial(2):
                    Tiles[base + t["friction"] + d] = T.reinterpret(
                        f32(Tiles, base + t["friction"] + d) * c["scale"], "uint32"
                    )
                Tiles[base + h["off_b"] + 3] = T.reinterpret(
                    f32(Tiles, base + h["off_b"] + 3) * c["scale"], "uint32"
                )
            else:
                if count > 0:
                    normal = T.alloc_local((3,), "float32")
                    tangent = T.alloc_local((3,), "float32")
                    bitangent = T.alloc_local((3,), "float32")
                    im_a = T.alloc_local((3,), "float32")
                    im_b = T.alloc_local((3,), "float32")
                    da = T.alloc_local((3,), "float32")
                    db = T.alloc_local((3,), "float32")
                    oa = T.alloc_local((3,), "float32")
                    ob = T.alloc_local((3,), "float32")
                    offset = T.alloc_local((3,), "float32")
                    center = T.alloc_local((3,), "float32")
                    ra = T.alloc_local((3,), "float32")
                    rb = T.alloc_local((3,), "float32")
                    ta = T.alloc_local((3,), "float32")
                    tb = T.alloc_local((3,), "float32")
                    ia = T.alloc_local((3,), "float32")
                    ib = T.alloc_local((3,), "float32")
                    for d in T.serial(3):
                        normal[d] = f32(Tiles, base + h["normal"] + d)
                        tangent[d] = f32(Tiles, base + h["tangent"] + d)
                        im_a[d] = f32(Tiles, base + h["body_a"] + d)
                        im_b[d] = f32(Tiles, base + h["body_b"] + d)
                        da[d] = f32(Tiles, base + h["diag_a"] + d)
                        db[d] = f32(Tiles, base + h["diag_b"] + d)
                        oa[d] = f32(Tiles, base + h["off_a"] + d)
                        ob[d] = f32(Tiles, base + h["off_b"] + d)
                        offset[d] = f32(Tiles, base + t["offset_b"] + d)
                        center[d] = f32(Tiles, base + t["center_a"] + d)
                    cross(normal, tangent, bitangent)
                    if operation == "mass":
                        for d in T.serial(3):
                            center[d] = 0.0
                        for k in T.serial(4):
                            if k < count:
                                arm = base + t["arm_a"] + k * l["twist_arm_words"]
                                for d in T.serial(3):
                                    ra[d] = f32(Tiles, arm + d)
                                    rb[d] = ra[d] + offset[d]
                                    center[d] += ra[d]
                                cross(ra, normal, ta)
                                cross(rb, normal, tb)
                                for d in T.serial(3):
                                    tb[d] = -tb[d]
                                inertia(da, oa, ta, ia)
                                inertia(db, ob, tb, ib)
                                kn = (
                                    (
                                        normal[0] * ((im_a[0] + im_b[0]) * normal[0])
                                        + normal[1] * ((im_a[1] + im_b[1]) * normal[1])
                                        + normal[2] * ((im_a[2] + im_b[2]) * normal[2])
                                    )
                                    + dot(ia, ta)
                                    + dot(ib, tb)
                                )
                                Tiles[arm + 3] = T.reinterpret(
                                    T.if_then_else(kn == 0.0, 0.0, 1.0 / kn), "uint32"
                                )
                        for d in T.serial(3):
                            center[d] /= T.cast(count, "float32")
                            Tiles[base + t["center_a"] + d] = T.reinterpret(
                                center[d], "uint32"
                            )
                        for k in T.serial(4):
                            if k < count:
                                for d in T.serial(3):
                                    ra[d] = (
                                        f32(
                                            Tiles,
                                            base
                                            + t["arm_a"]
                                            + k * l["twist_arm_words"]
                                            + d,
                                        )
                                        - center[d]
                                    )
                                Tiles[base + t["radius"] + k] = T.reinterpret(
                                    T.sqrt(dot(ra, ra)), "uint32"
                                )
                        a0 = T.alloc_local((3,), "float32")
                        a1 = T.alloc_local((3,), "float32")
                        b0 = T.alloc_local((3,), "float32")
                        b1 = T.alloc_local((3,), "float32")
                        for d in T.serial(3):
                            rb[d] = center[d] + offset[d]
                        cross(center, tangent, a0)
                        cross(center, bitangent, a1)
                        cross(rb, tangent, b0)
                        cross(rb, bitangent, b1)
                        for d in T.serial(3):
                            b0[d] = -b0[d]
                            b1[d] = -b1[d]
                        inertia(da, oa, a0, ia)
                        inertia(db, ob, b0, ib)
                        k00 = (
                            (
                                tangent[0] * ((im_a[0] + im_b[0]) * tangent[0])
                                + tangent[1] * ((im_a[1] + im_b[1]) * tangent[1])
                                + tangent[2] * ((im_a[2] + im_b[2]) * tangent[2])
                            )
                            + dot(a0, ia)
                            + dot(b0, ib)
                        )
                        Tiles[base + t["offset_b"] + 3] = T.reinterpret(k00, "uint32")
                        Tiles[base + t["friction"] + 2] = T.reinterpret(
                            2.0 * (dot(ia, a1) + dot(ib, b1)), "uint32"
                        )
                        inertia(da, oa, a1, ia)
                        inertia(db, ob, b1, ib)
                        k11 = (
                            (
                                bitangent[0] * ((im_a[0] + im_b[0]) * bitangent[0])
                                + bitangent[1] * ((im_a[1] + im_b[1]) * bitangent[1])
                                + bitangent[2] * ((im_a[2] + im_b[2]) * bitangent[2])
                            )
                            + dot(a1, ia)
                            + dot(b1, ib)
                        )
                        Tiles[base + t["center_a"] + 3] = T.reinterpret(k11, "uint32")
                        inertia(da, oa, normal, ia)
                        inertia(db, ob, normal, ib)
                        kt = dot(normal, ia) + dot(normal, ib)
                        Tiles[base + h["diag_b"] + 3] = T.reinterpret(
                            T.if_then_else((kt < -1e-20) | (kt > 1e-20), 1.0 / kt, 0.0),
                            "uint32",
                        )
                        if count == 1:
                            Tiles[base + h["off_b"] + 3] = T.uint32(0)
                    else:
                        va = T.alloc_local((6,), "float32")
                        vb = T.alloc_local((6,), "float32")
                        va_base = (
                            T.cast(Tiles[base + h["diag_a"] + 3], "int32")
                            * l["velocity_words"]
                        )
                        vb_base = (
                            T.cast(Tiles[base + h["off_a"] + 3], "int32")
                            * l["velocity_words"]
                        )
                        for d in T.serial(3):
                            va[d] = f32(Velocities, va_base + l["velocity_linear"] + d)
                            va[d + 3] = f32(
                                Velocities, va_base + l["velocity_angular"] + d
                            )
                            vb[d] = f32(Velocities, vb_base + l["velocity_linear"] + d)
                            vb[d + 3] = f32(
                                Velocities, vb_base + l["velocity_angular"] + d
                            )
                        if operation == "warmstart":
                            force = T.alloc_local((3,), "float32")
                            torque_a = T.alloc_local((3,), "float32")
                            torque_b = T.alloc_local((3,), "float32")
                            impulse = T.alloc_local((3,), "float32")
                            for d in T.serial(3):
                                force[d] = 0.0
                                torque_a[d] = 0.0
                                torque_b[d] = 0.0
                            for k in T.serial(4):
                                if k < count:
                                    for d in T.serial(3):
                                        ra[d] = f32(
                                            Tiles,
                                            base
                                            + t["arm_a"]
                                            + k * l["twist_arm_words"]
                                            + d,
                                        )
                                        rb[d] = ra[d] + offset[d]
                                        impulse[d] = normal[d] * f32(
                                            Tiles, base + t["normal_impulse"] + k
                                        )
                                        force[d] += impulse[d]
                                    cross(ra, impulse, ta)
                                    cross(rb, impulse, tb)
                                    for d in T.serial(3):
                                        torque_a[d] += ta[d]
                                        torque_b[d] += tb[d]
                            for d in T.serial(3):
                                impulse[d] = tangent[d] * f32(
                                    Tiles, base + t["friction"]
                                ) + bitangent[d] * f32(Tiles, base + t["friction"] + 1)
                                force[d] += impulse[d]
                                rb[d] = center[d] + offset[d]
                            cross(center, impulse, ta)
                            cross(rb, impulse, tb)
                            for d in T.serial(3):
                                twist = normal[d] * f32(Tiles, base + h["off_b"] + 3)
                                torque_a[d] += ta[d] + twist
                                torque_b[d] += tb[d] + twist
                            inertia(da, oa, torque_a, ia)
                            inertia(db, ob, torque_b, ib)
                            for d in T.serial(3):
                                va[d] += im_a[d] * force[d]
                                va[d + 3] += ia[d]
                                vb[d] -= im_b[d] * force[d]
                                vb[d + 3] -= ib[d]
                        if operation == "solve":
                            pa = (
                                T.cast(Tiles[base + h["body_a"] + 3], "int32")
                                * l["pose_words"]
                            )
                            pb = (
                                T.cast(Tiles[base + h["body_b"] + 3], "int32")
                                * l["pose_words"]
                            )
                            temp = T.alloc_local((3,), "float32")
                            na = T.alloc_local((3,), "float32")
                            nb = T.alloc_local((3,), "float32")
                            delta_n = T.alloc_local((3,), "float32")
                            translation = T.alloc_local((3,), "float32")
                            rotated(Poses, pa, l["pose_rotation"], normal, temp, True)
                            rotated(Tiles, base, t["frame_a"], temp, na, False)
                            rotated(Poses, pb, l["pose_rotation"], normal, temp, True)
                            rotated(Tiles, base, t["frame_b"], temp, nb, False)
                            for d in T.serial(3):
                                delta_n[d] = na[d] - nb[d]
                                translation[d] = f32(
                                    Poses, pa + l["pose_translation"] + d
                                ) - f32(Poses, pb + l["pose_translation"] + d)
                            separation = dot(translation, normal) - dot(offset, nb)
                            is_static = (
                                (im_a[0] == 0.0) & (im_a[1] == 0.0) & (im_a[2] == 0.0)
                            ) | ((im_b[0] == 0.0) & (im_b[1] == 0.0) & (im_b[2] == 0.0))
                            cfm = T.if_then_else(is_static, Params[4], Params[2])
                            erp = T.if_then_else(is_static, Params[5], Params[3])
                            for k in T.serial(4):
                                if k < count:
                                    arm = base + t["arm_a"] + k * l["twist_arm_words"]
                                    for d in T.serial(3):
                                        ra[d] = f32(Tiles, arm + d)
                                        rb[d] = ra[d] + offset[d]
                                    dist = f32(Tiles, base + t["dist"] + k) + (
                                        separation + dot(ra, delta_n)
                                    )
                                    normal_vel = T.alloc_local((1,), "float32")
                                    normal_vel[0] = f32(
                                        Tiles, base + t["normal_vel"] + k
                                    )
                                    rhs = T.alloc_var(
                                        "float32",
                                        init=normal_vel[0]
                                        + T.max(dist, 0.0) * Params[0],
                                    )
                                    softness = T.alloc_var("float32", init=1.0)
                                    if c["use_bias"]:
                                        rhs += T.min(T.max(dist * erp, -Params[1]), 0.0)
                                        if dist <= 0.0:
                                            softness = cfm
                                    cross(ra, normal, ta)
                                    cross(rb, normal, tb)
                                    for d in T.serial(3):
                                        tb[d] = -tb[d]
                                    dvel = (
                                        relative_velocity(normal, ta, tb, va, vb) + rhs
                                    )
                                    previous = f32(
                                        Tiles, base + t["normal_impulse"] + k
                                    )
                                    impulse = softness * T.max(
                                        previous - f32(Tiles, arm + 3) * dvel, 0.0
                                    )
                                    delta = impulse - previous
                                    Tiles[base + t["normal_impulse"] + k] = (
                                        T.reinterpret(impulse, "uint32")
                                    )
                                    inertia(da, oa, ta, ia)
                                    inertia(db, ob, tb, ib)
                                    for d in T.serial(3):
                                        va[d] += normal[d] * im_a[d] * delta
                                        va[d + 3] += ia[d] * delta
                                        vb[d] += normal[d] * im_b[d] * -delta
                                        vb[d + 3] += ib[d] * delta
                            if c["solve_friction"]:
                                friction_rhs = T.alloc_local((2,), "float32")
                                friction_rhs[0] = 0.0
                                friction_rhs[1] = 0.0
                                if c["use_bias"]:
                                    ca = T.alloc_local((3,), "float32")
                                    cb = T.alloc_local((3,), "float32")
                                    drift = T.alloc_local((3,), "float32")
                                    rotated(
                                        Tiles, base, t["frame_a"], center, temp, True
                                    )
                                    rotated(
                                        Poses, pa, l["pose_rotation"], temp, ca, False
                                    )
                                    for d in T.serial(3):
                                        rb[d] = center[d] + offset[d]
                                    rotated(Tiles, base, t["frame_b"], rb, temp, True)
                                    rotated(
                                        Poses, pb, l["pose_rotation"], temp, cb, False
                                    )
                                    for d in T.serial(3):
                                        drift[d] = (
                                            (
                                                ca[d]
                                                + f32(
                                                    Poses,
                                                    pa + l["pose_translation"] + d,
                                                )
                                            )
                                            - (
                                                cb[d]
                                                + f32(
                                                    Poses,
                                                    pb + l["pose_translation"] + d,
                                                )
                                            )
                                        ) * Params[0]
                                    friction_rhs[0] = dot(drift, tangent)
                                    friction_rhs[1] = dot(drift, bitangent)
                                limit = f32(Tiles, base + h["tangent"] + 3)
                                tangent_limit = T.alloc_local((1,), "float32")
                                tangent_limit[0] = 0.0
                                twist_limit = T.alloc_local((1,), "float32")
                                twist_limit[0] = 0.0
                                for k in T.serial(4):
                                    if k < count:
                                        impulse = f32(
                                            Tiles, base + t["normal_impulse"] + k
                                        )
                                        tangent_limit[0] += impulse
                                        twist_limit[0] += impulse * f32(
                                            Tiles, base + t["radius"] + k
                                        )
                                tangent_limit[0] *= limit
                                twist_limit[0] *= limit
                                twist = T.alloc_local((1,), "float32")
                                twist[0] = 0.0
                                if count > 1:
                                    for d in T.serial(3):
                                        temp[d] = va[d + 3] - vb[d + 3]
                                    old_twist = f32(Tiles, base + h["off_b"] + 3)
                                    twist[0] = T.min(
                                        T.max(
                                            old_twist
                                            - f32(Tiles, base + h["diag_b"] + 3)
                                            * dot(normal, temp),
                                            -twist_limit[0],
                                        ),
                                        twist_limit[0],
                                    )
                                    delta = twist[0] - old_twist
                                    inertia(da, oa, normal, ia)
                                    inertia(db, ob, normal, ib)
                                    for d in T.serial(3):
                                        va[d + 3] += ia[d] * delta
                                        vb[d + 3] -= ib[d] * delta
                                a0 = T.alloc_local((3,), "float32")
                                a1 = T.alloc_local((3,), "float32")
                                b0 = T.alloc_local((3,), "float32")
                                b1 = T.alloc_local((3,), "float32")
                                for d in T.serial(3):
                                    rb[d] = center[d] + offset[d]
                                cross(center, tangent, a0)
                                cross(center, bitangent, a1)
                                cross(rb, tangent, b0)
                                cross(rb, bitangent, b1)
                                for d in T.serial(3):
                                    b0[d] = -b0[d]
                                    b1[d] = -b1[d]
                                d0 = (
                                    relative_velocity(tangent, a0, b0, va, vb)
                                    + friction_rhs[0]
                                )
                                d1 = (
                                    relative_velocity(bitangent, a1, b1, va, vb)
                                    + friction_rhs[1]
                                )
                                k00 = f32(Tiles, base + t["offset_b"] + 3)
                                k11 = f32(Tiles, base + t["center_a"] + 3)
                                k01 = f32(Tiles, base + t["friction"] + 2) * 0.5
                                determinant = k00 * k11 - k01 * k01
                                inv_det = T.if_then_else(
                                    (determinant < -1e-20) | (determinant > 1e-20),
                                    1.0 / determinant,
                                    0.0,
                                )
                                tangent_new = T.alloc_local((2,), "float32")
                                old0 = f32(Tiles, base + t["friction"])
                                old1 = f32(Tiles, base + t["friction"] + 1)
                                tangent_new[0] = old0 - (k11 * d0 - k01 * d1) * inv_det
                                tangent_new[1] = old1 - (k00 * d1 - k01 * d0) * inv_det
                                length = T.sqrt(
                                    tangent_new[0] * tangent_new[0]
                                    + tangent_new[1] * tangent_new[1]
                                )
                                if length > tangent_limit[0]:
                                    for d in T.serial(2):
                                        tangent_new[d] *= tangent_limit[0] / length
                                delta0 = tangent_new[0] - old0
                                delta1 = tangent_new[1] - old1
                                for d in T.serial(3):
                                    temp[d] = (
                                        tangent[d] * delta0 + bitangent[d] * delta1
                                    )
                                    ta[d] = a0[d] * delta0 + a1[d] * delta1
                                    tb[d] = b0[d] * delta0 + b1[d] * delta1
                                inertia(da, oa, ta, ia)
                                inertia(db, ob, tb, ib)
                                for d in T.serial(3):
                                    va[d] += im_a[d] * temp[d]
                                    va[d + 3] += ia[d]
                                    vb[d] -= im_b[d] * temp[d]
                                    vb[d + 3] += ib[d]
                                for d in T.serial(2):
                                    Tiles[base + t["friction"] + d] = T.reinterpret(
                                        tangent_new[d], "uint32"
                                    )
                                Tiles[base + h["off_b"] + 3] = T.reinterpret(
                                    twist[0], "uint32"
                                )
                        wa = T.alloc_local((1,), "uint32")
                        wb = T.alloc_local((1,), "uint32")
                        writable_a(Tiles, base, wa)
                        writable_b(Tiles, base, wb)
                        for d in T.serial(3):
                            if wa[0] != 0:
                                Velocities[va_base + l["velocity_linear"] + d] = (
                                    T.reinterpret(va[d], "uint32")
                                )
                                Velocities[va_base + l["velocity_angular"] + d] = (
                                    T.reinterpret(va[d + 3], "uint32")
                                )
                            if wb[0] != 0:
                                Velocities[vb_base + l["velocity_linear"] + d] = (
                                    T.reinterpret(vb[d], "uint32")
                                )
                                Velocities[vb_base + l["velocity_angular"] + d] = (
                                    T.reinterpret(vb[d + 3], "uint32")
                                )

    return one


def init_points_with_poses(layout, case):
    from kernel_support import cross

    h, t, l, c, m, mp = (
        layout[k]
        for k in ("header", "twist", "link", "contact", "manifold", "manifold_point")
    )

    @T.macro
    def one(Links, Contacts, SolverPoses, Velocities, States, Tiles, i, j):
        base = i // 32 * layout["tile_words"] + i % 32 * 4
        lb = i * layout["link_words"]
        cb = j * layout["contact_words"]
        mb = cb + c["contact"]
        sp = (case["new_base"] + j) * layout["state_words"] + layout["state_pose_a"]
        pa = T.cast(Links[lb + l["solver_body_a"]], "int32") * layout["pose_words"]
        pb = T.cast(Links[lb + l["solver_body_b"]], "int32") * layout["pose_words"]
        va = T.cast(Contacts[cb + c["bodies"]], "int32") * layout["velocity_words"]
        vb = T.cast(Contacts[cb + c["bodies"] + 1], "int32") * layout["velocity_words"]
        q = T.alloc_local((4,), "float32")
        normal = T.alloc_local((3,), "float32")
        direction = T.alloc_local((3,), "float32")
        offset = T.alloc_local((3,), "float32")
        for d in T.serial(4):
            q[d] = f32(States, sp + layout["pose_rotation"] + d)
            Tiles[base + t["frame_a"] + d] = SolverPoses[
                pa + layout["pose_rotation"] + d
            ]
            Tiles[base + t["frame_b"] + d] = SolverPoses[
                pb + layout["pose_rotation"] + d
            ]
        for d in T.serial(3):
            normal[d] = f32(Contacts, mb + m["normal_a"] + d)
            offset[d] = f32(SolverPoses, pa + layout["pose_translation"] + d) - f32(
                SolverPoses, pb + layout["pose_translation"] + d
            )
            Tiles[base + t["offset_b"] + d] = T.reinterpret(offset[d], "uint32")
        quat_rotate(q, normal, direction)
        for d in T.serial(3):
            direction[d] = -direction[d]
            Tiles[base + h["normal"] + d] = T.reinterpret(direction[d], "uint32")
        sign = T.if_then_else(direction[2] < 0.0, -1.0, 1.0)
        aa = -1.0 / (sign + direction[2])
        Tiles[base + h["tangent"]] = T.reinterpret(
            direction[0] * direction[1] * aa, "uint32"
        )
        Tiles[base + h["tangent"] + 1] = T.reinterpret(
            sign + direction[1] * direction[1] * aa, "uint32"
        )
        Tiles[base + h["tangent"] + 2] = T.reinterpret(-direction[1], "uint32")
        for k in T.serial(4):
            if k < T.cast(Contacts[mb + m["len"]], "int32"):
                src = mb + m["points_a"] + k * layout["manifold_point_words"]
                arm = base + t["arm_a"] + k * layout["twist_arm_words"]
                point = T.alloc_local((3,), "float32")
                world = T.alloc_local((3,), "float32")
                ra = T.alloc_local((3,), "float32")
                rb = T.alloc_local((3,), "float32")
                wa = T.alloc_local((3,), "float32")
                wb = T.alloc_local((3,), "float32")
                ca = T.alloc_local((3,), "float32")
                cc = T.alloc_local((3,), "float32")
                dv = T.alloc_local((3,), "float32")
                for d in T.serial(3):
                    point[d] = (
                        f32(Contacts, src + mp["pt"] + d)
                        + normal[d] * f32(Contacts, src + mp["dist"]) / 2.0
                    )
                    wa[d] = f32(Velocities, va + layout["velocity_angular"] + d)
                    wb[d] = f32(Velocities, vb + layout["velocity_angular"] + d)
                quat_rotate(q, point, world)
                for d in T.serial(3):
                    ra[d] = (
                        world[d] + f32(States, sp + layout["pose_translation"] + d)
                    ) - f32(SolverPoses, pa + layout["pose_translation"] + d)
                    rb[d] = ra[d] + offset[d]
                    Tiles[arm + d] = T.reinterpret(ra[d], "uint32")
                cross(wa, ra, ca)
                cross(wb, rb, cc)
                for d in T.serial(3):
                    dv[d] = (
                        (f32(Velocities, va + layout["velocity_linear"] + d) + ca[d])
                        - f32(Velocities, vb + layout["velocity_linear"] + d)
                    ) - cc[d]
                Tiles[base + t["normal_vel"] + k] = T.reinterpret(
                    f32(Links, lb + l["restitution"]) * dot(dv, direction), "uint32"
                )
                Tiles[base + t["dist"] + k] = Contacts[src + mp["dist"]]

    return one


def recycle_from(layout):
    from kernel_support import cross

    h, t, l = (layout[k] for k in ("header", "twist", "link"))

    @T.macro
    def one(Links, Velocities, OldTiles, Tiles, i, old):
        base = i // 32 * layout["tile_words"] + i % 32 * 4
        ob = old // 32 * layout["tile_words"] + old % 32 * 4
        lb = i * layout["link_words"]
        va = T.cast(Links[lb + l["solver_body_a"]], "int32") * layout["velocity_words"]
        vb = T.cast(Links[lb + l["solver_body_b"]], "int32") * layout["velocity_words"]
        for d in T.serial(4):
            Tiles[base + h["normal"] + d] = OldTiles[ob + h["normal"] + d]
            Tiles[base + t["frame_a"] + d] = OldTiles[ob + t["frame_a"] + d]
            Tiles[base + t["frame_b"] + d] = OldTiles[ob + t["frame_b"] + d]
            Tiles[base + t["offset_b"] + d] = OldTiles[ob + t["offset_b"] + d]
            Tiles[base + t["center_a"] + d] = OldTiles[ob + t["center_a"] + d]
        for d in T.serial(3):
            Tiles[base + h["tangent"] + d] = OldTiles[ob + h["tangent"] + d]
            Tiles[base + t["friction"] + d] = OldTiles[ob + t["friction"] + d]
        Tiles[base + h["diag_b"] + 3] = OldTiles[ob + h["diag_b"] + 3]
        Tiles[base + h["off_b"] + 3] = OldTiles[ob + h["off_b"] + 3]
        for k in T.serial(4):
            if k < T.cast(OldTiles[ob + h["normal"] + 3], "int32"):
                arm = base + t["arm_a"] + k * layout["twist_arm_words"]
                src = ob + t["arm_a"] + k * layout["twist_arm_words"]
                for d in T.serial(4):
                    Tiles[arm + d] = OldTiles[src + d]
                Tiles[base + t["normal_impulse"] + k] = OldTiles[
                    ob + t["normal_impulse"] + k
                ]
                Tiles[base + t["dist"] + k] = OldTiles[ob + t["dist"] + k]
                Tiles[base + t["radius"] + k] = OldTiles[ob + t["radius"] + k]
                ra = T.alloc_local((3,), "float32")
                rb = T.alloc_local((3,), "float32")
                wa = T.alloc_local((3,), "float32")
                wb = T.alloc_local((3,), "float32")
                ca = T.alloc_local((3,), "float32")
                cb = T.alloc_local((3,), "float32")
                dv = T.alloc_local((3,), "float32")
                normal = T.alloc_local((3,), "float32")
                for d in T.serial(3):
                    ra[d] = f32(OldTiles, src + d)
                    rb[d] = ra[d] + f32(OldTiles, ob + t["offset_b"] + d)
                    wa[d] = f32(Velocities, va + layout["velocity_angular"] + d)
                    wb[d] = f32(Velocities, vb + layout["velocity_angular"] + d)
                    normal[d] = f32(OldTiles, ob + h["normal"] + d)
                cross(wa, ra, ca)
                cross(wb, rb, cb)
                for d in T.serial(3):
                    dv[d] = (
                        (f32(Velocities, va + layout["velocity_linear"] + d) + ca[d])
                        - f32(Velocities, vb + layout["velocity_linear"] + d)
                    ) - cb[d]
                Tiles[base + t["normal_vel"] + k] = T.reinterpret(
                    f32(Links, lb + l["restitution"]) * dot(dv, normal), "uint32"
                )

    return one
