"""Partial counterpart of dynamics/solver.rs; body lifecycle entry points."""

import tilelang.language as T
from kernel_support import dot, f32, launch
from .body import integrate_linearized


def _integrate_linearized_body(layout, n):
    integrate = integrate_linearized(layout)

    @T.macro
    def one(Velocities, Poses, Params, i):
        if i < n:
            vp = i * layout["velocity_words"]
            v = T.alloc_local((3,), "float32")
            w = T.alloc_local((3,), "float32")
            for d in T.serial(3):
                v[d] = f32(Velocities, vp + layout["velocity_linear"] + d)
                w[d] = f32(Velocities, vp + layout["velocity_angular"] + d)
            vl = T.sqrt(dot(v, v))
            wl = T.sqrt(dot(w, w))
            if vl > Params[1]:
                for d in T.serial(3):
                    v[d] = v[d] * (Params[1] / vl)
            if wl > Params[2]:
                for d in T.serial(3):
                    w[d] = w[d] * (Params[2] / wl)
            for d in T.serial(3):
                Velocities[vp + layout["velocity_linear"] + d] = T.reinterpret(
                    v[d], "uint32"
                )
                Velocities[vp + layout["velocity_angular"] + d] = T.reinterpret(
                    w[d], "uint32"
                )
            integrate(Poses, i, v, w, Params[0])

    return one


def gpu_integrate_linearized(layout, case, target):
    n = case["active_bodies"]
    integrate = _integrate_linearized_body(layout, n)

    @T.macro
    def one(Tiles, Velocities, Poses, Params, i):
        integrate(Velocities, Poses, Params, i)

    return launch(one, layout, case, target, n)


def _is_dynamic(layout, WorldMass, i):
    p = i * layout["world_words"] + layout["world_inv_mass"]
    # Rust tests the whole vector, not each axis independently.
    return (
        (f32(WorldMass, p) != 0.0)
        | (f32(WorldMass, p + 1) != 0.0)
        | (f32(WorldMass, p + 2) != 0.0)
    )


def gpu_solver_cleanup(layout, case):
    @T.macro
    def one(Counts, SolverVelocities, Velocities, WorldMass, FirstSlot, HubCounts, i):
        if i == 0:
            HubCounts[layout["hub_hubs"]] = T.uint32(0)
            HubCounts[layout["hub_slots"]] = T.uint32(0)
        # The launch covers every slot, specializing Rust's grid-stride loop.
        if i < case["slots"]:
            Counts[i] = T.uint32(0)
            FirstSlot[i] = T.uint32(0xFFFFFFFF)
            vp = i * layout["velocity_words"]
            for d in T.serial(3):
                SolverVelocities[vp + layout["velocity_linear"] + d] = T.if_then_else(
                    _is_dynamic(layout, WorldMass, i),
                    Velocities[vp + layout["velocity_linear"] + d],
                    T.uint32(0),
                )
                SolverVelocities[vp + layout["velocity_angular"] + d] = T.if_then_else(
                    _is_dynamic(layout, WorldMass, i),
                    Velocities[vp + layout["velocity_angular"] + d],
                    T.uint32(0),
                )

    return one


def gpu_init_solver_vels_inc(layout, case):
    @T.macro
    def one(Increments, WorldMass, Params, i):
        if i < case["count"]:
            vp = i * layout["velocity_words"]
            for d in T.serial(3):
                Increments[vp + layout["velocity_linear"] + d] = T.reinterpret(
                    T.if_then_else(
                        _is_dynamic(layout, WorldMass, i),
                        Params[3 + d] * Params[0],
                        T.float32(0),
                    ),
                    "uint32",
                )
                Increments[vp + layout["velocity_angular"] + d] = T.uint32(0)

    return one


def gpu_apply_solver_vels_inc(layout, case):
    @T.macro
    def one(SolverVelocities, Increments, i):
        if i < case["count"]:
            vp = i * layout["velocity_words"]
            for d in T.serial(3):
                lp = vp + layout["velocity_linear"] + d
                ap = vp + layout["velocity_angular"] + d
                SolverVelocities[lp] = T.reinterpret(
                    f32(SolverVelocities, lp) + f32(Increments, lp), "uint32"
                )
                SolverVelocities[ap] = T.reinterpret(
                    f32(SolverVelocities, ap) + f32(Increments, ap), "uint32"
                )

    return one


def _prepend_translation(layout, sign):
    """glamx 0.3.1 Pose3::prepend_translation, using glam's quaternion product.

    Unlike component-only writes, constructing a Pose resets its explicit padding.
    """

    @T.macro
    def one(Source, LocalMass, Destination, i):
        pp = i * layout["pose_words"]
        cp = i * layout["local_words"] + layout["local_com"]
        q = T.alloc_local((4,), "float32")
        c = T.alloc_local((3,), "float32")
        cross = T.alloc_local((3,), "float32")
        for d in T.serial(4):
            q[d] = f32(Source, pp + layout["pose_rotation"] + d)
            Destination[pp + layout["pose_rotation"] + d] = Source[
                pp + layout["pose_rotation"] + d
            ]
        for d in T.serial(3):
            c[d] = f32(LocalMass, cp + d) * sign
        cross[0] = q[1] * c[2] - q[2] * c[1]
        cross[1] = q[2] * c[0] - q[0] * c[2]
        cross[2] = q[0] * c[1] - q[1] * c[0]
        scale = q[3] * q[3] - dot(q, q)
        projection = dot(c, q) * 2.0
        for d in T.serial(3):
            rotated = (c[d] * scale + q[d] * projection) + cross[d] * (q[3] * 2.0)
            Destination[pp + layout["pose_translation"] + d] = T.reinterpret(
                f32(Source, pp + layout["pose_translation"] + d) + rotated, "uint32"
            )
        Destination[pp + layout["pose_padding"]] = T.uint32(0)

    return one


def gpu_init_solver_bodies(layout, case):
    prepend = _prepend_translation(layout, 1.0)

    @T.macro
    def one(Poses, LocalMass, SolverPoses, i):
        if i < case["count"]:
            prepend(Poses, LocalMass, SolverPoses, i)

    return one


def gpu_solver_finalize(layout, case):
    prepend = _prepend_translation(layout, -1.0)

    @T.macro
    def one(Velocities, SolverVelocities, Poses, SolverPoses, LocalMass, i):
        if i < case["count"]:
            vp = i * layout["velocity_words"]
            for d in T.serial(3):
                Velocities[vp + layout["velocity_linear"] + d] = SolverVelocities[
                    vp + layout["velocity_linear"] + d
                ]
                Velocities[vp + layout["velocity_angular"] + d] = SolverVelocities[
                    vp + layout["velocity_angular"] + d
                ]
            prepend(SolverPoses, LocalMass, Poses, i)

    return one


def _adjacency(layout, scatter):
    """Shared filtering of the two Rust graph construction entry points."""
    cw, wm = layout["contact_words"], layout["world_mass_words"]
    body = layout["contact"]["bodies"]
    length = layout["contact"]["contact"] + layout["manifold"]["len"]
    mass = layout["world_mass"]["inv_mass"]

    @T.macro
    def one(Contacts, Counts, Ids, WorldMass, IsMultibody, Plan, i, groups):
        total = T.Cast("int32", Plan[layout["plan"]["bound"]])
        for chunk in T.serial(T.ceildiv(total, groups * 64)):
            contact = chunk * groups * 64 + i
            if contact < total:
                # Gaps may contain stale, out-of-range body IDs. Filter length
                # before even reading body-owned state, matching the Rust shader.
                if Contacts[contact * cw + length] != 0:
                    a = T.Cast("int32", Contacts[contact * cw + body])
                    b = T.Cast("int32", Contacts[contact * cw + body + 1])
                    if (IsMultibody[a] == 0) & (IsMultibody[b] == 0):
                        for side in T.unroll(2, explicit=True):
                            owner = T.if_then_else(side == 0, a, b)
                            active = (
                                (f32(WorldMass, owner * wm + mass) != 0.0)
                                | (f32(WorldMass, owner * wm + mass + 1) != 0.0)
                                | (f32(WorldMass, owner * wm + mass + 2) != 0.0)
                            )
                            if active:
                                if scatter:
                                    slot = T.Cast(
                                        "int32",
                                        T.atomic_add(
                                            Counts[owner], T.uint32(1), return_prev=True
                                        ),
                                    )
                                    Ids[slot] = T.Cast("uint32", contact)
                                else:
                                    T.atomic_add(Counts[owner], T.uint32(1))

    return one


def gpu_solver_count_constraints(layout):
    return _adjacency(layout, False)


def gpu_solver_sort_constraints(layout):
    return _adjacency(layout, True)
