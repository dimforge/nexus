"""Counterparts of dynamics/tile_kernels.rs entry points for prepared colors.

Direct solve/warmstart adapters still supply one disjoint color. Cache scaling,
ordered body gathering and split-slot warmstart use runtime native graph buffers;
production color schedules and complete frame stepping remain separate work.
"""

import tilelang.language as T
from kernel_support import launch

from .coulomb_tiles import validate_layout
from .solver_utils import solve_points, warmstart_constraint


def gpu_solve_constraints(layout, case, target):
    validate_layout(layout)
    if case.get("native_color_range"):
        from colored_kernels import make_kernel

        return make_kernel(layout, case, target, "colored_solve")
    return launch(solve_points(layout, case), layout, case, target, case["count"])


def gpu_warmstart_constraints(layout, case, target):
    validate_layout(layout)
    if case.get("native_color_range"):
        from colored_kernels import make_kernel

        return make_kernel(layout, case, target, "colored_warmstart")
    return launch(
        warmstart_constraint(layout, case), layout, case, target, case["count"]
    )


def transfer_contact_impulses(layout):
    """Nearest old local-anchor pair within 10cm; strict ties retain the first."""
    if layout.get("friction_model") == "twist":
        return _transfer_twist_contact_impulses(layout)

    import tilelang.language as T
    from kernel_support import cross, dot, f32

    h, p = layout["header"], layout["point"]

    @T.macro
    def one(OldTiles, Tiles, i, old):
        base = i // 32 * layout["tile_words"] + i % 32 * 4
        ob = old // 32 * layout["tile_words"] + old % 32 * 4
        direction = T.alloc_local((3,), "float32")
        tangent = T.alloc_local((3,), "float32")
        bitangent = T.alloc_local((3,), "float32")
        od = T.alloc_local((3,), "float32")
        ot = T.alloc_local((3,), "float32")
        obt = T.alloc_local((3,), "float32")
        for d in T.serial(3):
            direction[d] = f32(Tiles, base + h["normal"] + d)
            tangent[d] = f32(Tiles, base + h["tangent"] + d)
            od[d] = f32(OldTiles, ob + h["normal"] + d)
            ot[d] = f32(OldTiles, ob + h["tangent"] + d)
        cross(direction, tangent, bitangent)
        cross(od, ot, obt)
        for k in T.serial(4):
            if k < Tiles[base + h["normal"] + 3]:
                dst = base + layout["points"] + k * layout["point_words"]
                best = T.alloc_local((1,), "int32")
                sq_best = T.alloc_local((1,), "float32")
                best[0] = -1
                sq_best[0] = T.float32(0.1) * T.float32(0.1)
                for j in T.serial(4):
                    if j < OldTiles[ob + h["normal"] + 3]:
                        src = ob + layout["points"] + j * layout["point_words"]
                        da = T.alloc_local((3,), "float32")
                        db = T.alloc_local((3,), "float32")
                        for d in T.serial(3):
                            da[d] = f32(OldTiles, src + p["local_a"] + d) - f32(
                                Tiles, dst + p["local_a"] + d
                            )
                            db[d] = f32(OldTiles, src + p["local_b"] + d) - f32(
                                Tiles, dst + p["local_b"] + d
                            )
                        sq = T.max(dot(da, da), dot(db, db))
                        if sq < sq_best[0]:
                            best[0] = j
                            sq_best[0] = sq
                if best[0] >= 0:
                    src = ob + layout["points"] + best[0] * layout["point_words"]
                    Tiles[dst + p["arm_a"] + 3] = OldTiles[src + p["arm_a"] + 3]
                    world = T.alloc_local((3,), "float32")
                    for d in T.serial(3):
                        world[d] = ot[d] * f32(OldTiles, src + p["friction"]) + obt[
                            d
                        ] * f32(OldTiles, src + p["friction"] + 1)
                    Tiles[dst + p["friction"]] = T.reinterpret(
                        dot(world, tangent), "uint32"
                    )
                    Tiles[dst + p["friction"] + 1] = T.reinterpret(
                        dot(world, bitangent), "uint32"
                    )

    return one


def gpu_prepare_constraints(layout, case):
    """Prepare independent output tile lanes from color-ordered Rust contact links."""
    if layout.get("friction_model") == "twist":
        from .twist_tiles import gpu_prepare_constraints as prepare_twist

        return prepare_twist(layout, case)

    import tilelang.language as T

    from .constraint import compute_effective_masses
    from .coulomb_tiles import constraint_warmstart
    from .solver_utils import (
        contact_to_constraint_header,
        contact_to_constraint_points,
        recycle_from,
    )

    # Eliminate unreachable gather/matching loops before TileLang index analysis.
    if case["count"] == 0:

        @T.macro
        def empty(
            Links,
            Contacts,
            WorldMass,
            SolverPoses,
            Velocities,
            States,
            OldTiles,
            Tiles,
            i,
        ):
            T.evaluate(0)

        return empty

    header = contact_to_constraint_header(layout)
    points = contact_to_constraint_points(layout, case)
    recycle = recycle_from(layout)
    transfer = transfer_contact_impulses(layout)
    masses = compute_effective_masses(layout, case)
    cache = constraint_warmstart(layout)
    l = layout["link"]

    @T.macro
    def one(
        Links, Contacts, WorldMass, SolverPoses, Velocities, States, OldTiles, Tiles, i
    ):
        if i < case["count"]:
            lb = i * layout["link_words"]
            if Links[lb + l["len"]] != 0:
                j = T.cast(Links[lb + l["contact"]], "int32")
                base = i // 32 * layout["tile_words"] + i % 32 * 4
                for k in T.serial(4):
                    Tiles[base + layout["coupling"] + k] = T.uint32(0)
                header(Links, Contacts, WorldMass, Tiles, i, j)
                if Links[lb + l["recycled"]] == 0:
                    points(
                        Links, Contacts, SolverPoses, Velocities, States, Tiles, i, j
                    )
                if Links[lb + l["previous_constraint"]] != T.uint32(0xFFFFFFFF):
                    old = T.cast(Links[lb + l["previous_constraint_index"]], "int32")
                    if Links[lb + l["recycled"]] != 0:
                        recycle(Links, Velocities, OldTiles, Tiles, i, old)
                    else:
                        transfer(OldTiles, Tiles, i, old)
                masses(Tiles, Velocities, SolverPoses, Contacts, i)
                cache(Tiles, i)

    return one


def gpu_scale_constraint_impulses(layout):
    """Scale active points and refresh the cache, including write_constraint resets."""
    import tilelang.language as T
    from kernel_support import f32

    from .coulomb_tiles import constraint_warmstart
    from .solver_utils import scale_impulses

    scale = scale_impulses(layout)
    cache = constraint_warmstart(layout)
    h, b = layout["header"], layout["batch"]

    @T.macro
    def one(Tiles, Buckets, BatchIds, Params, i):
        end = T.cast(BatchIds[b["num_batches"]], "int32") * T.cast(
            BatchIds[b["solver_color_buckets_stride"]], "int32"
        )
        if i < T.cast(Buckets[end - 1], "int32"):
            scale(Tiles, i, f32(Params, layout["warmstart_coefficient"]))
            base = i // 32 * layout["tile_words"] + i % 32 * 4
            # Rust read_constraint/write_constraint preserves active fields, zeros
            # inactive coupling and the Coulomb header's two model-specific words.
            Tiles[base + h["diag_b"] + 3] = T.uint32(0)
            Tiles[base + h["off_b"] + 3] = T.uint32(0)
            for k in T.serial(4):
                if k >= Tiles[base + h["normal"] + 3]:
                    Tiles[base + layout["coupling"] + k] = T.uint32(0)
            cache(Tiles, i)

    return one


def gpu_gather_warmstart_velocities(layout):
    """Each ordinary body owns its output; accumulate in Rust adjacency-list order."""
    import tilelang.language as T
    from kernel_support import f32

    b = layout["batch"]

    @T.macro
    def one(Counts, Ids, Indices, Tiles, Velocities, HubFirstSlot, BatchIds, body):
        count = T.cast(BatchIds[b["bodies_len"]], "int32") * T.cast(
            BatchIds[b["num_batches"]], "int32"
        )
        if body < count:
            if HubFirstSlot[body] == T.uint32(0xFFFFFFFF):
                # A local buffer avoids invalid Metal code for a mutable scalar
                # assigned from a bounds-guarded load. Body zero must not read -1.
                start = T.alloc_local((1,), "int32")
                start[0] = 0
                if body > 0:
                    start[0] = T.cast(Counts[body - 1], "int32")
                end = T.cast(Counts[body], "int32")
                velocity = T.alloc_local((6,), "float32")
                for d in T.serial(3):
                    velocity[d] = f32(
                        Velocities,
                        body * layout["velocity_words"] + layout["velocity_linear"] + d,
                    )
                    velocity[d + 3] = f32(
                        Velocities,
                        body * layout["velocity_words"]
                        + layout["velocity_angular"]
                        + d,
                    )
                for entry in T.serial(start[0], end):
                    index = T.cast(Indices[T.cast(Ids[entry], "int32")], "int32")
                    pair = (
                        index // 32 * layout["tile_words"]
                        + layout["warmstart"]
                        + index % 32 * 2 * layout["velocity_words"]
                    )
                    side = T.if_then_else(
                        Tiles[pair + layout["velocity_padding1"]]
                        == T.cast(body, "uint32"),
                        0,
                        1,
                    )
                    source = pair + side * layout["velocity_words"]
                    for d in T.serial(3):
                        velocity[d] = velocity[d] + f32(
                            Tiles, source + layout["velocity_linear"] + d
                        )
                        velocity[d + 3] = velocity[d + 3] + f32(
                            Tiles, source + layout["velocity_angular"] + d
                        )
                for d in T.serial(3):
                    Velocities[
                        body * layout["velocity_words"] + layout["velocity_linear"] + d
                    ] = T.reinterpret(velocity[d], "uint32")
                    Velocities[
                        body * layout["velocity_words"] + layout["velocity_angular"] + d
                    ] = T.reinterpret(velocity[d + 3], "uint32")

    return one


def gpu_hub_warmstart_constraints(layout):
    """Recompute each split slot's delta from its constraint, not the cached pair."""
    import tilelang.language as T
    from kernel_support import f32

    from .coulomb_tiles import tile_warmstart

    delta = tile_warmstart(layout)
    h, hc = layout["header"], layout["hub"]

    @T.macro
    def one(Velocities, Tiles, HubSlotConstraint, HubCounts, Indices, slot):
        count = T.min(
            T.cast(HubCounts[hc["slots"]], "int32"),
            T.cast(HubCounts[hc["pool"]], "int32"),
        )
        if slot < count:
            vel_slot = T.cast(HubCounts[hc["base"]], "int32") + slot
            # Keep the graph indirection out of every expanded point expression;
            # otherwise CPU interval simplification grows prohibitively expensive.
            index = T.alloc_local((1,), "int32")
            index[0] = T.cast(
                Indices[T.cast(HubSlotConstraint[slot], "int32")], "int32"
            )
            a = T.alloc_local((6,), "float32")
            b = T.alloc_local((6,), "float32")
            delta(Tiles, index[0], a, b)
            base = index[0] // 32 * layout["tile_words"] + index[0] % 32 * 4
            use_a = Tiles[base + h["diag_a"] + 3] == T.cast(vel_slot, "uint32")
            for d in T.serial(3):
                linear = f32(
                    Velocities,
                    vel_slot * layout["velocity_words"] + layout["velocity_linear"] + d,
                ) + T.if_then_else(use_a, a[d], b[d])
                angular = f32(
                    Velocities,
                    vel_slot * layout["velocity_words"]
                    + layout["velocity_angular"]
                    + d,
                ) + T.if_then_else(use_a, a[d + 3], b[d + 3])
                Velocities[
                    vel_slot * layout["velocity_words"] + layout["velocity_linear"] + d
                ] = T.reinterpret(linear, "uint32")
                Velocities[
                    vel_slot * layout["velocity_words"] + layout["velocity_angular"] + d
                ] = T.reinterpret(angular, "uint32")

    return one


def gpu_solve_constraints_unbiased_cached(layout, case, target):
    validate_layout(layout)
    from colored_kernels import make_kernel

    return make_kernel(layout, case, target, "colored_cached")


def _colored_constraint_body(layout, case, operation):
    """Native color_constraints range, runtime bias mode, and cached writeback."""
    from .coulomb_tiles import constraint_warmstart
    from .sim_params import contact_coefficients

    if case["count"] == 0:

        @T.macro
        def empty(
            Tiles, Velocities, Poses, RbdParams, Buckets, BatchIds, Color, Mode, gid
        ):
            pass

        return empty
    solve_factory = solve_points
    warmstart_factory = warmstart_constraint
    if layout.get("friction_model") == "twist":
        from .twist_friction import (
            solve as solve_factory,
        )
        from .twist_friction import (
            warmstart as warmstart_factory,
        )
        from .twist_tiles import constraint_warmstart

    variants = [
        solve_factory(layout, dict(case, use_bias=bias, solve_friction=friction))
        for bias, friction in [(False, True), (True, False), (True, True)]
    ]
    unbiased, biased, both = variants
    warmstart = warmstart_factory(layout, case)
    cache = constraint_warmstart(layout)
    coefficients = contact_coefficients(layout)
    batch = layout["batch"]
    groups = case["groups"]

    @T.macro
    def one(Tiles, Velocities, Poses, RbdParams, Buckets, BatchIds, Color, Mode, gid):
        Params = T.alloc_local((6,), "float32")
        coefficients(RbdParams, Params)
        nb = T.Cast("int32", BatchIds[batch["num_batches"]])
        color = T.Cast("int32", Color[0])
        start = T.Cast("int32", Buckets[color * nb - 1])
        end = T.Cast("int32", Buckets[(color + 1) * nb - 1])
        index = T.alloc_local((1,), "int32")
        for chunk in T.serial(T.ceildiv(end - start, groups * 64)):
            index[0] = start + gid + chunk * groups * 64
            if index[0] < end:
                if operation == "colored_warmstart":
                    warmstart(Tiles, Velocities, Poses, Params, index[0])
                elif operation == "colored_cached":
                    unbiased(Tiles, Velocities, Poses, Params, index[0])
                    cache(Tiles, index[0])
                else:
                    if Mode[0] == T.uint32(0):
                        unbiased(Tiles, Velocities, Poses, Params, index[0])
                    elif Mode[0] == T.uint32(1):
                        biased(Tiles, Velocities, Poses, Params, index[0])
                    else:
                        both(Tiles, Velocities, Poses, Params, index[0])

    return one


def _transfer_twist_contact_impulses(layout):
    """Generic nearest-anchor transfer specialized to native TwistConstraint fields."""
    import tilelang.language as T
    from kernel_support import cross, dot, f32, quat_rotate

    h, t = layout["header"], layout["twist"]

    @T.macro
    def anchors(Tiles, b, k, a, c):
        ra = T.alloc_local((3,), "float32")
        rb = T.alloc_local((3,), "float32")
        qa = T.alloc_local((4,), "float32")
        qb = T.alloc_local((4,), "float32")
        for d in T.serial(3):
            ra[d] = f32(Tiles, b + t["arm_a"] + k * layout["twist_arm_words"] + d)
            rb[d] = ra[d] + f32(Tiles, b + t["offset_b"] + d)
        for d in T.serial(4):
            qa[d] = f32(Tiles, b + t["frame_a"] + d) * T.if_then_else(d < 3, -1.0, 1.0)
            qb[d] = f32(Tiles, b + t["frame_b"] + d) * T.if_then_else(d < 3, -1.0, 1.0)
        quat_rotate(qa, ra, a)
        quat_rotate(qb, rb, c)

    @T.macro
    def one(OldTiles, Tiles, i, old):
        base = i // 32 * layout["tile_words"] + i % 32 * 4
        ob = old // 32 * layout["tile_words"] + old % 32 * 4
        tangent = T.alloc_local((3,), "float32")
        normal = T.alloc_local((3,), "float32")
        bitangent = T.alloc_local((3,), "float32")
        ot = T.alloc_local((3,), "float32")
        on = T.alloc_local((3,), "float32")
        obt = T.alloc_local((3,), "float32")
        world = T.alloc_local((3,), "float32")
        for d in T.serial(3):
            tangent[d] = f32(Tiles, base + h["tangent"] + d)
            normal[d] = f32(Tiles, base + h["normal"] + d)
            ot[d] = f32(OldTiles, ob + h["tangent"] + d)
            on[d] = f32(OldTiles, ob + h["normal"] + d)
        cross(normal, tangent, bitangent)
        cross(on, ot, obt)
        for d in T.serial(3):
            world[d] = ot[d] * f32(OldTiles, ob + t["friction"]) + obt[d] * f32(
                OldTiles, ob + t["friction"] + 1
            )
        count = T.cast(Tiles[base + h["normal"] + 3], "int32")
        for k in T.serial(4):
            if k < count:
                a = T.alloc_local((3,), "float32")
                b = T.alloc_local((3,), "float32")
                oa = T.alloc_local((3,), "float32")
                oc = T.alloc_local((3,), "float32")
                da = T.alloc_local((3,), "float32")
                db = T.alloc_local((3,), "float32")
                best = T.alloc_local((1,), "int32")
                sq_best = T.alloc_local((1,), "float32")
                best[0] = -1
                sq_best[0] = T.float32(0.1) * T.float32(0.1)
                anchors(Tiles, base, k, a, b)
                for j in T.serial(4):
                    if j < T.cast(OldTiles[ob + h["normal"] + 3], "int32"):
                        anchors(OldTiles, ob, j, oa, oc)
                        for d in T.serial(3):
                            da[d] = oa[d] - a[d]
                            db[d] = oc[d] - b[d]
                        sq = T.max(dot(da, da), dot(db, db))
                        if sq < sq_best[0]:
                            sq_best[0] = sq
                            best[0] = j
                if best[0] >= 0:
                    Tiles[base + t["normal_impulse"] + k] = OldTiles[
                        ob + t["normal_impulse"] + best[0]
                    ]
                    Tiles[base + t["friction"]] = T.reinterpret(
                        f32(Tiles, base + t["friction"]) + dot(world, tangent), "uint32"
                    )
                    Tiles[base + t["friction"] + 1] = T.reinterpret(
                        f32(Tiles, base + t["friction"] + 1) + dot(world, bitangent),
                        "uint32",
                    )
                    Tiles[base + h["off_b"] + 3] = T.reinterpret(
                        f32(Tiles, base + h["off_b"] + 3)
                        + f32(OldTiles, ob + h["off_b"] + 3),
                        "uint32",
                    )
        if count > 0:
            for d in T.serial(2):
                Tiles[base + t["friction"] + d] = T.reinterpret(
                    f32(Tiles, base + t["friction"] + d) / T.cast(count, "float32"),
                    "uint32",
                )
            if count > 1:
                Tiles[base + h["off_b"] + 3] = T.reinterpret(
                    f32(Tiles, base + h["off_b"] + 3) / T.cast(count, "float32"),
                    "uint32",
                )
            else:
                Tiles[base + h["off_b"] + 3] = T.uint32(0)

    return one
