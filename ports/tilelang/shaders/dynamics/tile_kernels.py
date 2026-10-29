"""Counterparts of dynamics/tile_kernels.rs entry points for prepared colors.

The replay supplies a single contiguous, disjoint color. Production bucket,
shared-static and gather schedules are deliberately not claimed by these adapters.
"""

from kernel_support import launch
from .solver_utils import solve_points, warmstart_constraint
from .coulomb_tiles import validate_layout


def gpu_solve_constraints(layout, case, target):
    validate_layout(layout)
    return launch(solve_points(layout, case), layout, case, target, case["count"])


def gpu_warmstart_constraints(layout, case, target):
    validate_layout(layout)
    return launch(
        warmstart_constraint(layout, case), layout, case, target, case["count"]
    )


def transfer_contact_impulses(layout):
    """Nearest old local-anchor pair within 10cm; strict ties retain the first."""
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
    import tilelang.language as T
    from .solver_utils import (
        contact_to_constraint_header,
        contact_to_constraint_points,
        recycle_from,
    )
    from .constraint import compute_effective_masses
    from .coulomb_tiles import constraint_warmstart

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
