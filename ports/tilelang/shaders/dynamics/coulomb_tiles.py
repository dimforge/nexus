"""ABI counterpart of dynamics/coulomb_tiles.rs::CoulombTile.

Offsets come from the live Rust exporter; scalar physics accesses this native
layout directly. This module does not introduce an independent packed layout.
"""


def validate_layout(layout):
    if (
        layout["version"] != 1
        or layout["tile_len"] != 32
        or layout["max_points"] != 4
        or layout["vec_scalar_words"] != 4
        or layout["friction_words"] != 4
    ):
        raise ValueError("Unsupported Rust replay ABI")


def constraint_warmstart(layout):
    """Cache the velocity deltas, tagged with body IDs, not split velocity slots."""
    import tilelang.language as T
    from .solver_utils import warmstart_delta

    delta = warmstart_delta(layout)
    h = layout["header"]

    @T.macro
    def one(Tiles, i):
        base = i // 32 * layout["tile_words"]
        lane = i % 32
        cache = base + layout["warmstart"] + lane * 2 * layout["velocity_words"]
        va = T.alloc_local((6,), "float32")
        vb = T.alloc_local((6,), "float32")
        for d in T.serial(6):
            va[d] = 0.0
            vb[d] = 0.0
        delta(Tiles, i, va, vb)
        for d in T.serial(3):
            Tiles[cache + layout["velocity_linear"] + d] = T.reinterpret(
                va[d], "uint32"
            )
            Tiles[cache + layout["velocity_angular"] + d] = T.reinterpret(
                va[d + 3], "uint32"
            )
            Tiles[cache + layout["velocity_words"] + layout["velocity_linear"] + d] = (
                T.reinterpret(vb[d], "uint32")
            )
            Tiles[cache + layout["velocity_words"] + layout["velocity_angular"] + d] = (
                T.reinterpret(vb[d + 3], "uint32")
            )
        Tiles[cache + layout["velocity_padding0"]] = T.uint32(0)
        Tiles[cache + layout["velocity_words"] + layout["velocity_padding0"]] = (
            T.uint32(0)
        )
        Tiles[cache + layout["velocity_padding1"]] = Tiles[
            base + h["body_a"] + lane * 4 + 3
        ]
        Tiles[cache + layout["velocity_words"] + layout["velocity_padding1"]] = Tiles[
            base + h["body_b"] + lane * 4 + 3
        ]

    return one


def tile_warmstart(layout):
    """The tiled reader has the same force/torque order as constraint_warmstart."""
    import tilelang.language as T
    from .solver_utils import warmstart_delta

    delta = warmstart_delta(layout)

    @T.macro
    def one(Tiles, i, a, b):
        for d in T.serial(6):
            a[d] = 0.0
            b[d] = 0.0
        delta(Tiles, i, a, b)

    return one
