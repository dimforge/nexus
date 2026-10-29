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
