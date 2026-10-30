"""Broad-phase selection over scene storage; all pair work executes on the target."""

from lbvh_program_validation import BUFFERS as LBVH

SHARED = (
    "Pairs",
    "Groups",
    "Filters",
    "Poses",
    "Shapes",
    "Vertices",
    "PairCount",
    "BatchIds",
)
MAPPING = {name: name if name in SHARED else "Lbvh" + name for name in LBVH}
EXTRA = tuple(name for name in MAPPING.values() if name.startswith("Lbvh"))


def make_program(l, c, target):
    import broadphase_kernels as broad
    import lbvh_program_kernels as lbvh

    if c.get("broadphase") != "lbvh":
        return broad.make_program(l, dict(c, kind="broadphase_pairs"), target)
    # Retain world bounds for diagnostics and downstream spatial queries. Pair
    # generation uses only the resident Morton/radix/tree program.
    bc = dict(
        c,
        kind="world_aabbs",
        buffer_words={n: c["buffer_words"][n] for n in broad.BUFFERS},
    )
    stages = broad.make_program(l, bc, target)
    case = dict(
        c,
        kind="lbvh_program",
        groups=1,
        buffer_words={n: c["buffer_words"][p] for n, p in MAPPING.items()},
    )
    for stage in lbvh.make_program(l, case, target):
        stages.append(
            dict(
                stage,
                name="lbvh_" + stage["name"],
                buffer_map={n: MAPPING[p] for n, p in stage["buffer_map"].items()},
            )
        )
    return stages
