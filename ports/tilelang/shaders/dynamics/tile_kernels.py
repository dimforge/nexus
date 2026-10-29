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
