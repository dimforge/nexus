"""Build-time entry selection; physics lives in the mirrored shaders/ tree."""

from kernel_support import launch
from shaders.dynamics import constraint, solver, tile_kernels


def make_kernel(layout, case, target):
    kind = case.get("kind", "solve")
    if kind == "solve":
        return tile_kernels.gpu_solve_constraints(layout, case, target)
    if kind == "integrate":
        return solver.gpu_integrate_linearized(layout, case, target)
    if kind == "effective_mass":
        return launch(
            constraint.compute_effective_masses(layout, case),
            layout,
            case,
            target,
            case["count"],
        )
    if kind == "warmstart":
        return tile_kernels.gpu_warmstart_constraints(layout, case, target)
    raise ValueError(f"Unknown operation {kind}")
