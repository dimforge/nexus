"""Compose the Rust TGS stage order over resident native buffers."""

from body_uniform_kernels import make_kernel as body_uniform_kernel
from body_kernels import make_kernel as body_kernel
from tgs_replay import body_case
from body_validation import BUFFERS as BODY_BUFFERS
from contact_kernels import make_kernel as contact_kernel
from contact_replay import BUFFERS as CONTACT_BUFFERS
from colored_kernels import make_kernel as colored_kernel, BUFFERS as COLORED_BUFFERS
from shaders.dynamics.solver import gpu_integrate_linearized
from tgs_uniform_kernels import make_kernel as uniform_kernel


def make_program(layout, case, target):
    plans = []
    cache = {}

    def add(kind, factory, mapping, blocks):
        if kind not in cache:
            cache[kind] = factory()
        plans.append(
            dict(
                name=f"{kind}_{len(plans)}",
                kernel=cache[kind],
                buffer_map=mapping,
                blocks=blocks,
                threads=64,
            )
        )

    def body(kind):
        add(
            kind,
            lambda: body_kernel(layout, body_case(case, kind), target),
            {n: n for n in BODY_BUFFERS},
            max((case["body_count"] + 63) // 64, 1),
        )

    def uniform(kind):
        add(
            kind,
            lambda: uniform_kernel(layout, case, target, kind),
            {
                n: n
                for n in (
                    "RbdParams",
                    "Gravity",
                    "Params",
                    "Tiles",
                    "Buckets",
                    "BatchIds",
                )
            },
            case["groups"] if kind == "scale_impulses" else 1,
        )

    def color(kind, c, mode):
        sizes = dict(
            case["buffer_words"],
            Velocities=case["buffer_words"]["SolverVelocities"],
            Poses=case["buffer_words"]["SolverPoses"],
            Color=2,
        )
        add(
            kind,
            lambda: colored_kernel(
                layout, dict(case, buffer_words=sizes), target, kind
            ),
            {
                n: {
                    "Velocities": "SolverVelocities",
                    "Poses": "SolverPoses",
                    "Color": f"Color{c}",
                    "Mode": mode,
                }.get(n, n)
                for n in COLORED_BUFFERS
            },
            case["groups"],
        )

    plans.append(
        dict(
            name="body_coefficients",
            kernel=body_uniform_kernel(layout, case, target),
            buffer_map={n: n for n in ("RbdParams", "Gravity", "Params")},
            blocks=1,
            threads=1,
        )
    )
    for kind in ["update_mprops", "cleanup", "init_bodies"]:
        body(kind)
    add(
        "prepare",
        lambda: contact_kernel(layout, case, target),
        {n: n for n in CONTACT_BUFFERS},
        case["groups"],
    )
    body("init_increments")
    for _ in range(case["substeps"]):
        body("apply_increments")
        if case["scale_warmstart"]:
            uniform("scale_impulses")
        for c in (1, 2):
            color("colored_warmstart", c, "Mode")
        for _ in range(max(case["pgs"], 1)):
            for c in (1, 2):
                color("colored_solve", c, "Mode")
        add(
            "integrate",
            lambda: gpu_integrate_linearized(
                layout, dict(case, active_bodies=case["body_count"]), target
            ),
            {
                "Tiles": "Tiles",
                "Velocities": "SolverVelocities",
                "Poses": "SolverPoses",
                "Params": "Params",
            },
            max((case["body_count"] + 63) // 64, 1),
        )
        for c in (1, 2):
            color("colored_solve", c, "Color0")
    body("finalize")
    return plans
