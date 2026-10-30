"""Resident bounds, pairs, contact plan and analytic contact generation."""

import broadphase_kernels as broad
import contact_plan_kernels as plan
import narrow_kernels as narrow
import scene_broadphase

BUFFERS = tuple(dict.fromkeys((*broad.BUFFERS, *narrow.BUFFERS, *plan.BUFFERS)))


def make_program(layout, case, target):
    def subcase(buffers, **changes):
        return dict(
            case, buffer_words={n: case["buffer_words"][n] for n in buffers}, **changes
        )

    reset = plan.make_program(
        layout,
        subcase(plan.BUFFERS, kind="reset_narrow", count=1, groups=1, threads=64),
        target,
    )
    bounds_and_pairs = scene_broadphase.make_program(layout, case, target)
    publish = plan.make_program(
        layout,
        subcase(plan.BUFFERS, kind="contact_plan", count=1, groups=1, threads=1),
        target,
    )
    contacts = narrow.make_program(
        layout,
        subcase(
            narrow.BUFFERS, kind="analytic_contacts", groups=case["contact_groups"]
        ),
        target,
    )
    return reset + bounds_and_pairs + publish + contacts
