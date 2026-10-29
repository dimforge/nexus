#!/usr/bin/env python3
"""Contact preparation ABI validation and exact/float output comparison."""

from stage_replay import parse_args, run_stage

BUFFERS = (
    "Links",
    "Contacts",
    "WorldMass",
    "SolverPoses",
    "Velocities",
    "States",
    "OldTiles",
    "Tiles",
)


def validate_layout(layout):
    from shaders.dynamics.coulomb_tiles import validate_layout as tile_layout

    tile_layout(layout)

    def fields(stride, mapping):
        if type(stride) is not int or not 0 < stride < 2**20:
            raise ValueError("Invalid contact ABI stride")
        used = set()
        for offset, width in mapping:
            if type(offset) is not int or not 0 <= offset <= stride - width:
                raise ValueError("Invalid contact ABI offset")
            words = set(range(offset, offset + width))
            if words & used:
                raise ValueError("Overlapping contact ABI fields")
            used |= words

    l = layout
    fields(l["link_words"], [(v, 1) for v in l["link"].values()])
    fields(
        l["contact_words"],
        [
            (v, {"contact": l["manifold_words"], "colliders": 2, "bodies": 2}.get(k, 1))
            for k, v in l["contact"].items()
        ],
    )
    fields(
        l["manifold_words"],
        [
            (l["manifold"]["points_a"], 4 * l["manifold_point_words"]),
            (l["manifold"]["normal_a"], 3),
            (l["manifold"]["len"], 1),
        ],
    )
    fields(
        l["manifold_point_words"],
        [(l["manifold_point"]["pt"], 3), (l["manifold_point"]["dist"], 1)],
    )
    fields(
        l["world_mass_words"],
        [
            (v, {"inv_mass": 3, "inv_inertia": 16, "com": 3}.get(k, 1))
            for k, v in l["world_mass"].items()
        ],
    )
    fields(
        l["pose_words"],
        [(l["pose_rotation"], 4), (l["pose_translation"], 3), (l["pose_padding"], 1)],
    )
    fields(
        l["velocity_words"],
        [
            (l["velocity_linear"], 3),
            (l["velocity_angular"], 3),
            (l["velocity_padding0"], 1),
            (l["velocity_padding1"], 1),
        ],
    )
    fields(l["state_words"], [(l["state_pose_a"], l["pose_words"])])
    fields(l["point_words"], [(v, 32 * 4) for v in l["point"].values()])
    fields(
        l["tile_words"],
        [(v, 32 * 4) for v in l["header"].values()]
        + [
            (l["coupling"], 32 * 4),
            (l["points"], 4 * l["point_words"]),
            (l["warmstart"], 32 * 2 * l["velocity_words"]),
        ],
    )


def validate_case(layout, case, arrays):
    import numpy as np

    validate_layout(layout)
    for key in (
        "count",
        "storage",
        "bodies",
        "new_base",
        "states",
        "tiles",
        "old_tiles",
        "iterations",
    ):
        if type(case[key]) is not int or not 0 <= case[key] < 2**31:
            raise ValueError("Invalid contact dimensions")
    if (
        not 0 <= case["count"] <= case["storage"]
        or not 1 <= case["iterations"] <= 1024
        or min(case[k] for k in ("storage", "bodies", "states", "tiles", "old_tiles"))
        < 1
        or case["count"] > case["tiles"] * 32
        or case["new_base"] + case["storage"] > case["states"]
    ):
        raise ValueError("Invalid contact dimensions")
    sizes = {
        "Links": case["storage"] * layout["link_words"],
        "Contacts": case["storage"] * layout["contact_words"],
        "WorldMass": case["bodies"] * layout["world_mass_words"],
        "SolverPoses": case["bodies"] * layout["pose_words"],
        "Velocities": case["bodies"] * layout["velocity_words"],
        "States": case["states"] * layout["state_words"],
        "OldTiles": case["old_tiles"] * layout["tile_words"],
        "Tiles": case["tiles"] * layout["tile_words"],
    }
    if set(arrays) != set(BUFFERS) or any(
        a.dtype != np.uint32 or a.shape != (sizes[n],) or a.size >= 2**31
        for n, a in arrays.items()
    ):
        raise ValueError("Invalid contact buffer sizes")

    def finite(a):
        if not np.isfinite(a).all():
            raise ValueError("Non-finite contact input")

    def pose(a, b):
        q = a.view(np.float32)[
            b + layout["pose_rotation"] : b + layout["pose_rotation"] + 4
        ]
        finite(q)
        finite(
            a.view(np.float32)[
                b + layout["pose_translation"] : b + layout["pose_translation"] + 3
            ]
        )
        if abs(np.linalg.norm(q) - 1) > 2e-6:
            raise ValueError("Invalid contact pose")

    l, c, m, h, p = (
        layout[k] for k in ("link", "contact", "manifold", "header", "point")
    )
    for i in range(case["count"]):
        link = arrays["Links"][
            i * layout["link_words"] : (i + 1) * layout["link_words"]
        ]
        if link[l["len"]] == 0:
            continue
        if not 1 <= link[l["len"]] <= 4 or link[l["recycled"]] not in (0, 1):
            raise ValueError("Invalid contact link")
        j = int(link[l["contact"]])
        if j >= case["storage"]:
            raise ValueError("Invalid contact source")
        contact = arrays["Contacts"][
            j * layout["contact_words"] : (j + 1) * layout["contact_words"]
        ]
        mb = c["contact"]
        if not 1 <= contact[mb + m["len"]] <= 4:
            raise ValueError("Invalid manifold length")
        for d, side in enumerate(("a", "b")):
            bid = int(link[l[f"solver_body_{side}"]])
            if bid >= case["bodies"] or bid != contact[c["bodies"] + d]:
                raise ValueError("Invalid or inconsistent contact body")
            pose(arrays["SolverPoses"], bid * layout["pose_words"])
            wm = arrays["WorldMass"].view(np.float32)[
                bid * layout["world_mass_words"] : (bid + 1)
                * layout["world_mass_words"]
            ]
            for key, width in (("inv_mass", 3), ("inv_inertia", 16)):
                finite(
                    wm[layout["world_mass"][key] : layout["world_mass"][key] + width]
                )
            vel = arrays["Velocities"].view(np.float32)[
                bid * layout["velocity_words"] : (bid + 1) * layout["velocity_words"]
            ]
            finite(vel[layout["velocity_linear"] : layout["velocity_linear"] + 3])
            finite(vel[layout["velocity_angular"] : layout["velocity_angular"] + 3])
        lf = link.view(np.float32)
        scalars = lf[[l["mass_scale_a"], l["mass_scale_b"], l["restitution"]]]
        finite(scalars)
        if np.any(scalars < 0):
            raise ValueError("Negative contact scale")
        finite(contact.view(np.float32)[[c["friction"], c["restitution"]]])
        if np.any(contact.view(np.float32)[[c["friction"], c["restitution"]]] < 0):
            raise ValueError("Negative contact material")
        if not link[l["recycled"]]:
            pose(
                arrays["States"],
                (case["new_base"] + j) * layout["state_words"] + layout["state_pose_a"],
            )
            normal = contact.view(np.float32)[
                mb + m["normal_a"] : mb + m["normal_a"] + 3
            ]
            finite(normal)
            if abs(np.linalg.norm(normal) - 1) > 2e-6:
                raise ValueError("Invalid contact normal")
            for k in range(int(contact[mb + m["len"]])):
                off = mb + m["points_a"] + k * layout["manifold_point_words"]
                finite(
                    contact.view(np.float32)[off : off + layout["manifold_point_words"]]
                )
        if link[l["previous_constraint"]] != 2**32 - 1:
            old = int(link[l["previous_constraint_index"]])
            if old >= case["old_tiles"] * 32:
                raise ValueError("Invalid old contact index")
            ob = old // 32 * layout["tile_words"] + old % 32 * 4
            n = int(arrays["OldTiles"][ob + h["normal"] + 3])
            if not 0 <= n <= 4:
                raise ValueError("Invalid old contact length")
            of = arrays["OldTiles"].view(np.float32)
            direction = of[ob + h["normal"] : ob + h["normal"] + 3]
            tangent = of[ob + h["tangent"] : ob + h["tangent"] + 3]
            finite(direction)
            finite(tangent)
            if (
                abs(np.linalg.norm(direction) - 1) > 2e-6
                or abs(np.linalg.norm(tangent) - 1) > 2e-6
                or abs(np.dot(direction, tangent)) > 2e-6
            ):
                raise ValueError("Invalid old contact frame")
            for k in range(n):
                src = ob + layout["points"] + k * layout["point_words"]
                for field in p.values():
                    finite(of[src + field : src + field + 4])
        elif link[l["recycled"]]:
            raise ValueError("Recycled link has no previous constraint")


def output_masks(layout, case, before):
    """Float arithmetic, copied fields, and untouched storage have distinct policies."""
    import numpy as np

    mutable = np.zeros(before["Tiles"].size, dtype=bool)
    floats = mutable.copy()
    l, c, m, h, p = (
        layout[k] for k in ("link", "contact", "manifold", "header", "point")
    )
    for i in range(case["count"]):
        link = before["Links"][
            i * layout["link_words"] : (i + 1) * layout["link_words"]
        ]
        if link[l["len"]] == 0:
            continue
        base = i // 32 * layout["tile_words"] + i % 32 * 4
        j = int(link[l["contact"]])
        count = int(
            before["Contacts"][j * layout["contact_words"] + c["contact"] + m["len"]]
        )
        recycled = bool(link[l["recycled"]])
        if recycled:
            old = int(link[l["previous_constraint_index"]])
            ob = old // 32 * layout["tile_words"] + old % 32 * 4
            count = int(before["OldTiles"][ob + h["normal"] + 3])
        for key, offset in h.items():
            mutable[base + offset : base + offset + 4] = True
            floats[base + offset : base + offset + 3] = not (
                recycled and key in ("normal", "tangent")
            )
        mutable[base + layout["coupling"] : base + layout["coupling"] + 4] = True
        floats[base + layout["coupling"] : base + layout["coupling"] + count] = True
        for k in range(count):
            pb = base + layout["points"] + k * layout["point_words"]
            for offset in p.values():
                mutable[pb + offset : pb + offset + 4] = True
                floats[pb + offset : pb + offset + 4] = True
            # Distance and normal impulse are direct bit copies (or exact zero).
            floats[pb + p["local_a"] + 3] = False
            floats[pb + p["arm_a"] + 3] = False
            if recycled:
                for field in ("arm_a", "arm_b", "local_a", "local_b"):
                    floats[pb + p[field] : pb + p[field] + 3] = False
                floats[pb + p["friction"] : pb + p["friction"] + 2] = False
        cache = (
            i // 32 * layout["tile_words"]
            + layout["warmstart"]
            + i % 32 * 2 * layout["velocity_words"]
        )
        mutable[cache : cache + 2 * layout["velocity_words"]] = True
        for side in (0, 1):
            for field in ("velocity_linear", "velocity_angular"):
                off = cache + side * layout["velocity_words"] + layout[field]
                floats[off : off + 3] = True
    return mutable, floats


def check_outputs(layout, case, before, actual, expected, contract):
    import numpy as np

    validate_case(layout, case, before)
    mutable, floats = output_masks(layout, case, before)
    errors = {}
    for name in BUFFERS:
        a, e = actual[name], expected[name]
        if (
            a.dtype != np.uint32
            or e.dtype != np.uint32
            or a.shape != before[name].shape
            or e.shape != a.shape
        ):
            raise ValueError("Invalid contact output size")
        if name != "Tiles":
            np.testing.assert_array_equal(a, before[name])
            errors[name] = 0.0
        else:
            np.testing.assert_array_equal(a[~mutable], before[name][~mutable])
            np.testing.assert_array_equal(e[~mutable], before[name][~mutable])
            np.testing.assert_array_equal(a[~floats], e[~floats])
            af, ef = a.view(np.float32)[floats], e.view(np.float32)[floats]
            if not np.isfinite(af).all() or not np.isfinite(ef).all():
                raise AssertionError("Non-finite prepared constraint")
            np.testing.assert_allclose(
                af, ef, atol=contract["atol"], rtol=contract["rtol"]
            )
            errors[name] = float(np.max(np.abs(af - ef))) if af.size else 0.0
    return errors


def main():
    from contact_kernels import make_kernel

    run_stage(
        parse_args(["prepare"], "prepare"),
        oracle="contacts",
        buffers=BUFFERS,
        make_kernel=make_kernel,
        validate_case=validate_case,
        check_outputs=check_outputs,
        launch=lambda case: (
            max((case["count"] + 63) // 64, 1),
            64,
            case["iterations"],
        ),
        extra_files=("ContactPlan.bin", "RecycleOffsets.bin"),
    )


if __name__ == "__main__":
    main()
