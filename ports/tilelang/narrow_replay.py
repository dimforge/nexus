"""Native analytic contact ABI and geometry-aware parity checks."""

BUFFERS = (
    "Pairs",
    "Plan",
    "Poses",
    "Shapes",
    "Contacts",
    "Parents",
    "Materials",
    "RbdParams",
)


def validate_case(layout, case, arrays):
    import numpy as np

    fields = ("count", "storage", "colliders", "groups", "iterations")
    if case["kind"] != "analytic_contacts" or any(
        type(case[k]) is not int or not 0 <= case[k] < 2**27 for k in fields
    ):
        raise ValueError("Invalid analytic contact dimensions")
    if not (
        case["count"] < case["storage"]
        and case["colliders"] > 0
        and 0 < case["groups"] * 64 < 2**31
        and case["iterations"] == 1
    ):
        raise ValueError("Invalid analytic contact capacities")
    sizes = {
        n: case["storage"] * layout[key]
        for n, key in [("Pairs", "pair_words"), ("Contacts", "contact_words")]
    }
    sizes.update(
        {
            n: case["colliders"] * layout[key]
            for n, key in [
                ("Poses", "pose_words"),
                ("Shapes", "shape_words"),
                ("Materials", "material_words"),
            ]
        }
    )
    sizes.update(
        Parents=case["colliders"],
        Plan=layout["plan_words"],
        RbdParams=layout["rbd_params_words"],
    )
    if (
        any(size >= 2**31 for size in sizes.values())
        or set(arrays) != set(BUFFERS)
        or case["buffer_words"] != sizes
        or any(
            a.dtype != np.uint32 or a.shape != (sizes[n],) for n, a in arrays.items()
        )
    ):
        raise ValueError("Invalid analytic contact buffers")
    plan = arrays["Plan"]
    p = layout["plan"]
    if (
        plan[p["pfm_base"]] != case["count"]
        or int(plan[p["bound"]]) != case["count"]
        or plan[p["pfm_len"]] != 0
    ):
        raise ValueError("Invalid analytic-only contact plan")
    params = arrays["RbdParams"].view(np.float32)
    pf = layout["rbd_params"]
    v = params[[pf["normalized_prediction_distance"], pf["length_unit"]]]
    if (
        not np.isfinite(v).all()
        or v[0] < 0
        or v[1] <= 0
        or not np.isfinite(np.float32(v[0] * v[1]))
    ):
        raise ValueError("Invalid prediction distance")
    pairs = arrays["Pairs"].reshape(-1, layout["pair_words"])[
        : case["count"], layout["pair_colliders"] : layout["pair_colliders"] + 2
    ]
    if np.any(pairs >= case["colliders"]):
        raise ValueError("Invalid collider pair")
    poses = arrays["Poses"].view(np.float32).reshape(-1, layout["pose_words"])
    shapes = arrays["Shapes"].reshape(-1, layout["shape_words"])
    materials = (
        arrays["Materials"].view(np.float32).reshape(-1, layout["material_words"])
    )
    sf = layout["shape"]
    types = layout["shape_types"]
    for i in np.unique(pairs):
        q = poses[i, layout["pose_rotation"] : layout["pose_rotation"] + 4]
        t = poses[i, layout["pose_translation"] : layout["pose_translation"] + 3]
        if (
            not np.isfinite(np.r_[q, t]).all()
            or abs(float(np.linalg.norm(q)) - 1) > 2e-6
        ):
            raise ValueError("Invalid collider pose")
        f = layout["material"]
        if not np.isfinite(materials[i, [f["friction"], f["restitution"]]]).all():
            raise ValueError("Nonfinite collider material")
        s = shapes[i]
        data = s.view(np.float32)
        ty = int(s[sf["a"] + 3])
        a = data[sf["a"] : sf["a"] + 3]
        b = data[sf["b"] : sf["b"] + 3]
        if ty == types["ball"]:
            geometry = a[:1]
            extents = geometry
        elif ty == types["cuboid"]:
            geometry = a
            extents = a
        elif ty in (types["cone"], types["cylinder"]):
            geometry = a[:2]
            extents = geometry
        elif ty == types["capsule"]:
            geometry = np.r_[a, b, data[sf["b"] + 3]]
            extents = geometry[-1:]
        else:
            continue
        if not np.isfinite(geometry).all() or np.any(extents < 0):
            raise ValueError("Invalid primitive geometry")


def check_outputs(layout, case, before, actual, expected, contract):
    import numpy as np
    from narrow_symmetry import symmetry_frame, validate_symmetry_contact
    from narrow_box_geometry import box_frame, validate_box_contact

    validate_case(layout, case, before)
    if set(actual) != set(before) or set(expected) != set(before):
        raise ValueError("Missing analytic contact output")
    for n in before:
        if (
            actual[n].dtype != np.uint32
            or actual[n].shape != before[n].shape
            or expected[n].shape != before[n].shape
        ):
            raise ValueError("Invalid analytic contact output shape")
        if n != "Contacts":
            np.testing.assert_array_equal(
                actual[n], before[n], err_msg=f"Modified readonly {n}"
            )
    cw = layout["contact_words"]
    a = actual["Contacts"].reshape(-1, cw)
    e = expected["Contacts"].reshape(-1, cw)
    b = before["Contacts"].reshape(-1, cw)
    np.testing.assert_array_equal(
        a[case["count"] :], b[case["count"] :], err_msg="Contact capacity guard"
    )
    cf, mf, pf = layout["contact"], layout["manifold"], layout["manifold_point"]
    m = cf["contact"]
    li = m + mf["len"]
    pairs = before["Pairs"].reshape(-1, layout["pair_words"])
    atol, rtol = contract["atol"], contract["rtol"]
    max_error = 0.0
    for i in range(case["count"]):
        length = int(e[i, li])
        if length > 4 or a[i, li] != length:
            raise AssertionError("Contact count differs")
        if not length:
            mask = np.arange(cw) != li
            np.testing.assert_array_equal(
                a[i, mask], b[i, mask], err_msg="Inert slot must change len only"
            )
            continue
        floats = np.zeros(cw, dtype=bool)
        for pi in range(length):
            p = m + mf["points_a"] + pi * layout["manifold_point_words"]
            floats[p + pf["pt"] : p + pf["pt"] + 3] = True
            floats[p + pf["dist"]] = True
        floats[m + mf["normal_a"] : m + mf["normal_a"] + 3] = True
        for f in ("friction", "restitution", "recycle_extent"):
            floats[cf[f]] = True
        np.testing.assert_array_equal(
            a[i, ~floats], e[i, ~floats], err_msg="Exact contact fields"
        )
        ids = pairs[i, layout["pair_colliders"] : layout["pair_colliders"] + 2]
        np.testing.assert_array_equal(a[i, cf["colliders"] : cf["colliders"] + 2], ids)
        np.testing.assert_array_equal(
            a[i, cf["bodies"] : cf["bodies"] + 2], before["Parents"][ids]
        )
        af, ef = a[i].view(np.float32), e[i].view(np.float32)
        if not np.isfinite(af[floats]).all() or not np.isfinite(ef[floats]).all():
            raise AssertionError("Nonfinite generated contact")
        max_error = max(max_error, float(np.max(np.abs(af[floats] - ef[floats]))))
        frame = symmetry_frame(layout, before, *map(int, ids))
        if frame is not None:
            if length != 1:
                raise AssertionError("Axial primitive contact must have one point")
            # Both implementations must produce actual closest surface points.
            # This rejects the old Rust capsule defect instead of legitimizing it.
            for row in (a[i], e[i]):
                validate_symmetry_contact(layout, frame, row, atol, rtol)
            p = m + mf["points_a"]
            floats[p + pf["pt"] : p + pf["pt"] + 3] = False
            floats[m + mf["normal_a"] : m + mf["normal_a"] + 3] = False
        box = box_frame(layout, before, *map(int, ids), contract)
        if box is not None:
            geometry = np.zeros(cw, dtype=bool)
            for pi in range(length):
                p = m + mf["points_a"] + pi * layout["manifold_point_words"]
                geometry[p + pf["pt"] : p + pf["pt"] + 3] = True
                geometry[p + pf["dist"]] = True
            for row in (a[i], e[i]):
                validate_box_contact(layout, box, row, length, atol, rtol)
            np.testing.assert_allclose(
                af[geometry],
                ef[geometry],
                atol=box[-1],
                rtol=rtol,
                err_msg=f"Box contact geometry {i}",
            )
            floats[geometry] = False
        np.testing.assert_allclose(
            af[floats], ef[floats], atol=atol, rtol=rtol, err_msg=f"Contact {i}"
        )
    return {"Contacts": max_error}


def main():
    from stage_replay import parse_args, run_stage
    from narrow_kernels import make_program

    run_stage(
        parse_args(["analytic_contacts"], "analytic_contacts"),
        oracle="narrow",
        buffers=BUFFERS,
        make_kernel=None,
        make_program=make_program,
        validate_case=validate_case,
        check_outputs=check_outputs,
        launch=lambda c: (c["groups"], 64, 1),
    )


if __name__ == "__main__":
    main()
