"""Native bounds and pair-buffer contracts; atomic pair order is unspecified."""

BUFFERS = (
    "Poses",
    "Shapes",
    "Aabbs",
    "BatchIds",
    "Vertices",
    "Pairs",
    "PairCount",
    "Groups",
    "Filters",
    "RbdParams",
)
OPERATIONS = ("world_aabbs", "brute_pairs", "brute_pairs_serial", "broadphase_pairs")


def validate_case(layout, case, arrays):
    import numpy as np

    integer_fields = (
        "count",
        "colliders",
        "batches",
        "capacity",
        "storage",
        "pair_capacity",
        "pair_storage",
        "vertices",
        "dispatch",
        "groups",
        "iterations",
        "initial_count",
    )
    if case["kind"] not in OPERATIONS or any(
        type(case[k]) is not int or not 0 <= case[k] < 2**27 for k in integer_fields
    ):
        raise ValueError("Invalid broad-phase dimensions")
    n, nb = case["colliders"], case["batches"]
    if (
        not 1 <= nb <= 64
        or n > 4096
        or case["count"] != n * nb
        or n > case["capacity"]
        or case["capacity"] * nb > case["storage"]
        or case["pair_capacity"] >= case["pair_storage"]
        or case["initial_count"] > case["pair_capacity"]
        or case["vertices"] == 0
        or case["iterations"] != 1
        or case["storage"] == 0
    ):
        raise ValueError("Invalid broad-phase capacities")
    dispatch = (
        n * nb if case["kind"] in ("world_aabbs", "brute_pairs_serial") else n * n * nb
    )
    if case["dispatch"] != dispatch or case["groups"] != max((dispatch + 63) // 64, 1):
        raise ValueError("Invalid broad-phase launch")
    sizes = {
        "Poses": case["storage"] * layout["pose_words"],
        "Shapes": case["storage"] * layout["shape_words"],
        "Aabbs": case["storage"] * layout["aabb_words"],
        "BatchIds": layout["batch_words"],
        "Vertices": case["vertices"] * layout["vertex_words"],
        "Pairs": case["pair_storage"] * layout["pair_words"],
        "PairCount": 3,
        "Groups": case["storage"] * layout["group_words"],
        "Filters": case["storage"] * 2,
        "RbdParams": layout["rbd_params_words"],
    }
    if (
        set(arrays) != set(BUFFERS)
        or case["buffer_words"] != sizes
        or any(
            a.dtype != np.uint32 or a.shape != (sizes[name],)
            for name, a in arrays.items()
        )
    ):
        raise ValueError("Invalid broad-phase buffers")
    b = layout["batch"]
    ids = arrays["BatchIds"]
    for key, expected in [
        ("num_batches", nb),
        ("colliders_len", n),
        ("colliders_batch_capacity", case["capacity"]),
        ("collision_pairs_capacity", case["pair_capacity"]),
    ]:
        if ids[b[key]] != expected:
            raise ValueError("Native broad-phase uniform mismatch")
    if arrays["PairCount"][0] != case["initial_count"]:
        raise ValueError("Invalid initial pair count")
    p = arrays["RbdParams"].view(np.float32)
    f = layout["rbd_params"]
    values = p[[f["normalized_prediction_distance"], f["length_unit"]]]
    if (
        not np.isfinite(values).all()
        or values[0] < 0
        or values[1] <= 0
        or not np.isfinite(np.float32(values[0] * values[1]))
    ):
        raise ValueError("Invalid broad-phase prediction distance")
    if case["kind"] in ("world_aabbs", "broadphase_pairs"):
        sf = layout["shape"]
        types = layout["shape_types"]
        poses = arrays["Poses"].view(np.float32).reshape(-1, layout["pose_words"])
        shapes = arrays["Shapes"].reshape(-1, layout["shape_words"])
        vertices = (
            arrays["Vertices"].view(np.float32).reshape(-1, layout["vertex_words"])
        )
        for i in range(n * nb):
            q = poses[i, layout["pose_rotation"] : layout["pose_rotation"] + 4]
            t = poses[i, layout["pose_translation"] : layout["pose_translation"] + 3]
            if (
                not np.isfinite(q).all()
                or not np.isfinite(t).all()
                or abs(float(np.linalg.norm(q)) - 1) > 2e-6
            ):
                raise ValueError("Invalid shape pose")
            s = shapes[i]
            fs = s.view(np.float32)
            ty = int(s[sf["a"] + 3])
            a = fs[sf["a"] : sf["a"] + 3]
            v = fs[sf["b"] : sf["b"] + 3]
            c = fs[sf["c"] : sf["c"] + 3]

            def finite(values):
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite shape geometry")

            if ty == types["ball"]:
                finite(a[:1])
                if a[0] < 0:
                    raise ValueError("Negative shape radius")
            elif ty == types["cuboid"]:
                finite(a)
                if np.any(a < 0):
                    raise ValueError("Negative box extent")
            elif ty in (types["cone"], types["cylinder"]):
                finite(a[:2])
                if np.any(a[:2] < 0):
                    raise ValueError("Negative axial shape extent")
            elif ty == types["capsule"]:
                finite(np.r_[a, v, fs[sf["b"] + 3]])
                if fs[sf["b"] + 3] < 0:
                    raise ValueError("Negative capsule radius")
            elif ty == types["triangle"]:
                finite(np.r_[a, v, c])
            elif ty in (types["trimesh"], types["polyline"]):
                finite(np.r_[v, c])
                if np.any(v > c):
                    raise ValueError("Inverted composite root bounds")
            elif ty == types["convex_poly"]:
                first, end = map(int, s[sf["a"] : sf["a"] + 2])
                # Rust reads the first vertex even for an empty range.
                if (
                    not 0 <= first < case["vertices"]
                    or not first <= end <= case["vertices"]
                ):
                    raise ValueError("Invalid convex vertex range")
                finite(
                    vertices[
                        first : max(end, first + 1),
                        layout["vertex_vector"] : layout["vertex_vector"] + 3,
                    ]
                )
    else:
        bounds = arrays["Aabbs"].view(np.float32).reshape(-1, layout["aabb_words"])
        a = layout["aabb"]
        for batch in range(nb):
            used = bounds[batch * case["capacity"] : batch * case["capacity"] + n]
            lo = used[:, a["mins"] : a["mins"] + 3]
            hi = used[:, a["maxs"] : a["maxs"] + 3]
            if (
                not np.isfinite(lo).all()
                or not np.isfinite(hi).all()
                or np.any(lo > hi)
            ):
                raise ValueError("Invalid input AABB")
    groups = arrays["Groups"].reshape(-1, layout["group_words"])[
        : n * nb, layout["groups"]["test_mode"]
    ]
    if np.any((groups != 0) & (groups != 1)):
        raise ValueError("Invalid Rust interaction test enum")


def candidates(layout, case, arrays, bounds):
    """Independent f32 predicate set; never used to generate runtime physics."""
    import numpy as np

    groups = arrays["Groups"].reshape(-1, layout["group_words"])
    filters = arrays["Filters"].reshape(-1, 2)
    boxes = bounds.view(np.float32).reshape(-1, layout["aabb_words"])
    p = arrays["RbdParams"].view(np.float32)
    f = layout["rbd_params"]
    dilation = np.float32(p[f["normalized_prediction_distance"]] * p[f["length_unit"]])
    g = layout["groups"]
    a = layout["aabb"]
    result = set()
    nb = case["batches"]
    n = case["colliders"]
    for batch in range(nb):
        for i in range(n):
            ia = i * nb + batch
            ga = groups[ia]
            ai = boxes[batch * case["capacity"] + i]
            lo = ai[a["mins"] : a["mins"] + 3] - dilation
            hi = ai[a["maxs"] : a["maxs"] + 3] + dilation
            for j in range(i + 1, n):
                ib = j * nb + batch
                gb = groups[ib]
                left = bool(ga[g["memberships"]] & gb[g["filter"]])
                right = bool(gb[g["memberships"]] & ga[g["filter"]])
                allowed = (
                    (left or right)
                    if ga[g["test_mode"]] == gb[g["test_mode"]] == 1
                    else (left and right)
                )
                if (
                    not allowed
                    or filters[ia, 0] == filters[ib, 0]
                    or (filters[ia, 1] != 0 and filters[ia, 1] == filters[ib, 1])
                ):
                    continue
                aj = boxes[batch * case["capacity"] + j]
                if np.all(lo <= aj[a["maxs"] : a["maxs"] + 3]) and np.all(
                    hi >= aj[a["mins"] : a["mins"] + 3]
                ):
                    result.add((ia, ib))
    return result


def check_outputs(layout, case, before, actual, expected, contract):
    import numpy as np

    validate_case(layout, case, before)
    if set(actual) != set(before) or set(expected) != set(before):
        raise ValueError("Missing broad-phase output")
    for name in before:
        if (
            actual[name].dtype != np.uint32
            or actual[name].shape != before[name].shape
            or expected[name].shape != before[name].shape
        ):
            raise ValueError("Invalid returned broad-phase buffer")
        if name not in ("Aabbs", "Pairs", "PairCount"):
            np.testing.assert_array_equal(
                actual[name], before[name], err_msg=f"Modified readonly {name}"
            )
    errors = {}
    bounds = case["kind"] in ("world_aabbs", "broadphase_pairs")
    if bounds:
        floats = np.zeros(before["Aabbs"].size, dtype=bool)
        mutable = floats.copy()
        a = layout["aabb"]
        for b in range(case["batches"]):
            for i in range(case["colliders"]):
                base = (b * case["capacity"] + i) * layout["aabb_words"]
                mutable[base : base + layout["aabb_words"]] = True
                for field in ("mins", "maxs"):
                    floats[base + a[field] : base + a[field] + 3] = True
        np.testing.assert_array_equal(
            actual["Aabbs"][~mutable],
            before["Aabbs"][~mutable],
            err_msg="AABB guard overwrite",
        )
        np.testing.assert_array_equal(
            actual["Aabbs"][mutable & ~floats],
            np.zeros(np.count_nonzero(mutable & ~floats), dtype=np.uint32),
            err_msg="AABB padding",
        )
        af, ef = (
            actual["Aabbs"].view(np.float32)[floats],
            expected["Aabbs"].view(np.float32)[floats],
        )
        if not np.isfinite(af).all() or not np.isfinite(ef).all():
            raise AssertionError("Nonfinite world bounds")
        np.testing.assert_allclose(af, ef, atol=contract["atol"], rtol=contract["rtol"])
        errors["Aabbs"] = float(np.max(np.abs(af - ef))) if af.size else 0.0
    else:
        np.testing.assert_array_equal(actual["Aabbs"], before["Aabbs"])
    if case["kind"] == "world_aabbs":
        for n in ("Pairs", "PairCount"):
            np.testing.assert_array_equal(actual[n], before[n])
        return errors
    allowed = candidates(layout, case, before, expected["Aabbs"])
    initial = case["initial_count"]
    total = initial + len(allowed)
    if int(actual["PairCount"][0]) != total or int(expected["PairCount"][0]) != total:
        raise AssertionError("Pair counter must include capacity overflow")
    np.testing.assert_array_equal(actual["PairCount"][1:], before["PairCount"][1:])
    start = min(initial, case["pair_capacity"])
    end = min(total, case["pair_capacity"])
    a = actual["Pairs"].reshape(-1, layout["pair_words"])
    b = before["Pairs"].reshape(a.shape)
    np.testing.assert_array_equal(a[:start], b[:start])
    np.testing.assert_array_equal(
        a[end:], b[end:], err_msg="Pair capacity guard overwrite"
    )
    off = layout["pair_colliders"]
    found = set(map(tuple, a[start:end, off : off + 2].tolist()))
    if len(found) != end - start or not found.issubset(allowed):
        raise AssertionError("Illegal or duplicate collision pair")
    if total <= case["pair_capacity"] and found != allowed:
        raise AssertionError("Missing collision pair")
    errors["PairCount"] = 0
    return errors


def main():
    from stage_replay import parse_args, run_stage
    from broadphase_kernels import make_program

    run_stage(
        parse_args(OPERATIONS, "broadphase_pairs"),
        oracle="broadphase",
        buffers=BUFFERS,
        make_kernel=None,
        make_program=make_program,
        validate_case=validate_case,
        check_outputs=check_outputs,
        launch=lambda c: (c["groups"], 64, 1),
    )


if __name__ == "__main__":
    main()
