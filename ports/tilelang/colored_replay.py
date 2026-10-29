"""Draft native color-range contracts and exact preservation checks."""

BUFFERS = (
    "Tiles",
    "Velocities",
    "Poses",
    "RbdParams",
    "Buckets",
    "BatchIds",
    "Color",
    "Color1",
    "Color2",
    "Color3",
    "Mode",
)
OPERATIONS = ("colored_solve", "colored_warmstart", "colored_cached", "colored_program")


def validate_case(layout, case, arrays):
    import numpy as np

    n = case["count"]
    nb = case["batches"]
    op = case["kind"]
    if (
        op not in OPERATIONS
        or type(n) is not int
        or not 0 <= n < 1 << 20
        or not 1 <= nb <= 64
        or not 1 <= case["groups"] <= 256
        or not 1 <= case["iterations"] <= 64
    ):
        raise ValueError("Invalid native color dimensions")
    sizes = {
        "Tiles": case["tiles"] * layout["tile_words"],
        "Velocities": case["bodies"] * layout["velocity_words"],
        "Poses": case["bodies"] * layout["pose_words"],
        "RbdParams": layout["rbd_params_words"],
        "Buckets": 5 * nb,
        "BatchIds": layout["batch_words"],
        **{name: 2 for name in ["Color", "Color1", "Color2", "Color3", "Mode"]},
    }
    if case["buffer_words"] != sizes or set(arrays) != set(sizes):
        raise ValueError("Invalid color-buffer ABI")
    for name, size in sizes.items():
        a = arrays[name]
        if a.dtype != np.uint32 or a.ndim != 1 or a.size != size:
            raise ValueError(f"Invalid {name} buffer")
    if case["tiles"] < max((n + 31) // 32, 1):
        raise ValueError("Short tile storage")
    b = layout["batch"]
    if (
        int(arrays["BatchIds"][b["num_batches"]]) != nb
        or int(arrays["BatchIds"][b["solver_color_buckets_stride"]]) != 5
    ):
        raise ValueError("Invalid native batch uniform")
    buckets = arrays["Buckets"].astype(np.uint64)
    if any(buckets[:nb]) or np.any(buckets[1:] < buckets[:-1]) or int(buckets[-1]) != n:
        raise ValueError("Invalid color bucket ends")
    if not 1 <= int(arrays["Color"][0]) <= 3 or any(
        int(arrays[f"Color{c}"][0]) != c for c in range(1, 4)
    ):
        raise ValueError("Invalid color uniform")
    if int(arrays["Mode"][0]) != case["mode"]:
        raise ValueError("Mode differs from fixture")
    params = arrays["RbdParams"].view(np.float32)
    f = layout["rbd_params"]
    for key in f:
        if not np.isfinite(params[f[key]]):
            raise ValueError("Nonfinite solver parameter")
    if (
        params[f["dt"]] < 0
        or params[f["length_unit"]] <= 0
        or params[f["normalized_max_corrective_velocity"]] < 0
    ):
        raise ValueError("Invalid timestep or scale")
    for prefix in ("contact", "static_contact"):
        if (
            params[f[prefix + "_natural_frequency"]] < 0
            or params[f[prefix + "_damping_ratio"]] <= 0
        ):
            raise ValueError("Invalid spring coefficients")
    floats = {name: np.zeros(size, dtype=bool) for name, size in sizes.items()}
    integers = {name: np.zeros(size, dtype=bool) for name, size in sizes.items()}
    selected = (
        list(range(1, 4)) if op == "colored_program" else [int(arrays["Color"][0])]
    )
    cached = op in ("colored_cached", "colored_program")
    solve = op != "colored_warmstart"
    friction = cached or case["mode"] != 1
    h, p = layout["header"], layout["point"]
    active = []
    for color in selected:
        used = set()
        readonly = set()
        start, end = map(int, (buckets[color * nb - 1], buckets[(color + 1) * nb - 1]))
        for i in range(start, end):
            base = i // 32 * layout["tile_words"] + i % 32 * 4
            length = int(arrays["Tiles"][base + h["normal"] + 3])
            if length > 4:
                raise ValueError("Invalid point count")
            for field in ("body_a", "body_b", "diag_a", "off_a"):
                index = int(arrays["Tiles"][base + h[field] + 3])
                if index >= case["bodies"]:
                    raise ValueError("Invalid body or velocity slot")
                if field in ("diag_a", "off_a"):
                    side = "a" if field == "diag_a" else "b"
                    source = arrays["Tiles"].view(np.float32)
                    writable = any(
                        source[base + h[f"{member}_{side}"] + d] != 0
                        for member in ("body", "diag", "off")
                        for d in range(3)
                    )
                    if index in used or (writable and index in readonly):
                        raise ValueError("Velocity writes overlap within a color")
                    if writable:
                        used.add(index)
                        for offset in (
                            layout["velocity_linear"],
                            layout["velocity_angular"],
                        ):
                            floats["Velocities"][
                                index * layout["velocity_words"] + offset : index
                                * layout["velocity_words"]
                                + offset
                                + 3
                            ] = True
                    else:
                        readonly.add(index)
                else:
                    q = index * layout["pose_words"] + layout["pose_rotation"]
                    quat = arrays["Poses"].view(np.float32)[q : q + 4]
                    if (
                        not np.isfinite(quat).all()
                        or abs(float(np.linalg.norm(quat)) - 1) > 2e-6
                    ):
                        raise ValueError("Invalid pose rotation")
            if solve:
                for k in range(length):
                    point = base + layout["points"] + k * layout["point_words"]
                    floats["Tiles"][point + p["arm_a"] + 3] = True
                    if friction:
                        floats["Tiles"][
                            point + p["friction"] : point + p["friction"] + 2
                        ] = True
            if cached:
                cache = (
                    i // 32 * layout["tile_words"]
                    + layout["warmstart"]
                    + i % 32 * 2 * layout["velocity_words"]
                )
                for side in range(2):
                    c = cache + side * layout["velocity_words"]
                    for offset in (
                        layout["velocity_linear"],
                        layout["velocity_angular"],
                    ):
                        floats["Tiles"][c + offset : c + offset + 3] = True
                    for offset in (
                        layout["velocity_padding0"],
                        layout["velocity_padding1"],
                    ):
                        integers["Tiles"][c + offset] = True
            active.append(i)
    return floats, integers, active


def check_outputs(layout, case, before, actual, expected, contract):
    import numpy as np

    floats, integers, active = validate_case(layout, case, before)
    errors = {}
    for name, mask in floats.items():
        a = actual[name]
        b = before[name]
        e = expected[name]
        if a.shape != b.shape or a.dtype != np.uint32:
            raise ValueError("Invalid returned buffer")
        np.testing.assert_array_equal(
            a[~(mask | integers[name])],
            b[~(mask | integers[name])],
            err_msg=f"Unexpected write to {name}",
        )
        np.testing.assert_array_equal(
            a[integers[name]],
            e[integers[name]],
            err_msg=f"Incorrect integer output {name}",
        )
        af, ef = a.view(np.float32)[mask], e.view(np.float32)[mask]
        if not np.isfinite(af).all() or not np.isfinite(ef).all():
            raise AssertionError("Nonfinite solver result")
        np.testing.assert_allclose(
            af, ef, atol=contract["atol"], rtol=contract["rtol"], err_msg=name
        )
        errors[name] = float(np.max(np.abs(af - ef))) if af.size else 0.0
    h, point = layout["header"], layout["point"]
    tiles = actual["Tiles"].view(np.float32)
    cached = case["kind"] in ("colored_cached", "colored_program")
    friction = cached or (case["kind"] == "colored_solve" and case["mode"] != 1)
    for i in active:
        base = i // 32 * layout["tile_words"] + i % 32 * 4
        if cached:
            start = (
                i // 32 * layout["tile_words"]
                + layout["warmstart"]
                + i % 32 * 2 * layout["velocity_words"]
            )
            for side, body in enumerate(("body_a", "body_b")):
                cell = start + side * layout["velocity_words"]
                if (
                    actual["Tiles"][cell + layout["velocity_padding0"]] != 0
                    or actual["Tiles"][cell + layout["velocity_padding1"]]
                    != before["Tiles"][base + h[body] + 3]
                ):
                    raise AssertionError(
                        "Warmstart cache body tag or padding is invalid"
                    )
        if case["kind"] != "colored_warmstart":
            count = int(before["Tiles"][base + h["normal"] + 3])
            for k in range(count):
                offset = base + layout["points"] + k * layout["point_words"]
                normal = float(tiles[offset + point["arm_a"] + 3])
                if normal < 0:
                    raise AssertionError("Negative normal impulse")
                if friction:
                    limit = float(tiles[base + h["tangent"] + 3]) * normal
                    tangent = tiles[
                        offset + point["friction"] : offset + point["friction"] + 2
                    ]
                    if float(np.linalg.norm(tangent)) > limit + contract[
                        "atol"
                    ] + contract["rtol"] * abs(limit):
                        raise AssertionError("Friction impulse outside Coulomb disk")
    return errors


def main():
    from colored_kernels import make_program
    from stage_replay import parse_args, run_stage

    args = parse_args(OPERATIONS, "colored_program")
    run_stage(
        args,
        oracle="colored",
        buffers=BUFFERS,
        make_kernel=None,
        make_program=make_program,
        validate_case=validate_case,
        check_outputs=check_outputs,
        launch=lambda case: (case["groups"], 64, case["iterations"]),
    )


if __name__ == "__main__":
    main()
