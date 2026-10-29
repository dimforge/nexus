"""Validate the resident native ball/box collision program by exact pair identity."""

import broadphase_replay as broad
import narrow_replay as narrow
import contact_plan_replay as plan

BUFFERS = tuple(dict.fromkeys((*broad.BUFFERS, *narrow.BUFFERS, *plan.BUFFERS)))


def subset(case, buffers, **changes):
    return dict(
        case, buffer_words={n: case["buffer_words"][n] for n in buffers}, **changes
    )


def select(data, names):
    return {n: data[n] for n in names}


def validate_case(layout, case, arrays):
    import numpy as np

    if (
        case["kind"] != "analytic_collision"
        or case["initial_count"] != 0
        or any(
            type(case[k]) is not int or not 0 < case[k] < 2**27
            for k in ("contact_storage", "contact_groups")
        )
    ):
        raise ValueError("Invalid analytic collision dimensions")
    if (
        case["contact_storage"] != 2 * case["pair_capacity"] + 3
        or case["contact_groups"] * 64 >= 2**31
        or case["pair_capacity"]
        < case["colliders"] * (case["colliders"] - 1) // 2 * case["batches"]
    ):
        raise ValueError("Analytic collision requires space for every candidate pair")
    sizes = {n: case["buffer_words"][n] for n in broad.BUFFERS}
    sizes.update(
        Parents=case["storage"],
        Materials=case["storage"] * layout["material_words"],
        Contacts=case["contact_storage"] * layout["contact_words"],
        PfmCount=3,
        Plan=2 * layout["plan_words"],
        PfmSortLen=3,
        ContactsIndirect=6,
        PfmIndirect=6,
        MbIndirect=6,
    )
    if (
        set(arrays) != set(BUFFERS)
        or case["buffer_words"] != sizes
        or any(size >= 2**31 for size in sizes.values())
        or any(
            v.dtype != np.uint32 or v.shape != (sizes[n],) for n, v in arrays.items()
        )
    ):
        raise ValueError("Invalid analytic collision buffers")
    broad.validate_case(
        layout,
        subset(case, broad.BUFFERS, kind="broadphase_pairs"),
        select(arrays, broad.BUFFERS),
    )
    n = case["count"]
    types = layout["shape_types"]
    shapes = arrays["Shapes"].reshape(-1, layout["shape_words"])[
        :n, layout["shape"]["a"] + 3
    ]
    if np.any((shapes != types["ball"]) & (shapes != types["cuboid"])):
        raise ValueError(
            "This complete analytic scene path currently supports balls and cuboids"
        )
    if not np.array_equal(
        arrays["Parents"][:n], arrays["Filters"].reshape(-1, 2)[:n, 0]
    ):
        raise ValueError("Collision filters and collider parent IDs disagree")
    m = layout["material"]
    materials = (
        arrays["Materials"].view(np.float32).reshape(-1, layout["material_words"])[:n]
    )
    if not np.isfinite(materials[:, [m["friction"], m["restitution"]]]).all():
        raise ValueError("Nonfinite collider materials")
    plan.validate_case(
        layout,
        subset(case, plan.BUFFERS, kind="contact_plan", count=1, groups=1, threads=1),
        select(arrays, plan.BUFFERS),
    )


def check_outputs(layout, case, before, actual, expected, contract):
    import numpy as np

    validate_case(layout, case, before)
    if set(actual) != set(before) or set(expected) != set(before):
        raise ValueError("Missing collision program outputs")
    for result in (actual, expected):
        if any(
            a.dtype != np.uint32 or a.shape != before[n].shape
            for n, a in result.items()
        ):
            raise ValueError("Invalid collision program output shape")
    bc = subset(case, broad.BUFFERS, kind="broadphase_pairs")
    errors = broad.check_outputs(
        layout,
        bc,
        select(before, broad.BUFFERS),
        select(actual, broad.BUFFERS),
        select(expected, broad.BUFFERS),
        contract,
    )
    # Validate publication independently from the native fixture, including reset
    # and all untouched words. Pair allocation order cannot change these totals.
    pb = {n: v.copy() for n, v in select(before, plan.BUFFERS).items()}
    pb["PairCount"] = actual["PairCount"]
    pb["PfmCount"][0] = 0
    pc = subset(case, plan.BUFFERS, kind="contact_plan", count=1, groups=1, threads=1)
    errors.update(
        plan.check_outputs(
            layout,
            pc,
            pb,
            select(actual, plan.BUFFERS),
            select(expected, plan.BUFFERS),
            dict(outputs=["Plan"]),
        )
    )
    count = int(actual["Plan"][layout["plan"]["pfm_base"]])
    nc = subset(
        case,
        narrow.BUFFERS,
        kind="analytic_contacts",
        count=count,
        storage=case["contact_storage"],
        colliders=case["storage"],
        groups=case["contact_groups"],
    )
    pw, off, cw = (
        layout["pair_words"],
        layout["pair_colliders"],
        layout["contact_words"],
    )
    expected_pairs = expected["Pairs"].reshape(-1, pw)[:count, off : off + 2]
    index = {tuple(pair): i for i, pair in enumerate(expected_pairs)}
    if len(index) != count:
        raise AssertionError("Duplicate native expected pair")
    actual_pairs = actual["Pairs"].reshape(-1, pw)[:count, off : off + 2]
    reordered = before["Contacts"].copy().reshape(-1, cw)
    ref = expected["Contacts"].reshape(-1, cw)
    for i, pair in enumerate(actual_pairs):
        reordered[i] = ref[index[tuple(pair)]]
    nb = select(before, narrow.BUFFERS)
    na = select(actual, narrow.BUFFERS)
    ne = select(expected, narrow.BUFFERS)
    ne["Contacts"] = reordered.ravel()
    # Reuse the isolated analytic checker after normalizing its equal-sized
    # pair/contact storage ABI. Only the validator pads; execution uses native buffers.
    pairs = np.zeros(nc["storage"] * pw, dtype=np.uint32)
    pairs[: len(actual["Pairs"])] = actual["Pairs"]
    for data in (nb, na, ne):
        data.update(Pairs=pairs, Plan=actual["Plan"][: layout["plan_words"]])
    nc["buffer_words"]["Pairs"] = len(pairs)
    nc["buffer_words"]["Plan"] = layout["plan_words"]
    errors.update(narrow.check_outputs(layout, nc, nb, na, ne, contract))
    return errors


def main():
    from stage_replay import parse_args, run_stage
    from collision_kernels import make_program

    run_stage(
        parse_args(["analytic_collision"], "analytic_collision"),
        oracle="collision",
        buffers=BUFFERS,
        make_kernel=None,
        make_program=make_program,
        validate_case=validate_case,
        check_outputs=check_outputs,
        launch=lambda c: (c["groups"], 64, 1),
    )


if __name__ == "__main__":
    main()
