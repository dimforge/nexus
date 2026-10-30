"""Native initial storage variants and complete resident retry scene validation."""

import itertools

import numpy as np
from lbvh_scene_validation import check_outputs as check_scene
from lbvh_scene_validation import validate_case


def validate_variants(l, case, before, variants):
    validate_case(l, case, before)
    ladder = case["capacity_ladder"]
    if (
        len(ladder) != len(variants)
        or len(ladder) < 2
        or any(type(v) is not int or v < 1 for v in ladder)
        or any(a >= b for a, b in itertools.pairwise(ladder))
        or ladder[-1] != case["pair_capacity"]
    ):
        raise ValueError("Invalid capacity ladder")
    growth = case["expected_growth_frames"]
    if (
        not growth
        or growth != sorted(set(growth))
        or any(type(v) is not int or not 1 <= v <= case["frames"] for v in growth)
    ):
        raise ValueError("Invalid required growth frames")
    special = {
        "pair_capacity",
        "pair_storage",
        "contact_capacity",
        "contact_storage",
        "count",
        "new_base",
        "buffer_words",
        "name",
        "capacity_ladder",
        "expected_growth_frames",
        "initial_upload_only",
    }
    for cap, (c, b) in zip(ladder, variants):
        if (
            c.get("initial_upload_only") is not True
            or c["pair_capacity"] != cap
            or {n: v for n, v in c.items() if n not in special}
            != {n: v for n, v in case.items() if n not in special}
        ):
            raise ValueError("Capacity variant changed the scene specialization")
        validate_case(l, c, b, capacity_variant=True)
        for name in b:
            if b[name].shape != before[name].shape:
                continue
            mask = np.ones(b[name].size, bool)
            if name == "BatchIds":
                mask[
                    [
                        l["batch"][n]
                        for n in ("collision_pairs_capacity", "contacts_capacity")
                    ]
                ] = False
            elif name in ("Config", "InitConfig", *(f"Config{i}" for i in range(8))):
                mask[l["sort_uniforms"]["max_keys_per_batch"]] = False
            np.testing.assert_array_equal(
                b[name][mask],
                before[name][mask],
                err_msg=f"Variant initial scene {name}",
            )
    for n in before:
        np.testing.assert_array_equal(
            variants[-1][1][n], before[n], err_msg=f"Final variant native upload {n}"
        )


def check_stats(case, stats):
    if len(stats) != case["frames"]:
        raise AssertionError("Incomplete resident frame stats")
    previous = 0
    growth = []
    for i, row in enumerate(stats, 1):
        variant, retries = row["variant"], row["retries"]
        if (
            type(variant) is not int
            or not previous <= variant < len(case["capacity_ladder"])
            or type(retries) is not int
            or not int(variant != previous) <= retries <= variant - previous
            or type(row["step"]) is not int
            or row["step"] != i
        ):
            raise AssertionError("Invalid resident commit/retry sequence")
        if retries:
            growth.append(i)
        cap = case["capacity_ladder"][variant]
        expected = (
            {"pairs", "deferred_pairs"}
            if case["kind"] == "general_scene"
            else {"pairs"}
        )
        if set(row["counts"]) != expected or any(
            type(v) is not int or not 0 <= v <= cap for v in row["counts"].values()
        ):
            raise AssertionError("Committed overflowing resident frame")
        if retries and max(row["counts"].values()) <= case["capacity_ladder"][previous]:
            raise AssertionError("Growth occurred without capacity pressure")
        previous = variant
    if (
        growth != case["expected_growth_frames"]
        or previous != len(case["capacity_ladder"]) - 1
    ):
        raise AssertionError("Required capacity growth was not executed")


def check_outputs(l, c, b, a, e, k, stats):
    check_stats(c, stats)
    # Final storage has the native full-capacity shape. Trace equality proves that
    # rejected attempts did not integrate a frame twice or lose past state.
    return check_scene(l, c, b, a, e, k)
