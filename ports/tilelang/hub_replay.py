#!/usr/bin/env python3
"""Live Rust hub replay with bounded graphs and legal atomic-allocation comparison."""

from stage_replay import parse_args, run_stage

BUFFERS = (
    "Counts",
    "Ids",
    "WorldMass",
    "BodyGroup",
    "FirstSlots",
    "HubList",
    "HubCounts",
    "Links",
    "SlotBody",
    "SlotConstraint",
    "Velocities",
    "SlotsIndirect",
    "AverageIndirect",
    "BatchIds",
)
OPERATIONS = ("hub_assign", "hub_split", "hub_dispatch", "hub_scatter", "hub_average")
OUTPUTS = {
    "hub_assign": ("FirstSlots", "HubList", "HubCounts"),
    "hub_split": ("Links", "SlotBody", "SlotConstraint"),
    "hub_dispatch": ("SlotsIndirect", "AverageIndirect"),
    "hub_scatter": ("Velocities",),
    "hub_average": ("Velocities",),
}


def validate_case(layout, case, arrays):
    import numpy as np
    from contact_replay import validate_layout

    validate_layout(layout)
    if (
        layout["hub_min_constraints"] != 32
        or layout["hub_average_workgroups"] != 256
        or layout["not_a_hub"] != 2**32 - 1
    ):
        raise ValueError("Unsupported hub scheduling constants")
    for key in ("hub", "batch"):
        stride, offsets = layout[key + "_words"], list(layout[key].values())
        if (
            type(stride) is not int
            or not 0 < stride < 2**20
            or len(set(offsets)) != len(offsets)
            or any(type(o) is not int or not 0 <= o < stride for o in offsets)
        ):
            raise ValueError("Invalid hub ABI")
    keys = (
        "count",
        "storage",
        "constraints",
        "constraint_count",
        "ids_storage",
        "hub_storage",
        "slot_storage",
        "velocities",
        "groups",
        "threads",
        "iterations",
    )
    if any(type(case[k]) is not int or not 0 <= case[k] < 2**31 for k in keys):
        raise ValueError("Invalid hub dimensions")
    if (
        case["kind"] not in OPERATIONS
        or not 1 <= case["iterations"] <= 1024
        or not 1 <= case["groups"] <= 65535
        or case["threads"] != (1 if case["kind"] == "hub_dispatch" else 64)
        or any(case[k] < 1 for k in keys if k not in ("count", "constraint_count"))
        or case["count"] > case["storage"]
        or case["constraint_count"] > case["constraints"]
    ):
        raise ValueError("Invalid hub dimensions")
    if case["kind"] == "hub_dispatch" and case["groups"] != 1:
        raise ValueError("Dispatch sizing requires one invocation")
    sizes = {
        "Counts": case["storage"],
        "Ids": case["ids_storage"],
        "WorldMass": case["storage"] * layout["world_mass_words"],
        "BodyGroup": case["storage"],
        "FirstSlots": case["storage"],
        "HubList": case["hub_storage"],
        "HubCounts": layout["hub_words"],
        "Links": case["constraints"] * layout["link_words"],
        "SlotBody": case["slot_storage"],
        "SlotConstraint": case["slot_storage"],
        "Velocities": case["velocities"] * layout["velocity_words"],
        "SlotsIndirect": 9,
        "AverageIndirect": 9,
        "BatchIds": layout["batch_words"],
    }
    if set(arrays) != set(BUFFERS) or any(
        a.dtype != np.uint32 or a.shape != (sizes[k],) or a.size >= 2**31
        for k, a in arrays.items()
    ):
        raise ValueError("Invalid hub buffer sizes")
    batch = {k: int(arrays["BatchIds"][v]) for k, v in layout["batch"].items()}
    hc = {k: int(arrays["HubCounts"][v]) for k, v in layout["hub"].items()}
    if (
        any(v >= 2**31 for v in (*batch.values(), *hc.values()))
        or batch["num_batches"] * batch["colliders_batch_capacity"] != case["count"]
        or batch["bodies_len"] > batch["colliders_batch_capacity"]
        or hc["base"] < case["count"]
        or hc["base"] + hc["pool"] > case["velocities"]
        or hc["pool"] > case["slot_storage"]
        or hc["pool"] // 32 + 1 > case["hub_storage"]
    ):
        raise ValueError("Invalid hub uniforms")
    counts = arrays["Counts"].astype(np.uint64)
    if (
        np.any(counts[1:] < counts[:-1])
        or counts[-1] > case["ids_storage"]
        or (case["count"] == 0 and counts[-1] != 0)
        or (
            case["count"] > 0
            and np.any(counts[case["count"] :] != counts[case["count"] - 1])
        )
    ):
        raise ValueError("Invalid hub adjacency counts")
    starts = np.concatenate((np.array([0], dtype=np.uint64), counts[:-1]))
    degrees = counts - starts
    links = arrays["Links"].reshape(-1, layout["link_words"])
    fields = layout["link"]
    seen = [0] * case["constraint_count"]
    for body in range(case["count"]):
        row = arrays["Ids"][int(starts[body]) : int(counts[body])]
        if len(set(map(int, row))) != len(row):
            raise ValueError("Duplicate hub adjacency entry")
        for cid in row:
            cid = int(cid)
            if cid >= case["constraint_count"]:
                raise ValueError("Invalid hub constraint ID")
            a, b = (
                int(links[cid, fields[k]]) for k in ("solver_body_a", "solver_body_b")
            )
            if a == b or a >= case["count"] or b >= case["count"] or body not in (a, b):
                raise ValueError("Invalid hub constraint owner")
            seen[cid] += 1
    if any(n != 2 for n in seen):
        raise ValueError("Incomplete hub adjacency")
    mass = arrays["WorldMass"].view(np.float32).reshape(-1, layout["world_mass_words"])
    im = mass[
        :, layout["world_mass"]["inv_mass"] : layout["world_mass"]["inv_mass"] + 3
    ]
    if not np.isfinite(im).all() or (im < 0).any():
        raise ValueError("Invalid hub inverse mass")
    candidates = {
        b: int(degrees[b])
        for b in range(case["count"])
        if degrees[b] > 32 and arrays["BodyGroup"][b] == b and np.any(im[b] != 0)
    }
    if sum(candidates.values()) >= 2**31:
        raise ValueError("Hub reservations can overflow")
    vels = arrays["Velocities"].view(np.float32).reshape(-1, layout["velocity_words"])
    for offset in (layout["velocity_linear"], layout["velocity_angular"]):
        if not np.isfinite(vels[:, offset : offset + 3]).all():
            raise ValueError("Non-finite hub velocities")
    if case["kind"] == "hub_assign":
        if (
            hc["slots"]
            or hc["hubs"]
            or case["iterations"] != 1
            or np.any(arrays["FirstSlots"] != np.uint32(2**32 - 1))
        ):
            raise ValueError(
                "Hub assignment must start with reset counters and first slots"
            )
        selected = []
    elif case["kind"] == "hub_dispatch":
        selected = []
    else:
        selected = list(
            map(int, arrays["HubList"][: min(hc["hubs"], hc["pool"] // 32 + 1)])
        )
        if len(set(selected)) != len(selected) or any(
            b not in candidates for b in selected
        ):
            raise ValueError("Invalid or repeated hub list body")
        ranges = sorted(
            (int(arrays["FirstSlots"][b]), int(degrees[b])) for b in selected
        )
        end = 0
        for start, length in ranges:
            if start != end or start + length > min(hc["slots"], hc["pool"]):
                raise ValueError("Invalid or overlapping hub slot ranges")
            end = start + length
        if any(
            arrays["FirstSlots"][b] != np.uint32(2**32 - 1)
            for b in range(case["storage"])
            if b not in selected
        ):
            raise ValueError("Unlisted body marked as hub")
        if case["kind"] == "hub_scatter":
            live = min(hc["slots"], hc["pool"])
            if live > case["groups"] * 64 or np.any(
                arrays["SlotBody"][:live] >= hc["base"]
            ):
                raise ValueError("Invalid hub scatter source or dispatch")
            # Split-side bookkeeping must agree for every accepted hub. Rust may
            # also scatter the unused pool tail after an overflowing reservation.
            for body in selected:
                start = int(arrays["FirstSlots"][body])
                degree = int(degrees[body])
                if np.any(arrays["SlotBody"][start : start + degree] != body):
                    raise ValueError("Hub scatter ownership mismatch")
    return {
        "candidates": candidates,
        "selected": selected,
        "degrees": degrees,
        "counts": hc,
    }


def check_assignment(layout, case, before, actual, context):
    import numpy as np

    hc = {k: int(actual["HubCounts"][v]) for k, v in layout["hub"].items()}
    candidates, initial = context["candidates"], context["counts"]
    assert hc["base"] == initial["base"] and hc["pool"] == initial["pool"], (
        "Hub capacity changed"
    )
    assert hc["slots"] == sum(candidates.values()), "Incorrect atomic slot total"
    assert hc["hubs"] <= min(len(candidates), hc["pool"] // 32 + 1), (
        "Incorrect atomic hub total"
    )
    selected = list(map(int, actual["HubList"][: hc["hubs"]]))
    assert len(set(selected)) == len(selected) and all(
        b in candidates for b in selected
    ), "Invalid allocated hub"
    np.testing.assert_array_equal(
        actual["HubList"][hc["hubs"] :], before["HubList"][hc["hubs"] :]
    )
    selected_set = set(selected)
    end = 0
    for start, n in sorted(
        (int(actual["FirstSlots"][b]), candidates[b]) for b in selected
    ):
        assert start == end and start + n <= hc["pool"], (
            "Overlapping or invalid atomic reservation"
        )
        end = start + n
    for body in range(case["storage"]):
        if body not in selected_set:
            assert actual["FirstSlots"][body] == before["FirstSlots"][body], (
                "Unallocated first slot changed"
            )
    remaining = [n for b, n in candidates.items() if b not in selected_set]
    # Successful reservations must precede the first overflow. The rest can
    # occur in any atomic order; hub-list order is independent of slot order.
    assert not remaining or any(n > hc["pool"] - end for n in remaining), (
        "Rejected hubs still fit without an overflow"
    )


def check_outputs(layout, case, before, actual, expected, contract):
    import numpy as np

    context = validate_case(layout, case, before)
    if (
        set(actual) != set(BUFFERS)
        or set(expected) != set(BUFFERS)
        or any(
            actual[k].dtype != np.uint32
            or actual[k].shape != before[k].shape
            or expected[k].shape != before[k].shape
            for k in BUFFERS
        )
    ):
        raise AssertionError("Invalid hub output buffers")
    result = {}
    for name in BUFFERS:
        result[name] = 0.0
        if case["kind"] == "hub_assign" and name in OUTPUTS["hub_assign"]:
            continue
        if case["kind"] == "hub_average" and name == "Velocities":
            mask = np.zeros(before[name].size, dtype=bool)
            for body in context["selected"]:
                for field in ("velocity_linear", "velocity_angular"):
                    start = body * layout["velocity_words"] + layout[field]
                    mask[start : start + 3] = True
            np.testing.assert_array_equal(actual[name][~mask], before[name][~mask])
            a, e = (
                actual[name].view(np.float32)[mask],
                expected[name].view(np.float32)[mask],
            )
            assert np.isfinite(a).all(), "Non-finite averaged velocity"
            np.testing.assert_allclose(
                a, e, atol=contract["atol"], rtol=contract["rtol"]
            )
            result[name] = float(np.max(np.abs(a - e))) if len(a) else 0.0
        else:
            np.testing.assert_array_equal(actual[name], expected[name])
    if case["kind"] == "hub_assign":
        check_assignment(layout, case, before, expected, context)
        check_assignment(layout, case, before, actual, context)
    return result


def main():
    from hub_kernels import make_kernel

    args = parse_args(OPERATIONS, "hub_assign")
    run_stage(
        args,
        oracle="hubs",
        buffers=BUFFERS,
        make_kernel=make_kernel,
        validate_case=validate_case,
        check_outputs=check_outputs,
        launch=lambda c: (c["groups"], c["threads"], c["iterations"]),
    )


if __name__ == "__main__":
    main()
