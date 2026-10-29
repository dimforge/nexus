#!/usr/bin/env python3
"""Rust graph-coloring parity with an explicit model for nondeterministic picks."""

BUFFERS = (
    "Counts",
    "Ids",
    "Links",
    "Colors",
    "Rands",
    "Colored",
    "Pending",
    "NumColors",
    "BodyGroup",
    "Plan",
    "BatchIds",
    "ContactsIndirect",
    "ColoringIndirect",
    "CurrentColor",
)
OPERATIONS = (
    "luby_reset",
    "luby_step",
    "topo_reset",
    "topo_clear",
    "topo_completion",
    "topo_step",
    "topo_fix",
)


def validate_case(layout, case, arrays):
    import numpy as np

    n, storage, bodies = case["count"], case["storage"], case["bodies"]
    op = case["kind"]
    scalar = op in ("topo_clear", "topo_completion")
    if (
        op not in OPERATIONS
        or type(n) is not int
        or not 0 <= n < 1 << 20
        or storage != n + 3
        or bodies != n + 4
        or case["iterations"] != 1
        or not 1 <= case["groups"] <= 65535
        or case["threads"] != (1 if scalar else 64)
    ):
        raise ValueError("Invalid coloring dimensions")
    if op in ("luby_reset", "topo_reset") and case["groups"] * 64 < n:
        raise ValueError("Reset grid does not cover contacts")
    sizes = {
        "Counts": bodies,
        "Ids": case["active_ids"] + 3,
        "Links": storage * layout["link_words"],
        "Colors": storage,
        "Rands": storage,
        "Colored": storage,
        "Pending": storage,
        "NumColors": 2,
        "BodyGroup": bodies,
        "Plan": layout["plan_words"],
        "BatchIds": layout["batch_words"],
        "ContactsIndirect": 9,
        "ColoringIndirect": 9,
        "CurrentColor": 2,
    }
    if case["buffer_words"] != sizes or set(arrays) != set(sizes):
        raise ValueError("Invalid coloring layout")
    for name, size in sizes.items():
        a = arrays[name]
        if a.dtype != np.uint32 or a.ndim != 1 or a.size != size:
            raise ValueError(f"Invalid {name} storage")
    if (
        int(arrays["Plan"][layout["plan"]["bound"]]) != n
        or int(arrays["BatchIds"][layout["batch"]["deterministic"]])
        != case["deterministic"]
    ):
        raise ValueError("Coloring uniform differs from specialization")
    if np.any(arrays["BodyGroup"] >= bodies):
        raise ValueError("Out-of-range body group")
    ends = arrays["Counts"].astype(np.uint64)
    if (
        np.any(ends[1:] < ends[:-1])
        or int(ends[-1]) != case["active_ids"]
        or case["active_ids"] > 2 * n
    ):
        raise ValueError("Invalid adjacency ends")
    links = arrays["Links"].reshape(-1, layout["link_words"])
    fields = layout["link"]
    static = case["static_groups"]
    if (
        not isinstance(static, list)
        or len(set(static)) != len(static)
        or any(type(g) is not int or not 0 <= g < bodies for g in static)
    ):
        raise ValueError("Invalid static-group metadata")
    rows = [[] for _ in range(bodies)]
    neighbors = [[] for _ in range(n)]
    active = []
    for i in range(n):
        length = int(links[i, fields["len"]])
        if length > 4:
            raise ValueError("Invalid link length")
        if length == 0:
            continue
        active.append(i)
        for side in ("a", "b"):
            body = int(links[i, fields[f"solver_body_{side}"]])
            if body >= bodies:
                raise ValueError("Out-of-range active coloring body")
            group = int(arrays["BodyGroup"][body])
            if group not in case["static_groups"]:
                rows[group].append(i)
    start = 0
    for group, row in enumerate(rows):
        end = int(ends[group])
        ids = list(map(int, arrays["Ids"][start:end]))
        if sorted(ids) != sorted(row):
            raise ValueError("Incomplete or invalid coloring adjacency")
        start = end
    for i in active:
        for side in ("a", "b"):
            body = int(links[i, fields[f"solver_body_{side}"]])
            group = int(arrays["BodyGroup"][body])
            if (
                op.startswith("topo")
                and int(links[i, fields[f"vel_slot_{side}"]]) != body
            ):
                continue
            neighbors[i].extend(j for j in rows[group] if j != i)
    active_mask = np.zeros(storage, dtype=bool)
    active_mask[active] = True
    if op == "luby_step":
        current = int(arrays["CurrentColor"][0])
        if (
            not 1 <= current < 0xFFFFFFFF
            or int(arrays["NumColors"][0]) + n > 0xFFFFFFFF
        ):
            raise ValueError("Invalid Luby color/counter")
        for i in range(n):
            color = int(arrays["Colors"][i])
            if (i not in active and color != 0) or (
                i in active and color != 0xFFFFFFFF and not 0 < color < current
            ):
                raise ValueError("Invalid prior Luby color")
    if op in ("topo_step", "topo_fix"):
        if np.any(arrays["Colors"][:n] >= 64) or np.any(arrays["Colored"][:n] > 1):
            raise ValueError("Invalid Topo color or flag")
        for i in range(n):
            if not active_mask[i] and (
                arrays["Colored"][i] != 1 or arrays["Colors"][i] != 0
            ):
                raise ValueError("Inactive Topo slot must be precolored zero")
        if case["deterministic"] and op == "topo_fix":
            pending = arrays["Pending"][:n]
            if np.any((pending >= 64) & (pending != 0xFFFFFFFF)):
                raise ValueError("Invalid pending color")
    masks = {name: np.zeros(size, dtype=bool) for name, size in sizes.items()}
    if op == "luby_reset":
        masks["Colors"][:n] = True
        masks["Rands"][:n] = True
    if op == "luby_step":
        masks["Colors"][:n] = arrays["Colors"][:n] == 0xFFFFFFFF
        masks["NumColors"][0] = True
    if op == "topo_reset":
        for name in ("Colors", "Colored", "Pending"):
            masks[name][:n] = True
        masks["NumColors"][0] = True
    if scalar:
        masks["NumColors"][0] = True
        masks["ColoringIndirect"][:3] = True
    if op == "topo_step":
        masks["NumColors"][0] = True
        if case["deterministic"]:
            masks["Pending"][:n] = True
        else:
            masks["Colors"][:n] = arrays["Colored"][:n] == 0
            masks["Colored"][:n] = arrays["Colored"][:n] == 0
    if op == "topo_fix":
        masks["Colored"][:n] = active_mask[:n]
        if case["deterministic"]:
            masks["Colors"][:n] = active_mask[:n]
    return masks, neighbors, active


def possible_first_free(candidate, choices):
    """A legal read observes an unchanged neighbor or its before/after new pick.

    A bipartite matching assigns each required lower color to a distinct neighbor
    read. Remaining reads must have an option other than the chosen free color.
    This avoids exponential enumeration and does not relax integer identities.
    """
    options = [set(values) - {candidate} for values in choices]
    if any(not values for values in options):
        return False
    matched = {}

    def assign(color, seen):
        for read, values in enumerate(options):
            if read not in seen and color in values:
                seen.add(read)
                if read not in matched or assign(matched[read], seen):
                    matched[read] = color
                    return True
        return False

    return all(assign(color, set()) for color in range(1, candidate))


def check_outputs(layout, case, before, actual, expected, contract):
    import numpy as np

    masks, neighbors, active = validate_case(layout, case, before)
    nondeterministic = case["kind"] == "topo_step" and not case["deterministic"]
    for name, mask in masks.items():
        if actual[name].shape != before[name].shape or actual[name].dtype != np.uint32:
            raise ValueError(f"Invalid returned {name}")
        np.testing.assert_array_equal(
            actual[name][~mask],
            before[name][~mask],
            err_msg=f"Unexpected write to {name}",
        )
        if not (nondeterministic and name == "Colors"):
            np.testing.assert_array_equal(
                actual[name],
                expected[name],
                err_msg=f"Rust coloring mismatch in {name}",
            )
    if nondeterministic:
        for result in (actual, expected):
            for i in active:
                if before["Colored"][i] != 0:
                    continue
                color = int(result["Colors"][i])
                if not 1 <= color <= 63:
                    raise AssertionError("Topo pick outside supported palette")
                choices = []
                for j in neighbors[i]:
                    values = {int(before["Colors"][j])}
                    if before["Colored"][j] == 0:
                        values.add(int(result["Colors"][j]))
                    choices.append(values)
                if not possible_first_free(color, choices):
                    raise AssertionError(f"No legal first-free witness for contact {i}")
    if case["kind"] == "luby_reset":
        for i in range(case["count"]):
            key = (i * 0xCC9E2D51) & 0xFFFFFFFF
            rotated = ((key << 15) | (key >> 17)) & 0xFFFFFFFF
            if int(actual["Rands"][i]) != (rotated * 0x1B873593) & 0xFFFFFFFF:
                raise AssertionError("Luby hash differs from modulo-2^32 definition")
    if case["kind"] == "luby_step":
        current = before["CurrentColor"][0]
        losers = 0
        for i in active:
            if before["Colors"][i] != 0xFFFFFFFF:
                continue
            priority = (int(before["Rands"][i]), i)
            wins = all(
                before["Colors"][j] != 0xFFFFFFFF
                or priority > (int(before["Rands"][j]), j)
                for j in neighbors[i]
            )
            wanted = current if wins else 0xFFFFFFFF
            if actual["Colors"][i] != wanted:
                raise AssertionError(
                    "Luby winner differs from immutable round priorities"
                )
            losers += not wins
        if int(actual["NumColors"][0]) != int(before["NumColors"][0]) + losers:
            raise AssertionError("Incorrect Luby uncolored count")
        for i in active:
            if actual["Colors"][i] == current and any(
                actual["Colors"][j] == current for j in neighbors[i]
            ):
                raise AssertionError("Adjacent Luby winners share a color")
    return {name: 0.0 for name in BUFFERS}


def main():
    from coloring_kernels import make_kernel
    from stage_replay import parse_args, run_stage

    args = parse_args(OPERATIONS, "topo_step")
    run_stage(
        args,
        oracle="colorings",
        buffers=BUFFERS,
        make_kernel=make_kernel,
        validate_case=validate_case,
        check_outputs=check_outputs,
        launch=lambda case: (case["groups"], case["threads"], case["iterations"]),
    )


if __name__ == "__main__":
    main()
