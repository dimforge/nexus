#!/usr/bin/env python3
"""Live Rust adjacency parity; atomic ID order is compared per body as a multiset."""

BUFFERS = (
    "Contacts",
    "Counts",
    "Ids",
    "WorldMass",
    "IsMultibody",
    "Plan",
    *[f"Aux{i}" for i in range(4)],
    *[f"Stride{i}" for i in range(4)],
)
OPERATIONS = ("adjacency_count", "adjacency_scatter", "adjacency_build")


def validate_case(layout, case, arrays):
    import numpy as np

    n, bodies = case["count"], case["bodies"]
    if (
        case["kind"] not in OPERATIONS
        or case["iterations"] != 1
        or type(n) is not int
        or not 0 <= n < (1 << 29)
        or type(bodies) is not int
        or not 0 <= bodies < (1 << 29)
        or not 1 <= case["groups"] <= 65535
        or case["batches"] < 1
        or bodies % case["batches"]
        or case["body_storage"] != bodies + 3
        or case["contact_storage"] != n + 3
    ):
        raise ValueError("Invalid adjacency dimensions")
    strides = [bodies]
    for _ in range(4):
        strides.append((strides[-1] + 255) // 256)
    levels = next(i for i in range(1, 5) if strides[i] <= 1)
    if case["strides"] != strides[:4] or case["levels"] != levels:
        raise ValueError("Invalid adjacency scan hierarchy")
    sizes = {
        "Contacts": (n + 3) * layout["contact_words"],
        "Counts": bodies + 3,
        "Ids": 2 * n + 3,
        "WorldMass": (bodies + 3) * layout["world_mass_words"],
        "IsMultibody": bodies + 3,
        "Plan": layout["plan_words"],
    }
    for i in range(4):
        sizes[f"Aux{i}"] = strides[i + 1] + 3
        sizes[f"Stride{i}"] = 2
    if case["buffer_words"] != sizes or set(arrays) != set(sizes):
        raise ValueError("Invalid adjacency buffer layout")
    for name, size in sizes.items():
        if (
            arrays[name].dtype != np.uint32
            or arrays[name].ndim != 1
            or arrays[name].size != size
        ):
            raise ValueError(f"Invalid {name} storage")
    if int(arrays["Plan"][layout["plan"]["bound"]]) != n:
        raise ValueError("Plan bound differs from specialization")
    for i in range(4):
        if int(arrays[f"Stride{i}"][0]) != strides[i]:
            raise ValueError("Invalid adjacency stride uniform")
    mass = (
        arrays["WorldMass"]
        .reshape(bodies + 3, layout["world_mass_words"])
        .view(np.float32)
    )
    offset = layout["world_mass"]["inv_mass"]
    if not np.isfinite(mass[:bodies, offset : offset + 3]).all():
        raise ValueError("Non-finite inverse mass")
    dynamic = np.any(mass[:bodies, offset : offset + 3] != 0, axis=1)
    contacts = arrays["Contacts"].reshape(n + 3, layout["contact_words"])
    length = layout["contact"]["contact"] + layout["manifold"]["len"]
    owner = layout["contact"]["bodies"]
    rows = [[] for _ in range(bodies)]
    for i in range(n):
        count = int(contacts[i, length])
        if count > 4:
            raise ValueError("Invalid contact point count")
        if count == 0:
            continue
        a, b = map(int, contacts[i, owner : owner + 2])
        if a >= bodies or b >= bodies:
            raise ValueError("Out-of-range active contact body")
        if arrays["IsMultibody"][a] or arrays["IsMultibody"][b]:
            continue
        for body in (a, b):
            if dynamic[body]:
                rows[body].append(i)
    counts = np.array([len(row) for row in rows], dtype=np.uint32)
    ends = np.cumsum(counts, dtype=np.uint64)
    starts = ends - counts
    initial = (
        starts
        if case["kind"] == "adjacency_scatter"
        else np.zeros(bodies, dtype=np.uint32)
    )
    if not np.array_equal(arrays["Counts"][:bodies], initial):
        raise ValueError(
            "Counts must contain reset counts or exact exclusive row starts"
        )
    masks = {name: np.zeros(size, dtype=bool) for name, size in sizes.items()}
    masks["Counts"][:bodies] = True
    if case["kind"] != "adjacency_count":
        masks["Ids"][: int(ends[-1]) if bodies else 0] = True
    if bodies and case["kind"] == "adjacency_build":
        for i in range(levels):
            masks[f"Aux{i}"][: strides[i + 1]] = True
    return masks, rows, counts, ends


def check_outputs(layout, case, before, actual, expected, contract):
    import numpy as np

    masks, rows, counts, ends = validate_case(layout, case, before)
    for name, mask in masks.items():
        if actual[name].shape != before[name].shape or actual[name].dtype != np.uint32:
            raise ValueError(f"Invalid returned {name}")
        np.testing.assert_array_equal(
            actual[name][~mask],
            before[name][~mask],
            err_msg=f"Unexpected write to {name}",
        )
        if name != "Ids" or case["kind"] == "adjacency_count":
            np.testing.assert_array_equal(
                actual[name],
                expected[name],
                err_msg=f"Rust adjacency mismatch in {name}",
            )
    wanted = counts if case["kind"] == "adjacency_count" else ends
    np.testing.assert_array_equal(actual["Counts"][: case["bodies"]], wanted)
    if case["kind"] != "adjacency_count":
        start = 0
        for body, row in enumerate(rows):
            end = int(ends[body])
            # Atomics may reorder IDs but cannot drop, duplicate, or move one
            # between bodies. Repeated self-contact sides remain repeated IDs.
            for values in (actual["Ids"], expected["Ids"]):
                np.testing.assert_array_equal(
                    np.sort(values[start:end]),
                    sorted(row),
                    err_msg=f"Adjacency membership for body {body}",
                )
            start = end
    return {n: 0.0 for n in BUFFERS}


def main():
    from graph_kernels import make_program
    from stage_replay import parse_args, run_stage

    args = parse_args(OPERATIONS, "adjacency_build")
    run_stage(
        args,
        oracle="graphs",
        buffers=BUFFERS,
        make_kernel=None,
        launch=None,
        make_program=make_program,
        validate_case=validate_case,
        check_outputs=check_outputs,
    )


if __name__ == "__main__":
    main()
