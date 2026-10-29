#!/usr/bin/env python3
"""Exact bucket counts/dispatch and semantic copied-link permutation checks."""

BUFFERS = (
    "Colors",
    "Links",
    "Plan",
    "Buckets",
    "BatchIds",
    "SortedLinks",
    "Indices",
    "Stats",
    "Indirect",
    *[f"Aux{i}" for i in range(4)],
    *[f"Stride{i}" for i in range(4)],
)
OPERATIONS = (
    "bucket_reset",
    "bucket_count",
    "bucket_scatter",
    "bucket_dispatch",
    "bucket_build",
)


def validate_case(layout, case, arrays):
    import numpy as np

    n, nb, stride = case["count"], case["batches"], case["color_stride"]
    if (
        case["kind"] not in OPERATIONS
        or case["iterations"] != 1
        or type(n) is not int
        or not 0 <= n < 1 << 28
        or type(nb) is not int
        or not 1 <= nb <= 65535
        or type(stride) is not int
        or not 3 <= stride <= 65535
        or nb * stride >= (1 << 30)
        or case["storage"] != n + 3
        or case["bucket_count"] != nb * stride
        or not 1 <= case["groups"] <= 65535
    ):
        raise ValueError("Invalid bucket dimensions")
    strides = [nb * stride]
    for _ in range(4):
        strides.append((strides[-1] + 255) // 256)
    levels = next(i for i in range(1, 5) if strides[i] <= 1)
    if case["levels"] != levels or case["strides"] != strides[:4]:
        raise ValueError("Invalid bucket scan hierarchy")
    sizes = {
        "Colors": n + 3,
        "Links": (n + 3) * layout["link_words"],
        "Plan": layout["plan_words"],
        "Buckets": nb * stride,
        "BatchIds": layout["batch_words"],
        "SortedLinks": (n + 3) * layout["link_words"],
        "Indices": n + 3,
        "Stats": 2 * layout["stats_words"],
        "Indirect": 9,
    }
    for i in range(4):
        sizes[f"Aux{i}"] = strides[i + 1] + 3
        sizes[f"Stride{i}"] = 2
    if case["buffer_words"] != sizes or set(arrays) != set(sizes):
        raise ValueError("Invalid bucket layout")
    for name, size in sizes.items():
        a = arrays[name]
        if a.dtype != np.uint32 or a.ndim != 1 or a.size != size:
            raise ValueError(f"Invalid {name} storage")
    if (
        int(arrays["Plan"][layout["plan"]["bound"]]) != n
        or int(arrays["BatchIds"][layout["batch"]["num_batches"]]) != nb
        or int(arrays["BatchIds"][layout["batch"]["solver_color_buckets_stride"]])
        != stride
    ):
        raise ValueError("Bucket uniform differs from specialization")
    for i in range(4):
        if int(arrays[f"Stride{i}"][0]) != strides[i]:
            raise ValueError("Invalid bucket stride uniform")
    links = arrays["Links"].reshape(-1, layout["link_words"])
    length = layout["link"]["len"]
    owner = layout["link"]["solver_body_a"]
    rows = [[] for _ in range(nb * stride)]
    for i in range(n):
        size = int(links[i, length])
        if size > 4:
            raise ValueError("Invalid link length")
        if not size:
            continue
        body = int(links[i, owner])
        color = int(arrays["Colors"][i])
        if body >= 1 << 31:
            raise ValueError("Invalid active link body")
        bucket = (color if 0 < color < stride - 1 else stride - 1) * nb + body % nb
        rows[bucket].append(i)
    counts = np.array([len(row) for row in rows], dtype=np.uint32)
    ends = np.cumsum(counts, dtype=np.uint64)
    starts = ends - counts
    op = case["kind"]
    if op == "bucket_count" and np.any(arrays["Buckets"]):
        raise ValueError("Bucket counting requires reset counters")
    if op == "bucket_scatter" and not np.array_equal(arrays["Buckets"], starts):
        raise ValueError("Scatter requires exact exclusive bucket starts")
    if op == "bucket_dispatch":
        values = arrays["Buckets"].astype(np.uint64)
        if np.any(values[1:] < values[:-1]) or int(values[-1]) >= 1 << 30:
            raise ValueError("Dispatch requires bounded cumulative bucket ends")
        if not case["clip"] and not np.array_equal(values, ends):
            raise ValueError("Invalid fixture bucket ends")
    if case["clip"] and op != "bucket_dispatch":
        raise ValueError("Synthetic large buckets only apply to dispatch sizing")
    masks = {name: np.zeros(size, dtype=bool) for name, size in sizes.items()}
    if op in ("bucket_reset", "bucket_build"):
        masks["Buckets"][:] = True
    if op in ("bucket_count", "bucket_scatter"):
        masks["Buckets"][counts != 0] = True
    if op in ("bucket_count", "bucket_build"):
        masks["SortedLinks"].reshape(-1, layout["link_words"])[:n, length] = True
    if op in ("bucket_scatter", "bucket_build"):
        masks["SortedLinks"][: int(ends[-1]) * layout["link_words"]] = True
        for row in rows:
            masks["Indices"][row] = True
    if op in ("bucket_dispatch", "bucket_build"):
        masks["Stats"][: layout["stats_words"]] = True
        masks["Indirect"][:3] = True
    if op == "bucket_build":
        for i in range(levels):
            masks[f"Aux{i}"][: strides[i + 1]] = True
    return masks, rows, counts, starts, ends


def check_outputs(layout, case, before, actual, expected, contract):
    import numpy as np

    masks, rows, counts, starts, ends = validate_case(layout, case, before)
    op = case["kind"]
    scatter = op in ("bucket_scatter", "bucket_build")
    for name, mask in masks.items():
        if actual[name].shape != before[name].shape or actual[name].dtype != np.uint32:
            raise ValueError(f"Invalid returned {name}")
        np.testing.assert_array_equal(
            actual[name][~mask],
            before[name][~mask],
            err_msg=f"Unexpected write to {name}",
        )
        if not (scatter and name in ("SortedLinks", "Indices")):
            np.testing.assert_array_equal(
                actual[name], expected[name], err_msg=f"Rust bucket mismatch in {name}"
            )
    if scatter:
        links = before["Links"].reshape(-1, layout["link_words"])
        for result in (actual, expected):
            desired = before["SortedLinks"].copy().reshape(-1, layout["link_words"])
            if op == "bucket_build":
                desired[: case["count"], layout["link"]["len"]] = 0
            for row, start, end in zip(rows, starts, ends):
                slots = result["Indices"][row].astype(np.uint64)
                np.testing.assert_array_equal(
                    np.sort(slots),
                    np.arange(start, end, dtype=np.uint64),
                    err_msg="Bucket membership/allocation",
                )
                for source, destination in zip(row, slots):
                    desired[int(destination)] = links[source]
            np.testing.assert_array_equal(
                result["SortedLinks"],
                desired.ravel(),
                err_msg="Copied contact link bits",
            )
        np.testing.assert_array_equal(actual["Buckets"], ends)
    if op in ("bucket_dispatch", "bucket_build"):
        nb = case["batches"]
        stride = case["color_stride"]
        buckets = actual["Buckets"]
        sizes = [0] + [
            int(buckets[(color + 1) * nb - 1]) - int(buckets[color * nb - 1])
            for color in range(1, stride - 1)
        ]
        max_size = max(sizes)
        highest = max((i for i, n in enumerate(sizes) if n), default=0)
        stats = layout["stats"]
        self_sizes = np.zeros(64, dtype=np.uint32)
        self_sizes[: min(64, len(sizes))] = sizes[:64]
        np.testing.assert_array_equal(
            actual["Stats"][stats["sizes"] : stats["sizes"] + 64], self_sizes
        )
        if (
            int(actual["Stats"][stats["max_bucket"]]) != max_size
            or int(actual["Stats"][stats["highest_color"]]) != highest
        ):
            raise AssertionError("Color dispatch statistics")
        np.testing.assert_array_equal(
            actual["Indirect"][:3], [min((max_size + 63) // 64, 256), 1, 1]
        )
    return {n: 0.0 for n in BUFFERS}


def main():
    from bucket_kernels import make_program
    from stage_replay import parse_args, run_stage

    args = parse_args(OPERATIONS, "bucket_build")
    run_stage(
        args,
        oracle="buckets",
        buffers=BUFFERS,
        make_kernel=None,
        launch=None,
        make_program=make_program,
        validate_case=validate_case,
        check_outputs=check_outputs,
    )


if __name__ == "__main__":
    main()
