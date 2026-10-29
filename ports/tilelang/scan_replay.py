#!/usr/bin/env python3
"""Exact Rust parity for batched prefix kernels and their ordered hierarchy."""

BUFFERS = (
    "Data",
    "Aux0",
    "Aux1",
    "Aux2",
    "Aux3",
    "Stride0",
    "Stride1",
    "Stride2",
    "Stride3",
)
OPERATIONS = ("prefix_sweep", "prefix_add", "prefix_scan")


def validate_case(layout, case, arrays):
    import numpy as np

    stride, batches = case["stride"], case["batches"]
    if (
        layout != {"word_bytes": 4, "workgroup_size": 256}
        or case["kind"] not in OPERATIONS
        or case["iterations"] != 1
        or type(stride) is not int
        or not 0 <= stride <= (1 << 31) - 256
        or type(batches) is not int
        or not 1 <= batches < (1 << 31) // 256
        or stride * batches >= (1 << 31) - 256
        or case["count"] != stride * batches
    ):
        raise ValueError("Invalid scan dimensions")
    strides = [stride]
    for _ in range(4):
        strides.append((strides[-1] + 255) // 256)
    levels = next(i for i in range(1, 5) if strides[i] <= 1)
    if case["strides"] != strides[:4] or case["levels"] != levels:
        raise ValueError("Invalid scan hierarchy")
    sizes = {"Data": stride * batches + 3}
    for i in range(4):
        sizes[f"Aux{i}"] = strides[i + 1] * batches + 3
        sizes[f"Stride{i}"] = 2
    if case["buffer_words"] != sizes or set(arrays) != set(BUFFERS):
        raise ValueError("Invalid scan storage layout")
    for name, size in sizes.items():
        a = arrays[name]
        if a.dtype != np.uint32 or a.ndim != 1 or a.size != size:
            raise ValueError(f"Invalid {name} storage")
    for i in range(4):
        if int(arrays[f"Stride{i}"][0]) != strides[i]:
            raise ValueError("Stride uniform differs from specialization")
    # Fixture/oracle contract intentionally excludes overflowing sums; original
    # Rust debug shader arithmetic traps while release GPU arithmetic wraps.
    data = arrays["Data"][: stride * batches].reshape(batches, stride)
    if case["kind"] == "prefix_add":
        for batch in range(batches):
            for group in range(strides[1]):
                words = data[batch, group * 256 : (group + 1) * 256].astype(np.uint64)
                if (
                    words + int(arrays["Aux0"][batch * strides[1] + group]) > 0xFFFFFFFF
                ).any():
                    raise ValueError("Overflowing prefix addition")
    elif (data.sum(axis=1, dtype=np.uint64) > 0xFFFFFFFF).any():
        raise ValueError("Overflowing scan total")
    masks = {n: np.zeros(sizes[n], dtype=bool) for n in BUFFERS}
    masks["Data"][: stride * batches] = True
    if stride and case["kind"] != "prefix_add":
        for i in range(levels if case["kind"] == "prefix_scan" else 1):
            masks[f"Aux{i}"][: strides[i + 1] * batches] = True
    return masks


def check_outputs(layout, case, before, actual, expected, contract):
    import numpy as np

    masks = validate_case(layout, case, before)
    for name, mask in masks.items():
        if actual[name].shape != before[name].shape or actual[name].dtype != np.uint32:
            raise ValueError(f"Invalid returned {name}")
        np.testing.assert_array_equal(
            actual[name][~mask],
            before[name][~mask],
            err_msg=f"Unexpected write to {name}",
        )
        np.testing.assert_array_equal(
            actual[name], expected[name], err_msg=f"Rust scan mismatch in {name}"
        )
    if case["kind"] == "prefix_scan":
        stride = case["stride"]
        for batch in range(case["batches"]):
            start = batch * stride
            values = before["Data"][start : start + stride]
            independent = np.zeros(stride, dtype=np.uint64)
            if stride > 1:
                independent[1:] = np.cumsum(values[:-1], dtype=np.uint64)
            np.testing.assert_array_equal(
                actual["Data"][start : start + stride], independent
            )
    return {n: 0.0 for n in BUFFERS}


def main():
    from scan_kernels import make_program
    from stage_replay import parse_args, run_stage

    args = parse_args(OPERATIONS, "prefix_scan")
    run_stage(
        args,
        oracle="scans",
        buffers=BUFFERS,
        make_kernel=None,
        make_program=make_program,
        validate_case=validate_case,
        check_outputs=check_outputs,
        launch=None,
    )


if __name__ == "__main__":
    main()
