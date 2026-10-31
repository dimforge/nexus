"""Exact native scheduling feedback with an independent palette/completion check."""

import numpy as np
from stage_replay import parse_args, run_stage

BUFFERS = ("Colors", "Links", "Plan", "Budget", "Requirements")


def validate_case(l, c, b):
    if (
        c["kind"] != "color_requirements"
        or type(c["count"]) is not int
        or not 0 <= c["count"] < 2**24
        or c["storage"] != c["count"] + 3
        or (c["groups"], c["threads"], c["iterations"]) != (1, 1, 1)
    ):
        raise ValueError("Invalid coloring requirement dimensions")
    sizes = {
        "Colors": c["storage"],
        "Links": c["storage"] * l["link_words"],
        "Plan": l["plan_words"],
        "Budget": 2,
        "Requirements": 5,
    }
    if (
        c["buffer_words"] != sizes
        or set(b) != set(BUFFERS)
        or any(a.dtype != np.uint32 or a.shape != (sizes[n],) for n, a in b.items())
        or b["Plan"][l["plan"]["bound"]] != c["count"]
    ):
        raise ValueError("Invalid coloring requirement storage")
    lengths = b["Links"].reshape(-1, l["link_words"])[: c["count"], l["link"]["len"]]
    if np.any(lengths > 4):
        raise ValueError("Invalid active contact length")
    return b["Colors"][: c["count"]][lengths != 0]


def check_outputs(l, c, b, a, e, k):
    colors = validate_case(l, c, b)
    maximum = int(colors.max()) if colors.size else 0
    required = [
        maximum if maximum < 64 else 0xFFFFFFFF,
        min(int(b["Budget"][0]) + 1, 0xFFFFFFFF) if np.any(colors == 0) else 0,
    ]
    for result in (a, e):
        if set(result) != set(BUFFERS):
            raise ValueError("Invalid coloring requirement outputs")
        for n in BUFFERS:
            if result[n].dtype != np.uint32 or result[n].shape != b[n].shape:
                raise ValueError("Invalid coloring requirement output shape")
            start = 2 if n == "Requirements" else 0
            np.testing.assert_array_equal(
                result[n][start:], b[n][start:], err_msg=f"Exact guard/input {n}"
            )
        np.testing.assert_array_equal(
            result["Requirements"][:2],
            required,
            err_msg="Independent coloring requirements",
        )
    return {}


def main():
    from color_requirement_kernels import make_kernel

    run_stage(
        parse_args(["color_requirements"], "color_requirements"),
        oracle="color-requirements",
        buffers=BUFFERS,
        make_kernel=make_kernel,
        validate_case=validate_case,
        check_outputs=check_outputs,
        launch=lambda c: (1, 64, 1),
    )


if __name__ == "__main__":
    main()
