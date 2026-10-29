"""Native body-parameter packing, including sentinel and zero-timestep behavior."""

BUFFERS = ("RbdParams", "Gravity", "Params")


def validate_case(layout, case, arrays):
    import numpy as np

    sizes = {"RbdParams": layout["rbd_params_words"], "Gravity": 4, "Params": 9}
    if (
        case["kind"] != "body_coefficients"
        or case["count"] != 6
        or not 1 <= case["iterations"] <= 64
        or case["buffer_words"] != sizes
        or set(arrays) != set(sizes)
    ):
        raise ValueError("Invalid body coefficient case")
    if any(a.dtype != np.uint32 or a.shape != (sizes[n],) for n, a in arrays.items()):
        raise ValueError("Invalid body coefficient buffers")
    f = layout["rbd_params"]
    p = arrays["RbdParams"].view(np.float32)
    if (
        not np.isfinite(
            p[[f["dt"], f["normalized_max_linear_velocity"], f["length_unit"]]]
        ).all()
        or p[f["dt"]] < 0
        or p[f["length_unit"]] <= 0
        or p[f["normalized_max_linear_velocity"]] < 0
        or not np.isfinite(arrays["Gravity"].view(np.float32)[:3]).all()
    ):
        raise ValueError("Invalid body coefficient input")


def check_outputs(layout, case, before, actual, expected, contract):
    import numpy as np

    validate_case(layout, case, before)
    if set(actual) != set(before) or set(expected) != set(before):
        raise ValueError("Invalid coefficient outputs")
    for n in before:
        if actual[n].dtype != np.uint32 or actual[n].shape != before[n].shape:
            raise ValueError("Invalid coefficient output shape")
        if n != "Params":
            np.testing.assert_array_equal(
                actual[n], before[n], err_msg="Modified native uniform"
            )
    a = actual["Params"]
    e = expected["Params"]
    f = layout["rbd_params"]
    p = before["RbdParams"]
    np.testing.assert_array_equal(a[6:], before["Params"][6:])
    np.testing.assert_array_equal(
        a[[0, 3, 4, 5]], np.r_[p[f["dt"]], before["Gravity"][:3]]
    )
    if p.view(np.float32)[f["dt"]] == 0 and a[2] != 0:
        raise AssertionError("Zero timestep angular cap")
    if (
        p[f["normalized_max_linear_velocity"]] == layout["constants"]["max_flt_bits"]
        and a[1] != p[f["normalized_max_linear_velocity"]]
    ):
        raise AssertionError("Native MAX_FLT sentinel")
    af, ef = a.view(np.float32)[1:3], e.view(np.float32)[1:3]
    if not np.isfinite(af).all() or not np.isfinite(ef).all():
        raise AssertionError("Nonfinite body coefficients")
    np.testing.assert_allclose(af, ef, atol=contract["atol"], rtol=contract["rtol"])
    return {"Params": float(np.max(np.abs(af - ef)))}


def main():
    from stage_replay import parse_args, run_stage
    from body_uniform_kernels import make_kernel

    run_stage(
        parse_args(["body_coefficients"], "body_coefficients"),
        oracle="tgs",
        buffers=BUFFERS,
        make_kernel=make_kernel,
        validate_case=validate_case,
        check_outputs=check_outputs,
        launch=lambda c: (1, 1, c["iterations"]),
    )


if __name__ == "__main__":
    main()
