"""Native Rust contact softness parameters, including exact sentinel behavior."""

BUFFERS = ("RbdParams", "Coefficients")


def validate_case(layout, case, arrays):
    import numpy as np

    sizes = {"RbdParams": layout["rbd_params_words"], "Coefficients": 9}
    if (
        case["kind"] != "contact_coefficients"
        or case["count"] != 6
        or case["iterations"] != 3
        or case["buffer_words"] != sizes
        or set(arrays) != set(sizes)
    ):
        raise ValueError("Invalid coefficient dimensions")
    for name, size in sizes.items():
        a = arrays[name]
        if a.dtype != np.uint32 or a.ndim != 1 or a.size != size:
            raise ValueError("Invalid coefficient buffer")
    p = arrays["RbdParams"].view(np.float32)
    fields = layout["rbd_params"]
    if any(not np.isfinite(p[offset]) for offset in fields.values()):
        raise ValueError("Nonfinite coefficient input")
    if (
        p[fields["dt"]] < 0
        or p[fields["length_unit"]] <= 0
        or p[fields["normalized_max_corrective_velocity"]] < 0
    ):
        raise ValueError("Invalid timestep or physical scale")
    for prefix in ("contact", "static_contact"):
        if (
            p[fields[prefix + "_natural_frequency"]] < 0
            or p[fields[prefix + "_damping_ratio"]] <= 0
        ):
            raise ValueError("Invalid spring parameters")


def check_outputs(layout, case, before, actual, expected, contract):
    import numpy as np

    validate_case(layout, case, before)
    for n in BUFFERS:
        if actual[n].shape != before[n].shape or actual[n].dtype != np.uint32:
            raise ValueError("Invalid returned coefficient buffer")
    np.testing.assert_array_equal(actual["RbdParams"], before["RbdParams"])
    np.testing.assert_array_equal(
        actual["Coefficients"][6:], before["Coefficients"][6:]
    )
    a, e = (
        actual["Coefficients"][:6].view(np.float32),
        expected["Coefficients"][:6].view(np.float32),
    )
    if not np.isfinite(a).all() or not np.isfinite(e).all():
        raise AssertionError("Nonfinite coefficient result")
    np.testing.assert_allclose(a, e, atol=contract["atol"], rtol=contract["rtol"])
    fields = layout["rbd_params"]
    if (
        before["RbdParams"][fields["normalized_max_corrective_velocity"]]
        == layout["constants"]["max_flt_bits"]
    ):
        if actual["Coefficients"][1] != layout["constants"]["max_flt_bits"]:
            raise AssertionError(
                "MAX_FLT must remain exact even when length_unit exceeds one"
            )
    if before["RbdParams"].view(np.float32)[fields["dt"]] == 0:
        np.testing.assert_array_equal(a[[0, 2, 4]], np.zeros(3, dtype=np.float32))
    return {"Coefficients": float(np.max(np.abs(a - e))), "RbdParams": 0.0}


def main():
    from coefficient_kernels import make_kernel
    from stage_replay import parse_args, run_stage

    args = parse_args(("contact_coefficients",), "contact_coefficients")
    run_stage(
        args,
        oracle="coefficients",
        buffers=BUFFERS,
        make_kernel=lambda layout, case, target: make_kernel(layout, target),
        validate_case=validate_case,
        check_outputs=check_outputs,
        launch=lambda case: (1, 1, case["iterations"]),
    )


if __name__ == "__main__":
    main()
