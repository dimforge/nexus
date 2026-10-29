"""Exact native narrow-phase counter/dispatch contracts."""

BUFFERS = (
    "PairCount",
    "PfmCount",
    "Plan",
    "PfmSortLen",
    "ContactsIndirect",
    "PfmIndirect",
    "MbIndirect",
    "BatchIds",
)
OPERATIONS = ("reset_narrow", "contact_plan")


def validate_case(layout, case, arrays):
    import numpy as np

    if case["kind"] not in OPERATIONS or any(
        type(case[k]) is not int or not 0 <= case[k] < 2**27
        for k in ("count", "groups", "threads", "iterations")
    ):
        raise ValueError("Invalid contact plan dimensions")
    reset = case["kind"] == "reset_narrow"
    if (
        case["iterations"] != 1
        or case["groups"] != (max((case["count"] + 63) // 64, 1) if reset else 1)
        or case["threads"] != (64 if reset else 1)
        or (not reset and case["count"] != 1)
    ):
        raise ValueError("Invalid contact plan launch")
    sizes = dict(
        PairCount=3,
        PfmCount=max(case["count"], 1) + 3 if reset else 3,
        Plan=2 * layout["plan_words"],
        PfmSortLen=3,
        ContactsIndirect=6,
        PfmIndirect=6,
        MbIndirect=6,
        BatchIds=layout["batch_words"],
    )
    if (
        set(arrays) != set(BUFFERS)
        or case["buffer_words"] != sizes
        or any(
            a.dtype != np.uint32 or a.shape != (sizes[n],) for n, a in arrays.items()
        )
    ):
        raise ValueError("Invalid contact plan buffers")
    b = layout["batch"]
    ids = arrays["BatchIds"]
    cap = int(ids[b["collision_pairs_capacity"]])
    if cap > 0x7FFFFFFF or int(ids[b["contacts_capacity"]]) != 2 * cap:
        raise ValueError("Native contact capacity must equal twice the pair capacity")


def expected_outputs(layout, case, before):
    result = {n: v.copy() for n, v in before.items()}
    if case["kind"] == "reset_narrow":
        result["PfmCount"][: case["count"]] = 0
    else:
        ids = before["BatchIds"]
        b = layout["batch"]
        p = layout["plan"]
        cap = int(ids[b["collision_pairs_capacity"]])
        pairs = min(int(before["PairCount"][0]), cap)
        pfm = min(int(before["PfmCount"][0]), cap)
        bound = pairs + pfm
        result["Plan"][p["bound"]] = bound
        result["Plan"][p["pfm_base"]] = pairs
        result["Plan"][p["pfm_len"]] = pfm
        result["PfmSortLen"][0] = pfm
        result["ContactsIndirect"][:3] = [(bound + 63) // 64, 1, 1]
        result["PfmIndirect"][:3] = [(pfm + 63) // 64, 1, 1]
        result["MbIndirect"][:3] = [
            int(ids[b["multibodies_batch_capacity"]]) if bound else 0,
            int(ids[b["num_batches"]]),
            1,
        ]
    return result


def check_outputs(layout, case, before, actual, expected, contract):
    import numpy as np

    validate_case(layout, case, before)
    if set(actual) != set(before) or set(expected) != set(before):
        raise ValueError("Missing contact plan output")
    independent = expected_outputs(layout, case, before)
    for n in before:
        for values in (actual, expected):
            if values[n].dtype != np.uint32 or values[n].shape != before[n].shape:
                raise ValueError("Invalid contact plan output shape")
            np.testing.assert_array_equal(values[n], independent[n], err_msg=n)
    return {n: 0 for n in contract["outputs"]}


def main():
    from stage_replay import parse_args, run_stage
    from contact_plan_kernels import make_program

    run_stage(
        parse_args(OPERATIONS, "contact_plan"),
        oracle="contact_plans",
        buffers=BUFFERS,
        make_kernel=None,
        make_program=make_program,
        validate_case=validate_case,
        check_outputs=check_outputs,
        launch=lambda c: (c["groups"], c["threads"], 1),
    )


if __name__ == "__main__":
    main()
