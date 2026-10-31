"""Exact canonical key/permutation contracts, including every radix workspace word."""

import radix_replay as radix

BASE_BUFFERS = (
    "Contacts",
    "Output",
    "InOrder",
    "Keys",
    "Values",
    "SortCount",
    "Plan",
    "BatchIds",
    "Selector",
)
BUFFERS = (
    BASE_BUFFERS
    + (
        "SortKeys",
        "PongKeys",
        "PongValues",
        "RadixAux",
        "RadixOutAux",
        "RadixBatchIds",
        "Counts",
        "Reduced",
        "SortGrid",
        "ReduceGrid",
        "Config",
        "InitConfig",
    )
    + tuple(f"Config{i}" for i in range(8))
)
OPERATIONS = ("contact_keys", "permute_contacts", "canonical_contacts")


def sort_mapping(p):
    m = {
        n: n for n in radix.BUFFERS if n != "InitConfig" and not n.startswith("Config")
    }
    m.update(
        NumKeys="SortCount",
        BatchIds="RadixBatchIds",
        Aux="RadixAux",
        OutAux="RadixOutAux",
        Config=f"Config{p}",
    )
    if p == 0:
        m.update(Keys="Keys", Values="Values")
    elif p % 2:
        m.update(Keys="PongKeys", Values="PongValues")
    else:
        m.update(Keys="SortKeys", Values="InOrder")
    if p % 2:
        m.update(OutKeys="SortKeys", OutValues="InOrder")
    else:
        m.update(OutKeys="PongKeys", OutValues="PongValues")
    return m


def validate_case(layout, case, arrays):
    import numpy as np

    if case["kind"] not in OPERATIONS or any(
        type(case[k]) is not int or not 0 <= case[k] < 2**27
        for k in ("count", "storage", "groups", "iterations")
    ):
        raise ValueError("Invalid canonical dimensions")
    n, storage = case["count"], case["storage"]
    if (
        not n < storage <= 64 * 1024 * 1024
        or not 0 < case["groups"] * 64 < 2**31
        or case["iterations"] != 1
    ):
        raise ValueError("Invalid canonical capacities or launch")
    sizes = {
        k: storage
        for k in (
            "InOrder",
            "Keys",
            "Values",
            "SortKeys",
            "PongKeys",
            "PongValues",
            "RadixAux",
            "RadixOutAux",
            "RadixBatchIds",
        )
    }
    sizes.update({k: storage * layout["contact_words"] for k in ("Contacts", "Output")})
    sizes.update(
        SortCount=1,
        Plan=2 * layout["plan_words"],
        BatchIds=layout["batch_words"],
        Selector=2,
        Counts=16 * ((storage + 1023) // 1024) + 3,
        Reduced=1027,
        SortGrid=6,
        ReduceGrid=6,
    )
    sizes.update(
        {
            k: layout["sort_uniforms_words"]
            for k in ("Config", "InitConfig", *(f"Config{i}" for i in range(8)))
        }
    )
    if (
        set(arrays) != set(BUFFERS)
        or case["buffer_words"] != sizes
        or any(s >= 2**31 for s in sizes.values())
        or any(
            v.dtype != np.uint32 or v.shape != (sizes[k],) for k, v in arrays.items()
        )
    ):
        raise ValueError("Invalid canonical buffers")
    if (
        arrays["Plan"][layout["plan"]["bound"]] != n
        or layout["default_contact"] != [0] * layout["contact_words"]
    ):
        raise ValueError("Invalid canonical contact plan or native default contact")
    shift = int(arrays["BatchIds"][layout["batch"]["contact_sort_collider_shift"]])
    if (
        shift >= 32
        or type(case["selector"]) is not int
        or not 0 <= case["selector"] < 8
        or case["selector"] != int(arrays["Selector"][0])
    ):
        raise ValueError("Invalid native contact key selector or shift")
    f = layout["sort_uniforms"]
    for name, shift_value in [("Config", 0), ("InitConfig", 1)] + [
        (f"Config{i}", i * 4) for i in range(8)
    ]:
        v = arrays[name]
        if (
            v[f["shift"]] != shift_value
            or v[f["max_keys_per_batch"]] != storage
            or v[f["has_aux"]] != 0
        ):
            raise ValueError("Invalid canonical radix uniforms")
    order = arrays["InOrder"][:n]
    if np.any(order >= n) or len(set(map(int, order))) != n:
        raise ValueError("Invalid input contact permutation")
    rows = arrays["Contacts"].reshape(-1, layout["contact_words"])[:n]
    length = rows[:, layout["contact"]["contact"] + layout["manifold"]["len"]]
    if np.any(length > 4):
        raise ValueError("Invalid native manifold length")
    if "collider_count" in case:
        bound = case["collider_count"]
        live = rows[length != 0]
        offset = layout["contact"]["colliders"]
        if (
            type(bound) is not int or not 0 <= bound < 2**32
            or np.any(live[:, offset : offset + 2] >= bound)
        ):
            raise ValueError("Contact collider IDs exceed the declared bound")
    if case["kind"] == "canonical_contacts":
        if case["passes"] not in ([7], [5, 2], [4, 1, 2]):
            raise ValueError("Unsupported native canonical field schedule")
        if case["passes"] == [7]:
            live = rows[length != 0]
            offset = layout["contact"]["colliders"]
            if shift > 16 or np.any(live[:, offset : offset + 2] >= 2**shift):
                raise ValueError("Packed contact key cannot represent collider IDs")


def expected_outputs(layout, case, before):
    import numpy as np

    a = {n: v.copy() for n, v in before.items()}
    n = case["count"]
    c, m = layout["contact"], layout["manifold"]
    rows = before["Contacts"].reshape(-1, layout["contact_words"])
    shift = int(before["BatchIds"][layout["batch"]["contact_sort_collider_shift"]])
    selectors = (
        case["passes"] if case["kind"] == "canonical_contacts" else [case["selector"]]
    )
    if case["kind"] != "permute_contacts":
        for selector in selectors:
            a["SortCount"][0] = n
            for i in range(n):
                index = i if selector >= 4 else int(a["InOrder"][i])
                row = rows[index]
                field = selector % 4
                if row[c["contact"] + m["len"]] == 0:
                    key = 2**32 - 1
                elif field == 3:
                    key = (
                        (int(row[c["colliders"]]) << shift)
                        | int(row[c["colliders"] + 1])
                    ) & 0xFFFFFFFF
                else:
                    key = int(
                        row[c["colliders"]]
                        if field == 2
                        else row[c["colliders"] + 1]
                        if field == 1
                        else row[c["subshape"]]
                    )
                a["Keys"][i] = key
                a["Values"][i] = index
            if case["kind"] == "canonical_contacts":
                # Independent sorted order, in addition to checking workspace state.
                stable = np.argsort(a["Keys"][:n], kind="stable")
                ids = a["Values"][:n][stable]
                bits = 32
                if "collider_count" in case and field in (1, 2):
                    bits = max(1, case["collider_count"].bit_length())
                # Independent integer workspace specification at byte granularity.
                for p in range(((bits + 7) // 8) * 2):
                    mapping = sort_mapping(p)
                    local = {k: a[v] for k, v in mapping.items()}
                    rc = dict(
                        kind="radix_pass",
                        capacity=case["storage"],
                        batches=1,
                        shift=p * 4,
                        has_aux=0,
                    )
                    if p == 0:
                        local = radix.expected_outputs(
                            layout, dict(rc, kind="radix_dispatch"), local
                        )
                    result = radix.expected_outputs(layout, rc, local)
                    for k, v in mapping.items():
                        a[v] = result[k]
                np.testing.assert_array_equal(a["InOrder"][:n], ids)
    if case["kind"] != "contact_keys":
        output = a["Output"].reshape(-1, layout["contact_words"])
        for i in range(n):
            row = rows[a["InOrder"][i]]
            output[i] = (
                row
                if row[c["contact"] + m["len"]]
                else np.array(layout["default_contact"], dtype=np.uint32)
            )
    return a


def check_outputs(layout, case, before, actual, expected, contract):
    import numpy as np

    validate_case(layout, case, before)
    spec = expected_outputs(layout, case, before)
    for result in (actual, expected):
        if set(result) != set(before) or any(
            v.dtype != np.uint32 or v.shape != before[k].shape
            for k, v in result.items()
        ):
            raise ValueError("Invalid canonical output storage")
        for k in before:
            np.testing.assert_array_equal(result[k], spec[k], err_msg=k)
    return {n: 0 for n in contract["outputs"]}


def main():
    from stage_replay import parse_args, run_stage
    from canonical_kernels import make_program

    run_stage(
        parse_args(OPERATIONS, "canonical_contacts"),
        oracle="canonical",
        buffers=BUFFERS,
        make_kernel=None,
        make_program=make_program,
        validate_case=validate_case,
        check_outputs=check_outputs,
        launch=lambda c: (c["groups"], 64, 1),
    )


if __name__ == "__main__":
    main()
