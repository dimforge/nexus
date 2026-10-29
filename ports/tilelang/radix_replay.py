"""Exact native radix ABI checks with an independent stable integer specification."""

BUFFERS = (
    "Config",
    "NumKeys",
    "Keys",
    "Values",
    "Aux",
    "OutKeys",
    "OutValues",
    "OutAux",
    "BatchIds",
    "Counts",
    "Reduced",
    "SortGrid",
    "ReduceGrid",
    "InitConfig",
) + tuple(f"Config{i}" for i in range(8))
STAGES = (
    "radix_count",
    "radix_reduce",
    "radix_scan",
    "radix_scan_add",
    "radix_scatter",
)
OPERATIONS = (
    "radix_dispatch",
    "radix_init_batched",
    *STAGES,
    "radix_pass",
    "radix_sort",
)


def validate_case(layout, case, arrays):
    import numpy as np

    if case["kind"] not in OPERATIONS or any(
        type(case[k]) is not int
        for k in ("capacity", "batches", "count", "shift", "has_aux", "iterations")
    ):
        raise ValueError("Invalid radix case")
    cap, batches = case["capacity"], case["batches"]
    if not (
        0 <= cap <= 64 * 1024 * 1024
        and 0 < batches < 2**20
        and cap * batches < 2**27
        and case["shift"] in range(0, 32, 4)
        and case["has_aux"] in (0, 1)
        and case["iterations"] == 1
    ):
        raise ValueError("Invalid radix dimensions or pass configuration")
    synthetic = case["synthetic_counts"]
    if type(synthetic) is not bool or (
        synthetic
        and case["kind"] not in ("radix_reduce", "radix_scan", "radix_scan_add")
    ):
        raise ValueError(
            "Synthetic histogram inputs are restricted to histogram stages"
        )
    size = 3 if synthetic else cap * batches + 3
    sizes = {
        n: size
        for n in ("Keys", "Values", "Aux", "OutKeys", "OutValues", "OutAux", "BatchIds")
    }
    sizes.update(
        NumKeys=batches,
        Counts=16 * ((cap + 1023) // 1024) * batches + 3,
        Reduced=1024 * batches + 3,
        SortGrid=6,
        ReduceGrid=6,
    )
    sizes.update(
        {
            n: layout["sort_uniforms_words"]
            for n in ("Config", "InitConfig", *(f"Config{i}" for i in range(8)))
        }
    )
    if (
        set(arrays) != set(BUFFERS)
        or case["buffer_words"] != sizes
        or any(
            v.dtype != np.uint32 or v.shape != (sizes[n],) for n, v in arrays.items()
        )
    ):
        raise ValueError("Invalid radix buffer storage")
    n = arrays["NumKeys"]
    if np.any(n > cap) or int(np.sum(n, dtype=np.uint64)) != case["count"]:
        raise ValueError("Invalid active radix key counts")
    f = layout["sort_uniforms"]
    for name, shift in [("Config", case["shift"]), ("InitConfig", batches)] + [
        (f"Config{i}", i * 4) for i in range(8)
    ]:
        v = arrays[name]
        if (
            v[f["shift"]] != shift
            or v[f["max_keys_per_batch"]] != cap
            or v[f["has_aux"]] != case["has_aux"]
        ):
            raise ValueError("Radix uniform differs from case configuration")


def expected_outputs(layout, case, before):
    import numpy as np

    a = {n: v.copy() for n, v in before.items()}
    cap = case["capacity"]
    batches = case["batches"]
    max_wgs = (cap + 1023) // 1024

    def stage(kind, shift):
        if kind == "radix_dispatch":
            maximum = int(np.max(a["NumKeys"]))
            groups = (maximum + 1023) // 1024
            a["SortGrid"][:3] = [groups, batches, 1]
            a["ReduceGrid"][:3] = [16 * ((groups + 1023) // 1024), batches, 1]
            return
        for b, count in enumerate(a["NumKeys"]):
            n = int(count)
            groups = (n + 1023) // 1024
            chunks = (groups + 1023) // 1024
            co = b * 16 * max_wgs
            ro = b * 1024
            ko = b * cap
            sl = slice(ko, ko + n)
            if kind == "radix_init_batched":
                a["BatchIds"][ko : ko + cap] = b
                a["OutKeys"][sl] = a["Keys"][sl]
                a["OutValues"][sl] = a["Values"][sl]
                a["OutKeys"][ko + n : ko + cap] = 2**32 - 1
                a["OutValues"][ko + n : ko + cap] = 0
            elif kind == "radix_count":
                for group in range(groups):
                    digits = (
                        a["Keys"][ko + group * 1024 : ko + min((group + 1) * 1024, n)]
                        >> shift
                    ) & 15
                    hist = np.bincount(digits, minlength=16)
                    for digit in range(16):
                        a["Counts"][co + digit * groups + group] = hist[digit]
            elif kind == "radix_reduce":
                for digit in range(16):
                    for chunk in range(chunks):
                        data = a["Counts"][
                            co + digit * groups + chunk * 1024 : co
                            + digit * groups
                            + min((chunk + 1) * 1024, groups)
                        ]
                        a["Reduced"][ro + digit * chunks + chunk] = (
                            int(np.sum(data, dtype=np.uint64)) & 0xFFFFFFFF
                        )
            elif kind == "radix_scan":
                length = 16 * chunks
                data = a["Reduced"][ro : ro + length].copy()
                if length:
                    a["Reduced"][ro : ro + length] = np.r_[
                        np.uint64(0), np.cumsum(data, dtype=np.uint64)[:-1]
                    ].astype(np.uint32)
            elif kind == "radix_scan_add":
                for digit in range(16):
                    for chunk in range(chunks):
                        start = co + digit * groups + chunk * 1024
                        end = co + digit * groups + min((chunk + 1) * 1024, groups)
                        data = a["Counts"][start:end].copy()
                        exclusive = np.r_[
                            np.uint64(0), np.cumsum(data, dtype=np.uint64)[:-1]
                        ]
                        a["Counts"][start:end] = (
                            exclusive
                            + np.uint64(a["Reduced"][ro + digit * chunks + chunk])
                        ).astype(np.uint32)
            elif kind == "radix_scatter":
                order = np.argsort((a["Keys"][sl] >> shift) & 15, kind="stable")
                for src, dst in [
                    ("Keys", "OutKeys"),
                    ("Values", "OutValues"),
                    ("Aux", "OutAux"),
                ]:
                    if src != "Aux" or case["has_aux"]:
                        a[dst][sl] = a[src][sl][order]

    op = case["kind"]
    if op == "radix_sort":
        for p in range(8):
            if p % 2:
                for x, y in [
                    ("Keys", "OutKeys"),
                    ("Values", "OutValues"),
                    ("Aux", "OutAux"),
                ]:
                    a[x], a[y] = a[y], a[x]
            for kind in STAGES:
                stage(kind, p * 4)
            if p % 2:
                for x, y in [
                    ("Keys", "OutKeys"),
                    ("Values", "OutValues"),
                    ("Aux", "OutAux"),
                ]:
                    a[x], a[y] = a[y], a[x]
    elif op == "radix_pass":
        for kind in STAGES:
            stage(kind, case["shift"])
    else:
        stage(op, case["shift"])
    return a


def check_outputs(layout, case, before, actual, expected, contract):
    import numpy as np

    validate_case(layout, case, before)
    spec = expected_outputs(layout, case, before)
    for result in (actual, expected):
        if set(result) != set(before) or any(
            v.dtype != np.uint32 or v.shape != before[n].shape
            for n, v in result.items()
        ):
            raise ValueError("Invalid radix output storage")
        for n in before:
            np.testing.assert_array_equal(result[n], spec[n], err_msg=n)
    return {n: 0 for n in contract["outputs"]}


def main():
    from stage_replay import parse_args, run_stage
    from radix_kernels import make_program

    run_stage(
        parse_args(OPERATIONS, "radix_sort"),
        oracle="radix",
        buffers=BUFFERS,
        make_kernel=None,
        make_program=make_program,
        validate_case=validate_case,
        check_outputs=check_outputs,
        launch=lambda c: (1, 256, 1),
    )


if __name__ == "__main__":
    main()
