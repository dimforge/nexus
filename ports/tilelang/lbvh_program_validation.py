"""Independent resident LBVH ordering, bounds, topology and pair validation."""

import lbvh_validation as stage
import numpy as np
from lbvh_pair_validation import check as check_pairs
from lbvh_validation import write_policy
from radix_replay import expected_outputs as radix_expected

EXTRA = (
    "Identity",
    "SortKeys",
    "PongKeys",
    "PongValues",
    "RadixAux",
    "RadixOutAux",
    "RadixBatchIds",
    "RadixCounts",
    "Reduced",
    "SortGrid",
    "ReduceGrid",
    "NumKeys",
    "Config",
    *(f"Config{i}" for i in range(8)),
)
BUFFERS = stage.BUFFERS + EXTRA


def sizes(l, c):
    cap, nb = c["capacity"], c["batches"]
    out = stage.sizes(l, c)
    out.update(
        {
            n: cap * nb + 3
            for n in (
                "Identity",
                "SortKeys",
                "PongKeys",
                "PongValues",
                "RadixAux",
                "RadixOutAux",
                "RadixBatchIds",
            )
        }
    )
    out.update(
        RadixCounts=16 * ((cap + 1023) // 1024) * nb + 3,
        Reduced=1024 * nb + 3,
        SortGrid=6,
        ReduceGrid=6,
        NumKeys=nb,
    )
    out.update(
        {
            n: l["sort_uniforms_words"]
            for n in ("Config", *(f"Config{i}" for i in range(8)))
        }
    )
    return out


def validate_case(l, c, b):
    if c["kind"] != "lbvh_program":
        raise ValueError("Invalid LBVH program kind")
    subset = {n: b[n] for n in stage.BUFFERS}
    stage.validate_case(
        l,
        dict(
            c, kind="lbvh_domain", buffer_words={n: v.size for n, v in subset.items()}
        ),
        subset,
    )
    expected = sizes(l, c)
    if (
        set(b) != set(BUFFERS)
        or c["buffer_words"] != expected
        or any(v.dtype != np.uint32 or v.shape != (expected[n],) for n, v in b.items())
    ):
        raise ValueError("Invalid LBVH program storage")
    cap, nb, n = c["capacity"], c["batches"], c["colliders"]
    np.testing.assert_array_equal(b["NumKeys"], np.full(nb, n, np.uint32))
    np.testing.assert_array_equal(b["FrontierLen"][:nb], np.zeros(nb, np.uint32))
    for batch in range(nb):
        np.testing.assert_array_equal(
            b["Identity"][batch * cap : batch * cap + n], np.arange(n, dtype=np.uint32)
        )
    f = l["sort_uniforms"]
    for name, shift in [("Config", 0), *[(f"Config{i}", i * 4) for i in range(8)]]:
        if (
            b[name][f["shift"]] != shift
            or b[name][f["max_keys_per_batch"]] != cap
            or b[name][f["has_aux"]] != 0
        ):
            raise ValueError("Invalid LBVH sort configuration")


def check_sort(l, c, b, r):
    physical = {n: v.copy() for n, v in b.items()}
    physical["Keys"] = r["Keys"].copy()
    for p in range(8):
        mapping = {
            n: n for n in ("Config", "NumKeys", "Reduced", "SortGrid", "ReduceGrid")
        }
        mapping.update(
            Aux="RadixAux",
            OutAux="RadixOutAux",
            BatchIds="RadixBatchIds",
            Counts="RadixCounts",
        )
        mapping.update(
            Keys="Keys" if p == 0 else "PongKeys" if p % 2 else "SortKeys",
            Values="Identity" if p == 0 else "PongValues" if p % 2 else "Sorted",
            OutKeys="SortKeys" if p % 2 else "PongKeys",
            OutValues="Sorted" if p % 2 else "PongValues",
        )
        local = {n: physical[q].copy() for n, q in mapping.items()}
        if p == 0:
            local = radix_expected(
                l, dict(c, kind="radix_dispatch", shift=0, has_aux=0), local
            )
        local = radix_expected(
            l, dict(c, kind="radix_pass", shift=4 * p, has_aux=0), local
        )
        for n, q in mapping.items():
            physical[q] = local[n]
    for n in ("Sorted", *EXTRA):
        np.testing.assert_array_equal(
            r[n], physical[n], err_msg=f"Resident LBVH radix {n}"
        )


def check_outputs(l, c, b, a, e, k):
    validate_case(l, c, b)
    base = {n: b[n] for n in stage.BUFFERS}
    masks = {n: np.zeros(v.size, bool) for n, v in base.items()}
    for op in stage.OPERATIONS:
        m, _ = write_policy(l, dict(c, kind=op), base)
        for n in m:
            masks[n] |= m[n]
    floats = np.zeros(b["Tree"].size, bool)
    f = l["lbvh_node"]
    af = l["aabb"]
    w = l["lbvh_node_words"]
    cap, nb, n = c["capacity"], c["batches"], c["colliders"]
    for batch in range(nb):
        masks["Sorted"][batch * cap : batch * cap + n] = True
        for i in range(max(2 * n - 1, 0)):
            off = (batch * 2 * cap + i) * w
            for field in ("mins", "maxs"):
                floats[
                    off + f["aabb"] + af[field] : off + f["aabb"] + af[field] + 3
                ] = True
            floats[off + f["max_extent"]] = True
    for r in (a, e):
        if set(r) != set(b) or any(
            v.dtype != np.uint32 or v.shape != b[name].shape for name, v in r.items()
        ):
            raise ValueError("Invalid LBVH program output storage")
        for name, m in masks.items():
            np.testing.assert_array_equal(
                r[name][~m], b[name][~m], err_msg=f"LBVH program guard/readonly {name}"
            )
        stage.check_domains(l, dict(c, kind="lbvh_domain"), b, r)
        stage.check_domains(l, dict(c, kind="lbvh_domain_merge"), r, r)
        stage.check_morton(l, c, dict(b, Domain=r["Domain"]), r)
        check_sort(l, c, b, r)
        stage.topology(l, c, r, escapes=True)
        stage.check_geometry(l, c, dict(b, Sorted=r["Sorted"]), r, all_internal=True)
        stage.check_frontier(l, c, b, r)
        check_pairs(l, c, dict(b, Tree=r["Tree"]), r)
        np.testing.assert_array_equal(
            r["Rounds"][0], (int(np.max(r["FrontierLen"][:nb])) + 255) // 256
        )
        np.testing.assert_array_equal(
            r["Indirect"][:3],
            [(min(int(r["PairCount"][0]), c["pair_capacity"]) + 63) // 64, 1, 1],
        )
        np.testing.assert_array_equal(r["PairCount"][1:], 0)
    errors = {}
    for name in b:
        if name in ("Pairs", "Frontier"):
            continue
        if name == "Tree":
            np.testing.assert_array_equal(
                a[name][~floats],
                e[name][~floats],
                err_msg="Exact LBVH program topology",
            )
            aa, ee = (r[name][floats].view(np.float32) for r in (a, e))
            np.testing.assert_allclose(aa, ee, atol=k["atol"], rtol=k["rtol"])
            errors[name] = float(np.max(abs(aa - ee), initial=0))
        else:
            np.testing.assert_array_equal(a[name], e[name], err_msg=name)
            errors[name] = 0
    return errors
