"""Deferred collision ownership, BVH safety, and contact provenance regressions."""

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deferred_validation import (
    BUFFERS,
    OPERATIONS,
    check_outputs,
    diagnostics,
    validate_case,
)
from replay import HERE


class DeferredTests(unittest.TestCase):
    def fixture(self, op="deferred_pairs", name="n567-c1200-g3-normal"):
        root = Path(
            os.environ.get(
                "NEXUS_TILELANG_DEFERRED_FIXTURES", HERE / "results" / op / "c/fixtures"
            )
        )
        d = root / op / name
        l = json.loads((root / "layout.json").read_text())
        c = json.loads((d / "case.json").read_text())
        k = json.loads((HERE / "contracts" / f"{op}.json").read_text())
        b = {name: np.fromfile(d / f"{name}.bin", np.uint32) for name in BUFFERS}
        e = {
            name: np.fromfile(d / f"{name}.expected.bin", np.uint32)
            if name in k["outputs"]
            else v.copy()
            for name, v in b.items()
        }
        return l, c, b, e, k

    @unittest.skipUnless(importlib.util.find_spec("tilelang"), "TileLang unavailable")
    def test_promoted_reduction_constructs_tir(self):
        # A Python min introduced by lint autofix previously failed before lowering.
        from deferred_kernels import make_kernel

        l, c, _, _, _ = self.fixture("reduce_contacts", "n0-c1-g1-normal")
        self.assertIsNotNone(make_kernel(l, c, "c", "reduce_contacts"))

    def test_all_live_native_cases(self):
        for op in OPERATIONS:
            k = json.loads((HERE / "contracts" / f"{op}.json").read_text())
            for name in k["required_cases"]:
                with self.subTest(op=op, name=name):
                    l, c, b, e, k = self.fixture(op, name)
                    check_outputs(l, c, b, e, e, k)

    def test_atomic_order_and_overflow_subset_are_semantic(self):
        for name in ("n567-c1200-g3-normal", "n567-c17-g2-overflow"):
            l, c, b, e, k = self.fixture(name=name)
            a = {name: v.copy() for name, v in e.items()}
            n = min(c["candidate_count"], c["capacity"])
            w = l["pfm_pair_words"]
            rows = a["PfmPairs"].reshape(-1, w)
            if c["candidate_count"] > c["capacity"]:
                candidates = b["PfmCandidates"].reshape(-1, w)[: c["candidate_count"]]
                rows[:n] = candidates[-n:][::-1]
            else:
                rows[:n] = rows[:n][::-1].copy()
            check_outputs(l, c, b, a, e, k)
            if n > 1:
                rows[1] = rows[0]
                with self.assertRaisesRegex(AssertionError, "membership/ownership"):
                    check_outputs(l, c, b, a, e, k)

    def test_requested_count_and_candidate_bits_are_exact(self):
        for mutation in ("count", "shape", "identity"):
            l, c, b, e, k = self.fixture()
            a = {name: v.copy() for name, v in e.items()}
            if mutation == "count":
                a["PfmCount"][0] += 1
            else:
                a["PfmPairs"][
                    l["pfm_pair"]["shape1" if mutation == "shape" else "pair_index"]
                ] ^= 1
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                check_outputs(l, c, b, a, e, k)

    def test_composite_cycles_and_bad_vertices_are_rejected(self):
        for mutation in ("cycle", "vertex", "range", "duplicate"):
            l, c, b, _, _ = self.fixture()
            shapes = b["Shapes"].reshape(-1, l["shape_words"])
            sf = l["shape"]
            mesh = shapes[7 * 2]
            _vb, ib, n = map(int, mesh[sf["a"] : sf["a"] + 3])
            self.assertGreater(n, 0)
            if mutation == "cycle":
                b["Indices"][ib] = 0
            elif mutation == "vertex":
                b["Indices"][ib + n * 3] = 0xFFFFFFFF
            elif mutation == "range":
                mesh[sf["a"] + 2] = len(b["Indices"])
            else:
                b["Indices"][ib + 2 * 3 + 2] = b["Indices"][ib + 3 + 2]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_case(l, c, b)

    def test_invalid_pose_plan_and_permutation(self):
        for mutation in ("pose", "plan", "order"):
            l, c, b, _, _ = self.fixture("deferred_manifolds")
            if mutation == "pose":
                b["Poses"].view(np.float32)[l["pose_rotation"] + 3] = 2
            elif mutation == "plan":
                b["Plan"][l["plan"]["pfm_len"]] += 1
            else:
                b["Order"][0] = 0xFFFFFFFF
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_case(l, c, b)

    def test_storage_guards_and_readonly_are_exact(self):
        for op in OPERATIONS:
            l, c, b, e, k = self.fixture(op, "n1-c4-g1-normal")
            for name in b:
                a = {name: v.copy() for name, v in e.items()}
                a[name][-1] ^= 1
                with self.subTest(op=op, name=name), self.assertRaises(AssertionError):
                    check_outputs(l, c, b, a, e, k)

    def test_matching_bad_contacts_are_rejected(self):
        for mutation in ("point", "normal", "body", "count"):
            l, c, b, e, k = self.fixture("deferred_manifolds")
            rows = e["Contacts"].reshape(-1, l["contact_words"])
            cf, mf = l["contact"], l["manifold"]
            base = int(b["Plan"][l["plan"]["pfm_base"]])
            length = int(b["Plan"][l["plan"]["pfm_len"]])
            ci = cf["contact"] + mf["len"]
            i = base + int(np.flatnonzero(rows[base : base + length, ci] > 0)[0])
            row = rows[i]
            if mutation == "point":
                row.view(np.float32)[cf["contact"] + mf["points_a"]] += 10
            elif mutation == "normal":
                row.view(np.float32)[
                    cf["contact"] + mf["normal_a"] : cf["contact"] + mf["normal_a"] + 3
                ] = 0
            elif mutation == "body":
                row[cf["bodies"]] ^= 1
            else:
                row[ci] = 5
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                check_outputs(l, c, b, e, e, k)

    def test_keys_are_checked_against_the_native_records(self):
        for op in ("pfm_pair_keys", "pfm_subshape_keys", "pfm_permuted_keys"):
            l, c, b, e, k = self.fixture(op)
            e["Keys"][0] ^= 1
            with self.subTest(op=op), self.assertRaises(AssertionError):
                check_outputs(l, c, b, e, e, k)

    def test_reduction_requires_sorted_runs_and_cannot_invent_points(self):
        l, c, b, e, k = self.fixture("reduce_contacts")
        changed = {name: v.copy() for name, v in b.items()}
        changed["Keys"][0] = 0xFFFFFFFF
        with self.assertRaises(ValueError):
            validate_case(l, c, changed)
        rows = e["Contacts"].reshape(-1, l["contact_words"])
        cf, mf = l["contact"], l["manifold"]
        base = int(b["Plan"][l["plan"]["pfm_base"]])
        count = int(b["Plan"][l["plan"]["pfm_len"]])
        i = base + int(
            np.flatnonzero(rows[base : base + count, cf["contact"] + mf["len"]] > 0)[0]
        )
        rows[i].view(np.float32)[cf["contact"] + mf["points_a"]] += 10
        with self.assertRaisesRegex(AssertionError, "fabricated"):
            check_outputs(l, c, b, e, e, k)

    def test_overflow_diagnostics_retain_requested_work(self):
        l, c, b, e, k = self.fixture(name="n567-c17-g2-overflow")
        d = diagnostics(l, c, b, e, e, k)
        self.assertGreater(d["requested"], d["written"])
        self.assertEqual(d["overflow"], d["requested"] - d["written"])


if __name__ == "__main__":
    unittest.main()
