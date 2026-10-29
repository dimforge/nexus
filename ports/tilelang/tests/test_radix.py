"""Stable native radix ordering, histogram boundaries and exact ABI rejection."""

import json, os, sys, unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay import HERE
import radix_replay as radix


class RadixTests(unittest.TestCase):
    def fixture(self, kind="radix_sort", name="n1025-b3-s4-a0-mixed"):
        root = (
            Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
            / kind
            / "c/fixtures"
        )
        l = json.loads((root / "layout.json").read_text())
        cases = json.loads((root / kind / "suite.json").read_text())["cases"]
        c = next(c for c in cases if c["name"] == name)
        d = root / kind / name
        b = {n: np.fromfile(d / f"{n}.bin", np.uint32) for n in radix.BUFFERS}
        e = {n: np.fromfile(d / f"{n}.expected.bin", np.uint32) for n in b}
        k = json.loads((HERE / f"contracts/{kind}.json").read_text())
        return l, c, b, e, k

    def test_all_native_stages_match_independent_specification(self):
        for op in radix.OPERATIONS:
            k = json.loads((HERE / f"contracts/{op}.json").read_text())
            for name in k["required_cases"]:
                with self.subTest(operation=op, case=name):
                    l, c, b, e, k = self.fixture(op, name)
                    radix.check_outputs(l, c, b, e, e, k)

    def test_duplicate_keys_preserve_values_and_auxiliary_records(self):
        l, c, b, e, k = self.fixture(name="n256-b2-s0-a1-equal")
        for batch, n in enumerate(b["NumKeys"]):
            sl = slice(batch * c["capacity"], batch * c["capacity"] + int(n))
            for name in ["Keys", "Values", "Aux"]:
                np.testing.assert_array_equal(e[name][sl], b[name][sl])
        a = {n: v.copy() for n, v in e.items()}
        a["Values"][[0, 1]] = a["Values"][[1, 0]]
        with self.assertRaises(AssertionError):
            radix.check_outputs(l, c, b, a, a, k)

    def test_every_word_and_guard_is_exact(self):
        for name in radix.BUFFERS:
            for index in [0, -1]:
                l, c, b, e, k = self.fixture()
                a = {n: v.copy() for n, v in e.items()}
                a[name][index] ^= 1
                with (
                    self.subTest(buffer=name, index=index),
                    self.assertRaises(AssertionError),
                ):
                    radix.check_outputs(l, c, b, a, e, k)

    def test_multichunk_histogram_and_empty_batch(self):
        for op in ["radix_reduce", "radix_scan", "radix_scan_add"]:
            l, c, b, e, k = self.fixture(op, "n1048577-b3-s0-a0-histograms")
            self.assertTrue(c["synthetic_counts"])
            self.assertEqual(len(b["Keys"]), 3)
            self.assertEqual((int(b["NumKeys"][0]) + 1023) // 1024, 1025)
            self.assertEqual(b["NumKeys"][2], 0)
            radix.check_outputs(l, c, b, e, e, k)
            a = {n: v.copy() for n, v in e.items()}
            a["Reduced"][1] ^= 1
            with self.assertRaises(AssertionError):
                radix.check_outputs(l, c, b, a, e, k)

    def test_batch_initialization_preserves_full_width_active_keys(self):
        l, c, b, e, k = self.fixture("radix_init_batched", "n1024-b2-s28-a1-mixed")
        for batch, n in enumerate(b["NumKeys"]):
            start = batch * c["capacity"]
            end = start + c["capacity"]
            np.testing.assert_array_equal(
                e["OutKeys"][start : start + int(n)], b["Keys"][start : start + int(n)]
            )
            np.testing.assert_array_equal(
                e["OutKeys"][start + int(n) : end],
                np.full(c["capacity"] - int(n), 2**32 - 1, np.uint32),
            )
            np.testing.assert_array_equal(
                e["BatchIds"][start:end], np.full(c["capacity"], batch, np.uint32)
            )

    def test_invalid_counts_uniforms_storage_and_scope_fail(self):
        for mutation in [
            "count",
            "shift",
            "aux",
            "capacity",
            "batches",
            "synthetic",
            "dtype",
            "shape",
            "iterations",
        ]:
            l, c, b, e, k = self.fixture()
            if mutation == "count":
                b["NumKeys"][0] = c["capacity"] + 1
            if mutation == "shift":
                b["Config7"][l["sort_uniforms"]["shift"]] = 32
            if mutation == "aux":
                b["Config"][l["sort_uniforms"]["has_aux"]] = 2
            if mutation == "capacity":
                c["capacity"] = 64 * 1024 * 1024 + 1
            if mutation == "batches":
                c["batches"] = 0
            if mutation == "synthetic":
                c["synthetic_counts"] = True
            if mutation == "dtype":
                b["Keys"] = b["Keys"].astype(np.int32)
            if mutation == "shape":
                b["Counts"] = b["Counts"][:-1]
            if mutation == "iterations":
                c["iterations"] = 2
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                radix.validate_case(l, c, b)
