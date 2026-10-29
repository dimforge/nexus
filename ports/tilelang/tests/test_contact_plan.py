"""Exact counter publication, overflow, reset and dispatch regressions."""

import json, os, sys, unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay import HERE
import contact_plan_replay as plan


class ContactPlanTests(unittest.TestCase):
    def fixture(self, kind="contact_plan", name="65"):
        root = (
            Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
            / kind
            / "c/fixtures"
        )
        l = json.loads((root / "layout.json").read_text())
        cases = json.loads((root / kind / "suite.json").read_text())["cases"]
        c = next(x for x in cases if x["name"] == name)
        d = root / kind / name
        b = {n: np.fromfile(d / f"{n}.bin", dtype=np.uint32) for n in plan.BUFFERS}
        e = {n: np.fromfile(d / f"{n}.expected.bin", dtype=np.uint32) for n in b}
        k = json.loads((HERE / f"contracts/{kind}.json").read_text())
        return l, c, b, e, k

    def test_native_cases_and_independent_integer_spec(self):
        for kind in plan.OPERATIONS:
            k = json.loads((HERE / f"contracts/{kind}.json").read_text())
            for name in k["required_cases"]:
                with self.subTest(operation=kind, case=name):
                    l, c, b, e, k = self.fixture(kind, name)
                    plan.check_outputs(l, c, b, e, e, k)

    def test_clamping_does_not_overflow_ceil_division(self):
        l, c, b, e, k = self.fixture(name="max-capacity")
        self.assertEqual(int(e["Plan"][l["plan"]["bound"]]), 2**32 - 2)
        self.assertEqual(int(e["ContactsIndirect"][0]), 2**26)
        np.testing.assert_array_equal(e["PairCount"], b["PairCount"])
        np.testing.assert_array_equal(e["PfmCount"], b["PfmCount"])

    def test_empty_and_zero_capacity_dispatches(self):
        for name in ["empty", "zero-capacity"]:
            l, c, b, e, k = self.fixture(name=name)
            for n in ["ContactsIndirect", "PfmIndirect"]:
                np.testing.assert_array_equal(e[n][:3], [0, 1, 1])
            np.testing.assert_array_equal(e["MbIndirect"][:3], [0, 3, 1])

    def test_every_output_word_and_guard_are_exact(self):
        for n in plan.BUFFERS:
            for index in [0, -1]:
                l, c, b, e, k = self.fixture()
                a = {n: v.copy() for n, v in e.items()}
                a[n][index] ^= 1
                with (
                    self.subTest(buffer=n, index=index),
                    self.assertRaises(AssertionError),
                ):
                    plan.check_outputs(l, c, b, a, e, k)

    def test_reset_length_and_guard(self):
        for count in [0, 1, 63, 64, 65, 129]:
            l, c, b, e, k = self.fixture("reset_narrow", f"n{count}")
            np.testing.assert_array_equal(
                e["PfmCount"][:count], np.zeros(count, dtype=np.uint32)
            )
            np.testing.assert_array_equal(e["PfmCount"][count:], b["PfmCount"][count:])
            a = {n: v.copy() for n, v in e.items()}
            a["PfmCount"][count] ^= 1
            with self.assertRaises(AssertionError):
                plan.check_outputs(l, c, b, a, e, k)

    def test_unsafe_capacities_launches_and_shapes_are_rejected(self):
        for mutation in [
            "capacity",
            "overflow",
            "launch",
            "iterations",
            "size",
            "dtype",
        ]:
            l, c, b, e, k = self.fixture()
            if mutation == "capacity":
                b["BatchIds"][l["batch"]["contacts_capacity"]] += 1
            if mutation == "overflow":
                b["BatchIds"][l["batch"]["collision_pairs_capacity"]] = 2**31
            if mutation == "launch":
                c["groups"] = 2
            if mutation == "iterations":
                c["iterations"] = 2
            if mutation == "size":
                b["PfmCount"] = b["PfmCount"][:-1]
            if mutation == "dtype":
                b["Plan"] = b["Plan"].astype(np.int32)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                plan.validate_case(l, c, b)
