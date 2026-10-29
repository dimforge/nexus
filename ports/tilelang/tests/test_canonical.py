"""Canonical contact identity and stable ordering independent of native output."""

import json, os, sys, unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay import HERE
import canonical_replay as canonical


class CanonicalTests(unittest.TestCase):
    def fixture(self, kind="canonical_contacts", name="n65-duplicates-s7"):
        root = (
            Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
            / kind
            / "c/fixtures"
        )
        l = json.loads((root / "layout.json").read_text())
        cases = json.loads((root / kind / "suite.json").read_text())["cases"]
        c = next(c for c in cases if c["name"] == name)
        d = root / kind / name
        b = {n: np.fromfile(d / f"{n}.bin", np.uint32) for n in canonical.BUFFERS}
        e = {n: np.fromfile(d / f"{n}.expected.bin", np.uint32) for n in b}
        k = json.loads((HERE / f"contracts/{kind}.json").read_text())
        return l, c, b, e, k

    def test_all_native_canonical_cases_match_independent_specification(self):
        for op in canonical.OPERATIONS:
            k = json.loads((HERE / f"contracts/{op}.json").read_text())
            for name in k["required_cases"]:
                with self.subTest(operation=op, case=name):
                    l, c, b, e, k = self.fixture(op, name)
                    canonical.check_outputs(l, c, b, e, e, k)

    def test_duplicate_identity_is_stable_and_inert_output_is_default(self):
        l, c, b, e, k = self.fixture()
        rows = b["Contacts"].reshape(-1, l["contact_words"])
        length = l["contact"]["contact"] + l["manifold"]["len"]
        live = [i for i in range(c["count"]) if rows[i, length]]
        inert = [i for i in range(c["count"]) if not rows[i, length]]
        np.testing.assert_array_equal(e["InOrder"][: c["count"]], live + inert)
        out = e["Output"].reshape(-1, l["contact_words"])
        np.testing.assert_array_equal(out[: len(live)], rows[live])
        np.testing.assert_array_equal(
            out[len(live) : c["count"]],
            np.zeros((len(inert), l["contact_words"]), np.uint32),
        )
        a = {n: v.copy() for n, v in e.items()}
        a["InOrder"][[0, 1]] = a["InOrder"][[1, 0]]
        with self.assertRaises(AssertionError):
            canonical.check_outputs(l, c, b, a, a, k)

    def test_all_integer_fields_scratch_and_guards_stay_exact(self):
        for name in canonical.BUFFERS:
            for index in [0, -1]:
                l, c, b, e, k = self.fixture()
                a = {n: v.copy() for n, v in e.items()}
                a[name][index] ^= 1
                with (
                    self.subTest(buffer=name, index=index),
                    self.assertRaises(AssertionError),
                ):
                    canonical.check_outputs(l, c, b, a, e, k)

    def test_packed_and_multifield_schedules_have_same_order(self):
        _, _, _, a, _ = self.fixture(name="n65-packed-s7")
        _, _, _, b, _ = self.fixture(name="n65-two-fields-s7")
        # These fixtures have no repeated pair IDs, so the two native schedules agree.
        np.testing.assert_array_equal(a["Output"], b["Output"])
        np.testing.assert_array_equal(a["InOrder"], b["InOrder"])

    def test_invalid_permutation_plan_packing_and_scope_fail(self):
        for mutation in [
            "order",
            "plan",
            "selector",
            "packing",
            "length",
            "passes",
            "uniform",
            "capacity",
            "dtype",
            "shape",
        ]:
            l, c, b, e, k = self.fixture(name="n65-packed-s7")
            if mutation == "order":
                b["InOrder"][0] = c["count"]
            if mutation == "plan":
                b["Plan"][l["plan"]["bound"]] += 1
            if mutation == "selector":
                b["Selector"][0] = 8
                c["selector"] = 8
            if mutation == "packing":
                b["BatchIds"][l["batch"]["contact_sort_collider_shift"]] = 1
            if mutation == "length":
                b["Contacts"][l["contact"]["contact"] + l["manifold"]["len"]] = 5
            if mutation == "passes":
                c["passes"] = [2, 1]
            if mutation == "uniform":
                b["Config3"][l["sort_uniforms"]["shift"]] = 13
            if mutation == "capacity":
                c["storage"] = 64 * 1024 * 1024 + 1
            if mutation == "dtype":
                b["Keys"] = b["Keys"].astype(np.int32)
            if mutation == "shape":
                b["Contacts"] = b["Contacts"][:-1]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                canonical.validate_case(l, c, b)
