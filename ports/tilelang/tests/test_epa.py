"""EPA regressions distinguish nonunique valid witnesses from lost penetration."""

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epa_validation import BUFFERS, OPERATIONS, check_outputs, validate_case
from replay import HERE


class EpaTests(unittest.TestCase):
    def fixture(self, op="epa_closest", name="n343"):
        root = Path(
            os.environ.get(
                "NEXUS_TILELANG_EPA_FIXTURES", HERE / "results" / op / "c/fixtures"
            )
        )
        d = root / op / name
        l = json.loads((root / "layout.json").read_text())
        c = json.loads((d / "case.json").read_text())
        k = json.loads((HERE / "contracts" / (op + ".json")).read_text())
        b = {n: np.fromfile(d / (n + ".bin"), np.uint32) for n in BUFFERS}
        e = {
            n: np.fromfile(d / (n + ".expected.bin"), np.uint32)
            if n in k["outputs"]
            else v.copy()
            for n, v in b.items()
        }
        return l, c, b, e, k

    def test_all_native_cases(self):
        for op in OPERATIONS:
            for n in [0, 1, 49, 65, 129, 343, "point_touch"]:
                with self.subTest(op=op, n=n):
                    l, c, b, e, k = self.fixture(
                        op, n if isinstance(n, str) else "n" + str(n)
                    )
                    check_outputs(l, c, b, e, e, k)

    def test_native_point_simplex_fallback_is_explicit(self):
        l, c, b, e, k = self.fixture("epa_closest", "point_touch")
        self.assertEqual(b["State"][l["simplex"]["dim"]], 0)
        np.testing.assert_array_equal(
            e["Output"][:9].view(np.float32), [0, 0, 0, 0, 0, 0, 0, 1, 0]
        )
        check_outputs(l, c, b, e, e, k)
        e["Output"].view(np.float32)[0] = 0.5
        with self.assertRaises(AssertionError):
            check_outputs(l, c, b, e, e, k)

    def test_tangential_contact_witnesses_are_nonunique(self):
        l, c, b, e, k = self.fixture()
        a = {n: v.copy() for n, v in e.items()}
        row = a["Output"].reshape(-1, l["support_output_words"])[106].view(np.float32)
        # Both contact points move tangentially inside overlapping cuboid faces.
        row[l["support_output"]["a"]] += np.float32(0.1)
        row[l["support_output"]["b"]] += np.float32(0.1)
        errors = check_outputs(l, c, b, a, e, k)
        self.assertGreater(errors["Output"], 0.09)

    def test_opposite_normals_require_actual_central_symmetry(self):
        l, c, b, e, k = self.fixture()
        a = {n: v.copy() for n, v in e.items()}
        row = a["Output"].reshape(-1, l["support_output_words"])[159].view(np.float32)
        row[:9] *= -1
        check_outputs(l, c, b, a, e, k)
        a = {n: v.copy() for n, v in e.items()}
        row = a["Output"].reshape(-1, l["support_output_words"])[106].view(np.float32)
        row[:9] *= -1
        with self.assertRaises(AssertionError):
            check_outputs(l, c, b, a, e, k)

    def test_matching_false_zero_depth_is_rejected(self):
        l, c, b, e, k = self.fixture()
        row = e["Output"].reshape(-1, l["support_output_words"])[159].view(np.float32)
        row[:6] = 0
        with self.assertRaisesRegex(AssertionError, "support bound"):
            check_outputs(l, c, b, e, e, k)

    def test_invalid_penetration_geometry_is_rejected(self):
        for mutation in ("outside", "normal", "depth", "tangent", "nan", "status"):
            l, c, b, e, k = self.fixture()
            row = e["Output"].reshape(-1, l["support_output_words"])[106]
            f = row.view(np.float32)
            if mutation == "outside":
                f[0] += 10
            if mutation == "normal":
                f[6:9] *= 2
            if mutation == "depth":
                f[2] -= 0.1
            if mutation == "tangent":
                f[0] += 0.1
            if mutation == "nan":
                f[0] = np.nan
            if mutation == "status":
                row[l["support_output"]["status"]] = 2
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                check_outputs(l, c, b, e, e, k)

    def test_invalid_simplex_seed_and_flags(self):
        for mutation in (
            "dimension",
            "surface",
            "decomposition",
            "origin",
            "active",
            "exact",
        ):
            l, c, b, _, _ = self.fixture()
            row = b["State"].reshape(-1, l["simplex_words"])[159]
            if mutation == "dimension":
                row[l["simplex"]["dim"]] = 4
            if mutation == "surface":
                row.view(np.float32)[3] += 0.1
                row.view(np.float32)[0] += 0.1
            if mutation == "decomposition":
                row.view(np.float32)[0] += 0.1
            if mutation == "origin":
                row[l["simplex"]["dim"]] = 0
            if mutation in ("active", "exact"):
                key = "epa_active" if mutation == "active" else "exact"
                b["Inputs"].reshape(-1, l["support_input_words"])[
                    159, l["support_input"][key]
                ] = 2
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_case(l, c, b)

    def test_guards_readonly_state_and_status_are_exact(self):
        for name in BUFFERS:
            l, c, b, e, k = self.fixture("epa_closest", "n1")
            a = {n: v.copy() for n, v in e.items()}
            a[name][-1] ^= 1
            with self.subTest(buffer=name), self.assertRaises(AssertionError):
                check_outputs(l, c, b, a, e, k)
        l, c, b, e, k = self.fixture()
        a = {n: v.copy() for n, v in e.items()}
        a["State"][159 * l["simplex_words"]] ^= 1
        with self.assertRaises(AssertionError):
            check_outputs(l, c, b, a, e, k)
        a = {n: v.copy() for n, v in e.items()}
        a["Output"][159 * l["support_output_words"] + l["support_output"]["status"]] = 0
        with self.assertRaises(AssertionError):
            check_outputs(l, c, b, a, e, k)


if __name__ == "__main__":
    unittest.main()
