"""Geometry ABI, exact simplex state and independent projection regressions."""

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geometry_validation import OPERATIONS, buffers, check_outputs, validate_case
from replay import HERE


class GeometryTests(unittest.TestCase):
    def fixture(self, operation="simplex_project", name="n24"):
        root = Path(
            os.environ.get(
                "NEXUS_TILELANG_GEOMETRY_FIXTURES",
                Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
                / operation
                / "c/fixtures",
            )
        )
        l = json.loads((root / "layout.json").read_text())
        d = root / operation / name
        c = json.loads((d / "case.json").read_text())
        k = json.loads((HERE / "contracts" / (operation + ".json")).read_text())
        b = {n: np.fromfile(d / (n + ".bin"), np.uint32) for n in buffers(operation)}
        e = {
            n: np.fromfile(d / (n + ".expected.bin"), np.uint32)
            if n in k["outputs"]
            else v.copy()
            for n, v in b.items()
        }
        return l, c, b, e, k

    def test_all_live_native_fixtures(self):
        for operation in OPERATIONS:
            for count in [0, 1, 24, 33, 65, 129]:
                with self.subTest(operation=operation, count=count):
                    l, c, b, e, k = self.fixture(operation, "n" + str(count))
                    check_outputs(l, c, b, e, e, k)

    def test_capacity_and_dispatch_rejected(self):
        for field in ("count", "storage", "groups", "iterations"):
            l, c, b, _, _ = self.fixture()
            c[field] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_case(l, c, b)

    def test_nonfinite_input_and_dimension_rejected(self):
        for mutation in ("point", "dim", "previous_index"):
            l, c, b, _, _ = self.fixture("simplex_add")
            s = l["simplex"]
            if mutation == "point":
                b["State"].view(np.float32)[s["vertices"]] = np.nan
            if mutation == "dim":
                b["State"][s["dim"]] = 3
            if mutation == "previous_index":
                b["State"][s["previous_vertices"]] = 4
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_case(l, c, b)
        l, c, b, _, _ = self.fixture("triangle_projection")
        b["Inputs"][l["projection_input"]["solid"]] = 2
        with self.assertRaises(ValueError):
            validate_case(l, c, b)

    def test_exact_features_copied_state_and_guards(self):
        for operation in OPERATIONS:
            l, c, b, e, k = self.fixture(operation)
            name = "State" if operation.startswith("simplex_") else "Output"
            offsets = (
                [-1, l["simplex"]["previous_vertices"]]
                if name == "State"
                else [-1, l["projection_output"]["feature"]]
            )
            for offset in offsets:
                a = {n: v.copy() for n, v in e.items()}
                a[name][offset] ^= 1
                with (
                    self.subTest(operation=operation, offset=offset),
                    self.assertRaises(AssertionError),
                ):
                    check_outputs(l, c, b, a, e, k)

    def test_rejected_duplicate_preserves_all_native_state(self):
        l, c, b, e, k = self.fixture("simplex_add", "n1")
        np.testing.assert_array_equal(b["State"], e["State"])
        a = {n: v.copy() for n, v in e.items()}
        a["State"][l["simplex"]["vertices"] : l["simplex"]["vertices"] + 3] = 0
        with self.assertRaisesRegex(AssertionError, "Exact geometry state"):
            check_outputs(l, c, b, a, e, k)

    def test_matching_wrong_projection_rejected_independently(self):
        for operation in (
            "segment_projection",
            "triangle_projection",
            "tetrahedron_projection",
            "simplex_project",
        ):
            l, c, b, e, k = self.fixture(operation, "n1")
            name = "State" if operation == "simplex_project" else "Output"
            offset = (
                l["simplex"]["projected"]
                if name == "State"
                else l["projection_output"]["point"]
            )
            e[name].view(np.float32)[offset] += np.float32(0.1)
            with (
                self.subTest(operation=operation),
                self.assertRaisesRegex(AssertionError, "Independent closest-point"),
            ):
                check_outputs(l, c, b, e, e, k)

    def test_matching_invalid_barycentrics_rejected(self):
        l, c, b, e, k = self.fixture("simplex_project", "n1")
        e["State"].view(np.float32)[l["simplex"]["projection"]] = np.float32(0.5)
        with self.assertRaisesRegex(AssertionError, "barycentric"):
            check_outputs(l, c, b, e, e, k)

    def test_readonly_input_preserved(self):
        l, c, b, e, k = self.fixture("triangle_projection")
        a = {n: v.copy() for n, v in e.items()}
        a["Inputs"][0] ^= 1
        with self.assertRaisesRegex(AssertionError, "Readonly"):
            check_outputs(l, c, b, a, e, k)


if __name__ == "__main__":
    unittest.main()
