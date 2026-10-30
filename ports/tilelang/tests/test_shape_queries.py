"""Public Shape wrapper regressions, including the exact native ball singularity."""

import copy
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay import HERE
from shape_query_replay import BUFFERS, OPERATIONS, check_outputs, validate_case


class ShapeQueryTests(unittest.TestCase):
    def fixture(self, op="shape_project_local_boundary", name="n129"):
        root = Path(
            os.environ.get(
                "NEXUS_TILELANG_SHAPE_QUERY_FIXTURES",
                Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
                / "shape_project_local/c/fixtures",
            )
        )
        l = json.loads((root / "layout.json").read_text())
        c = next(
            x
            for x in json.loads((root / op / "suite.json").read_text())["cases"]
            if x["name"] == name
        )
        k = json.loads((HERE / f"contracts/{op}.json").read_text())
        d = root / op / name
        b = {n: np.fromfile(d / f"{n}.bin", np.uint32) for n in BUFFERS}
        e = {
            n: np.fromfile(d / f"{n}.expected.bin", np.uint32)
            if n == "Results"
            else v.copy()
            for n, v in b.items()
        }
        return l, c, b, e, k

    def test_all_native_geometry(self):
        for op in OPERATIONS:
            for n in (0, 1, 31, 32, 33, 63, 64, 65, 129):
                with self.subTest(op=op, n=n):
                    l, c, b, e, k = self.fixture(op, f"n{n}")
                    check_outputs(l, c, b, e, e, k)

    def test_invalid_input(self):
        for mutation in ("index", "pose", "dimension", "count", "point"):
            l, c, b, _, _ = self.fixture()
            if mutation == "index":
                b["Queries"][l["query"]["shape_index"]] = c["shapes"]
            elif mutation == "pose":
                b["Poses"][: l["pose_words"]] = 0
            elif mutation == "dimension":
                b["Shapes"].view(np.float32)[l["shape"]["a"]] = -1.0
            elif mutation == "count":
                b["Count"][0] += 1
            else:
                b["Queries"][l["query"]["point"]] = 0x7FC00000
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_case(l, c, b)

    def test_singularity_is_required_only_at_ball_center(self):
        l, c, b, e, k = self.fixture()
        self.assertTrue(np.isnan(e["Results"][:3].view(np.float32)).all())
        for index, value in ((0, 0.0), (l["result_words"], float("nan"))):
            bad = copy.deepcopy(e)
            bad["Results"].view(np.float32)[index] = value
            with self.subTest(index=index), self.assertRaises(AssertionError):
                check_outputs(l, c, b, bad, bad, k)

    def test_exact_flags_fallback_and_guards(self):
        l, c, b, e, k = self.fixture()
        for name, index in (
            ("Results", l["result"]["inside"]),
            ("Results", l["result"]["padding"]),
            ("Results", 6 * l["result_words"]),
            ("Results", c["count"] * l["result_words"]),
            ("Shapes", 0),
            ("Count", 1),
        ):
            a = copy.deepcopy(e)
            a[name][index] ^= 1
            with (
                self.subTest(name=name, index=index),
                self.assertRaises(AssertionError),
            ):
                check_outputs(l, c, b, a, e, k)

    def test_both_wrong_results_fail_independent_distance(self):
        for op in OPERATIONS:
            l, c, b, e, k = self.fixture(op)
            e["Results"].view(np.float32)[l["result_words"]] += 0.2
            with (
                self.subTest(op=op),
                self.assertRaisesRegex(AssertionError, "Independent shape"),
            ):
                check_outputs(l, c, b, e, e, k)


if __name__ == "__main__":
    unittest.main()
