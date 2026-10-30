"""Query safety and independent-geometry regressions using fresh native fixtures."""

import copy
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from point_query_replay import BUFFERS, OPERATIONS, check_outputs, validate_case
from replay import HERE


class PointQueryTests(unittest.TestCase):
    def fixture(self, op="mesh_projection", name="n129"):
        root = Path(
            os.environ.get(
                "NEXUS_TILELANG_POINT_QUERY_FIXTURES",
                HERE / "results-point-queries-0098/mesh_projection/c/fixtures",
            )
        )
        l = json.loads((root / "layout.json").read_text())
        folder = root / op
        c = next(
            x
            for x in json.loads((folder / "suite.json").read_text())["cases"]
            if x["name"] == name
        )
        k = json.loads((HERE / f"contracts/{op}.json").read_text())
        d = folder / name
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

    def test_invalid_traversal_and_vertices(self):
        for mutation in ("cycle", "escape", "triangle", "vertex", "finite", "count"):
            l, c, b, _, _ = self.fixture()
            root = c["index_root"]
            if mutation == "cycle":
                b["Indices"][root] = 0
            elif mutation == "escape":
                b["Indices"][root + 1] = c["nodes"] + 1
            elif mutation == "triangle":
                b["Indices"][root + 5] = c["triangles"]
            elif mutation == "vertex":
                b["Indices"][root + c["nodes"] * 3] = c["vertices_count"]
            elif mutation == "finite":
                b["Queries"][l["query"]["point"]] = 0x7FC00000
            else:
                b["Count"][0] += 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_case(l, c, b)

    def test_pseudo_normal_edge_index(self):
        l, c, b, _, _ = self.fixture("mesh_pseudo_normal")
        b["Queries"][l["query"]["feature"]] = 1
        b["Queries"][l["query"]["feature_id"]] = 3
        with self.assertRaises(ValueError):
            validate_case(l, c, b)

    def test_exact_flags_readonly_and_guards(self):
        l, c, b, e, k = self.fixture()
        for name, index in (
            ("Results", l["result"]["found"]),
            ("Results", l["result"]["padding"]),
            ("Results", c["count"] * l["result_words"]),
            ("Vertices", 0),
            ("Count", 1),
        ):
            a = copy.deepcopy(e)
            a[name][index] ^= 1
            with (
                self.subTest(name=name, index=index),
                self.assertRaises(AssertionError),
            ):
                check_outputs(l, c, b, a, e, k)

    def test_both_wrong_outputs_fail_geometry(self):
        for op in OPERATIONS:
            l, c, b, e, k = self.fixture(op)
            e["Results"].view(np.float32)[l["result"]["point"]] += 0.2
            with (
                self.subTest(op=op),
                self.assertRaisesRegex(AssertionError, "Independent query geometry"),
            ):
                check_outputs(l, c, b, e, e, k)


if __name__ == "__main__":
    unittest.main()
