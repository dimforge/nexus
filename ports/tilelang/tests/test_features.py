"""Support-feature and polygonal-clipping ABI and geometric regressions."""

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feature_validation import CLIP_BUFFERS, FACE_BUFFERS, check_outputs, validate_case
from replay import HERE


class FeatureTests(unittest.TestCase):
    def fixture(self, op="support_face", name="n168"):
        root = Path(
            os.environ.get(
                "NEXUS_TILELANG_FEATURE_FIXTURES", HERE / "results" / op / "c/fixtures"
            )
        )
        d = root / op / name
        l = json.loads((root / "layout.json").read_text())
        c = json.loads((d / "case.json").read_text())
        k = json.loads((HERE / "contracts" / (op + ".json")).read_text())
        b = {
            n: np.fromfile(d / (n + ".bin"), np.uint32)
            for n in FACE_BUFFERS
            if op == "support_face"
        }
        if op != "support_face":
            b = {n: np.fromfile(d / (n + ".bin"), np.uint32) for n in CLIP_BUFFERS}
        e = {
            n: np.fromfile(d / (n + ".expected.bin"), np.uint32)
            if n == "Output"
            else v.copy()
            for n, v in b.items()
        }
        return l, c, b, e, k

    def test_all_native_cases(self):
        for op, counts in [
            ("support_face", [0, 1, 7, 33, 65, 129, 168]),
            ("polygonal_contacts", [0, 1, 25, 65, 129, 300]),
        ]:
            for n in counts:
                with self.subTest(op=op, n=n):
                    l, c, b, e, k = self.fixture(op, "n" + str(n))
                    check_outputs(l, c, b, e, e, k)

    def test_invalid_index_ranges_and_degenerate_hull(self):
        for mutation in ("first", "last", "index", "degenerate"):
            l, c, b, _, _ = self.fixture()
            shape = b["Shapes"].reshape(-1, l["shape_words"])[5]
            offset = l["shape"]["b"]
            first, last = map(int, shape[offset : offset + 2])
            if mutation == "first":
                shape[offset] = last
            if mutation == "last":
                shape[offset + 1] = len(b["Indices"]) + 1
            if mutation == "index":
                b["Indices"][first] = 99
            if mutation == "degenerate":
                b["Indices"][first:last] = 0
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_case(l, c, b)

    def test_copied_feature_vertices_are_bit_exact(self):
        l, c, b, e, k = self.fixture()
        a = {n: v.copy() for n, v in e.items()}
        a["Output"][6 * l["feature_words"] + l["feature"]["vertices"]] ^= 1
        with self.assertRaisesRegex(AssertionError, "copied feature"):
            check_outputs(l, c, b, a, e, k)

    def test_matching_bad_support_geometry_is_rejected(self):
        l, c, b, e, k = self.fixture()
        e["Output"].view(np.float32)[l["feature_words"]] = 5
        with self.assertRaisesRegex(AssertionError, "outside its shape"):
            check_outputs(l, c, b, e, e, k)

    def test_matching_bad_clipped_geometry_is_rejected(self):
        for mutation in ("position", "distance", "normal"):
            l, c, b, e, k = self.fixture("polygonal_contacts", "n300")
            rows = e["Output"].reshape(-1, l["feature_manifold_words"])
            fields = l["feature_manifold"]
            i = int(np.flatnonzero(rows[:, fields["count"]] > 0)[0])
            f = rows[i].view(np.float32)
            if mutation == "position":
                f[fields["points"]] += 10
            if mutation == "distance":
                f[fields["points"] + l["feature_point"]["dist"]] = 1
            if mutation == "normal":
                f[fields["normal"]] = 1e-20
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                check_outputs(l, c, b, e, e, k)

    def test_invalid_clipping_inputs(self):
        for mutation in ("count", "axis", "pose", "prediction", "flipped"):
            l, c, b, _, _ = self.fixture("polygonal_contacts", "n25")
            if mutation == "count":
                b["Features"][l["feature"]["count"]] = 5
            if mutation == "axis":
                b["Inputs"].view(np.float32)[l["polygonal_input"]["normal"]] = 1
            if mutation == "pose":
                b["Poses"].view(np.float32)[l["pose_rotation"] + 3] = 2
            if mutation == "prediction":
                b["Inputs"].view(np.float32)[l["polygonal_input"]["prediction"]] = -1
            if mutation == "flipped":
                b["Inputs"][l["polygonal_input"]["flipped"]] = 2
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_case(l, c, b)

    def test_guards_and_count_are_exact(self):
        for op in ("support_face", "polygonal_contacts"):
            l, c, b, e, k = self.fixture(op, "n1")
            for name in b:
                a = {n: v.copy() for n, v in e.items()}
                a[name][-1] ^= 1
                with self.subTest(op=op, name=name), self.assertRaises(AssertionError):
                    check_outputs(l, c, b, a, e, k)
            a = {n: v.copy() for n, v in e.items()}
            field = l["feature"] if op == "support_face" else l["feature_manifold"]
            a["Output"][field["count"]] ^= 1
            with self.assertRaises(AssertionError):
                check_outputs(l, c, b, a, e, k)


if __name__ == "__main__":
    unittest.main()
