"""PFM parity policy, guarded ABI, and independent geometry regressions."""

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pfm_validation import BUFFERS, check_outputs, diagnostics, validate_case
from replay import HERE


class PfmTests(unittest.TestCase):
    def fixture(self, op="pfm_manifold", n=343):
        root = Path(
            os.environ.get(
                "NEXUS_TILELANG_PFM_FIXTURES", HERE / "results" / op / "c/fixtures"
            )
        )
        d = root / op / (f"n{n}" if isinstance(n, int) else n)
        l = json.loads((root / "layout.json").read_text())
        c = json.loads((d / "case.json").read_text())
        k = json.loads((HERE / "contracts" / f"{op}.json").read_text())
        b = {name: np.fromfile(d / f"{name}.bin", np.uint32) for name in BUFFERS}
        e = {
            name: np.fromfile(d / f"{name}.expected.bin", np.uint32)
            if name in k["outputs"]
            else value.copy()
            for name, value in b.items()
        }
        return l, c, b, e, k

    def test_live_native_cases(self):
        for op in ("pfm_subshape", "pfm_manifold"):
            for n in (0, 1, 49, 65, 129, 343):
                with self.subTest(op=op, n=n):
                    l, c, b, e, k = self.fixture(op, n)
                    check_outputs(l, c, b, e, e, k)

    def test_triangle_fallback_rejects_only_the_invalid_contact(self):
        l, c, b, e, k = self.fixture(n="triangle_fallback")
        check_outputs(l, c, b, e, e, k)
        rows = e["Output"].reshape(-1, l["feature_manifold_words"])
        self.assertEqual(int(rows[0, l["feature_manifold"]["count"]]), 0)
        self.assertGreater(int(rows[1, l["feature_manifold"]["count"]]), 0)

    def test_inactive_dedup_payload_is_not_a_contact(self):
        l, c, b, e, k = self.fixture()
        a = {name: v.copy() for name, v in e.items()}
        row = a["Output"].reshape(-1, l["feature_manifold_words"])[78]
        self.assertEqual(int(row[l["feature_manifold"]["count"]]), 1)
        row.view(np.float32)[
            l["feature_point_words"] : 2 * l["feature_point_words"]
        ] = [9, 8, 7, 6]
        check_outputs(l, c, b, a, e, k)
        row[l["feature_manifold"]["count"]] = 2
        with self.assertRaises(AssertionError):
            check_outputs(l, c, b, a, e, k)

    def test_native_secondary_witness_regressions_are_fixed(self):
        l, c, b, e, k = self.fixture()
        misses = diagnostics(l, c, b, e, e, k)
        self.assertEqual(misses["actual_secondary_containment_misses"], [])
        self.assertEqual(misses["reference_secondary_containment_misses"], [])
        rows = e["Output"].reshape(-1, l["feature_manifold_words"])
        for i in (115, 121, 165, 228):
            self.assertEqual(int(rows[i, l["feature_manifold"]["count"]]), 1)

    def test_matching_bad_secondary_geometry_is_rejected(self):
        l, c, b, e, k = self.fixture()
        # The former capsule/cone edge candidate lies on the capsule, but its
        # implicit second witness misses the cone. Keep the original f32 bits.
        row = e["Output"].reshape(-1, l["feature_manifold_words"])[121]
        row[:4] = [1023907314, 1054707559, 1047452852, 3171626680]
        with self.assertRaisesRegex(AssertionError, "secondary point outside"):
            check_outputs(l, c, b, e, e, k)

    def test_matching_bad_geometry_is_rejected(self):
        for mutation in ("point", "normal", "distance"):
            l, c, b, e, k = self.fixture()
            rows = e["Output"].reshape(-1, l["feature_manifold_words"])
            fields = l["feature_manifold"]
            i = int(np.flatnonzero(rows[:, fields["count"]] > 0)[0])
            f = rows[i].view(np.float32)
            if mutation == "point":
                f[fields["points"]] += 10
            elif mutation == "normal":
                f[fields["normal"] : fields["normal"] + 3] = 0
            else:
                f[fields["points"] + l["feature_point"]["dist"]] = -100
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                check_outputs(l, c, b, e, e, k)

    def test_invalid_inputs(self):
        for mutation in ("shape", "index", "pose", "prediction"):
            l, c, b, _, _ = self.fixture()
            if mutation == "shape":
                b["Shapes"][l["shape"]["a"] + 3] = 99
            elif mutation == "index":
                b["Indices"][3] = 100000
            elif mutation == "pose":
                b["Poses"].view(np.float32)[l["pose_rotation"] + 3] = 2
            else:
                b["Inputs"].view(np.float32)[l["pfm_input"]["prediction"]] = -1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_case(l, c, b)

    def test_guards_and_conversions_are_exact(self):
        for op in ("pfm_subshape", "pfm_manifold"):
            l, c, b, e, k = self.fixture(op, 1)
            for name in b:
                a = {name: v.copy() for name, v in e.items()}
                a[name][-1] ^= 1
                with self.subTest(op=op, name=name), self.assertRaises(AssertionError):
                    check_outputs(l, c, b, a, e, k)
            for name in ("SubShapes", "Radii"):
                a = {name: v.copy() for name, v in e.items()}
                a[name][0] ^= 1
                with self.subTest(op=op, name=name), self.assertRaises(AssertionError):
                    check_outputs(l, c, b, a, e, k)

    def test_nonunique_tangent_and_reflected_points(self):
        l, c, b, e, k = self.fixture()
        a = {name: v.copy() for name, v in e.items()}
        rows = a["Output"].reshape(-1, l["feature_manifold_words"])
        mf = l["feature_manifold"]
        row = rows[78].view(np.float32)
        row[mf["points"] : mf["points"] + 3] = row[
            mf["points"] + l["feature_point_words"] : mf["points"]
            + l["feature_point_words"]
            + 3
        ]
        check_outputs(l, c, b, a, e, k)
        row = rows[106].view(np.float32)
        row[mf["points"]] -= 0.01
        with self.assertRaisesRegex(AssertionError, "tangent drift"):
            check_outputs(l, c, b, a, e, k)
        a = {name: v.copy() for name, v in e.items()}
        row = a["Output"].reshape(-1, l["feature_manifold_words"])[182].view(np.float32)
        row[mf["points"] : mf["normal"]].reshape(4, l["feature_point_words"])[
            :, :3
        ] *= -1
        row[mf["normal"] : mf["normal"] + 3] *= -1
        check_outputs(l, c, b, a, e, k)
        # A sign flip at noncoincident origins has no symmetry exemption.
        a = {name: v.copy() for name, v in e.items()}
        row = a["Output"].reshape(-1, l["feature_manifold_words"])[78].view(np.float32)
        row[mf["normal"] : mf["normal"] + 3] *= -1
        with self.assertRaises(AssertionError):
            check_outputs(l, c, b, a, e, k)

    def test_bad_secondary_witness_does_not_unlock_relaxation(self):
        l, c, b, e, k = self.fixture()
        a = {name: v.copy() for name, v in e.items()}
        row = a["Output"].reshape(-1, l["feature_manifold_words"])[121]
        row[:4] = [1023907314, 1054707559, 1047452852, 3171626680]
        with self.assertRaisesRegex(AssertionError, "secondary point outside"):
            check_outputs(l, c, b, a, e, k)


if __name__ == "__main__":
    unittest.main()
