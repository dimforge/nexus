"""Convex support optimality and nonunique valid GJK witness regressions."""

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay import HERE
from support_validation import BUFFERS, OPERATIONS, check_outputs, validate_case


class SupportTests(unittest.TestCase):
    def fixture(self, operation="gjk_closest", name="n343"):
        root = Path(
            os.environ.get(
                "NEXUS_TILELANG_SUPPORT_FIXTURES",
                Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
                / operation
                / "c/fixtures",
            )
        )
        d = root / operation / name
        l = json.loads((root / "layout.json").read_text())
        c = json.loads((d / "case.json").read_text())
        k = json.loads((HERE / "contracts" / (operation + ".json")).read_text())
        b = {n: np.fromfile(d / (n + ".bin"), np.uint32) for n in BUFFERS}
        e = {
            n: np.fromfile(d / (n + ".expected.bin"), np.uint32)
            if n in k["outputs"]
            else v.copy()
            for n, v in b.items()
        }
        return l, c, b, e, k

    def test_all_native_cases(self):
        for operation in OPERATIONS:
            for count in [0, 1, 49, 65, 129, 343]:
                with self.subTest(operation=operation, count=count):
                    l, c, b, e, k = self.fixture(operation, "n" + str(count))
                    check_outputs(l, c, b, e, e, k)

    def test_scaled_floor_contact_and_exact_cso_construction(self):
        l, c, b, e, k = self.fixture("gjk_closest", "floor_cylinder")
        check_outputs(l, c, b, e, e, k)
        rows = e["Output"].reshape(-1, l["support_output_words"])
        np.testing.assert_array_equal(
            rows[:8, l["support_output"]["status"]], [0, 1] * 4
        )
        changed = {name: value.copy() for name, value in e.items()}
        # One ULP of fabricated CSO geometry must fail, even for the largest pair.
        off = 6 * l["simplex_words"] + l["simplex"]["vertices"]
        changed["State"][off] ^= 1
        with self.assertRaisesRegex(AssertionError, "CSO decomposition"):
            check_outputs(l, c, b, changed, e, k)

    def test_invalid_shape_pose_or_direction(self):
        for mutation in ("tag", "radius", "pose", "direction", "exact", "seed_b"):
            l, c, b, _, _ = self.fixture("gjk_closest", "n1")
            if mutation == "tag":
                b["Shapes"][l["shape"]["a"] + 3] = 0xFFFFFFFF
            if mutation == "radius":
                b["Shapes"].view(np.float32)[l["shape"]["a"]] = -1
            if mutation == "pose":
                b["Poses"].view(np.float32)[l["pose_rotation"] + 3] = 2
            if mutation == "direction":
                b["Inputs"][
                    l["support_input"]["direction"] : l["support_input"]["direction"]
                    + 3
                ] = 0
            if mutation == "exact":
                b["Inputs"][l["support_input"]["exact"]] = 2
            if mutation == "seed_b":
                f = b["State"].view(np.float32)
                off = l["simplex"]["vertices"]
                f[off + 6] += np.float32(0.1)
                f[off] = f[off + 3] - f[off + 6]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_case(l, c, b)

    def test_capacity_dispatch_and_guards(self):
        for field in ("count", "storage", "groups", "iterations"):
            l, c, b, _, _ = self.fixture("gjk_closest", "n1")
            c[field] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_case(l, c, b)
        for name in BUFFERS:
            l, c, b, e, k = self.fixture("gjk_closest", "n1")
            a = {n: v.copy() for n, v in e.items()}
            a[name][-1] ^= 1
            with self.subTest(buffer=name), self.assertRaises(AssertionError):
                check_outputs(l, c, b, a, e, k)

    def test_matching_nonoptimal_support_rejected(self):
        for op in ("support_local", "support_world", "cso_support"):
            l, c, b, e, k = self.fixture(op, "n1")
            off = l["support_output"]["a"]
            e["Output"].view(np.float32)[off] += np.float32(0.1)
            with self.subTest(operation=op), self.assertRaises(AssertionError):
                check_outputs(l, c, b, e, e, k)

    def test_nonintersection_preserves_copied_bits_even_within_float_tolerance(self):
        l, c, b, e, k = self.fixture("gjk_closest", "n1")
        a = {n: v.copy() for n, v in e.items()}
        a["State"][l["simplex"]["vertices"]] ^= 1
        with self.assertRaisesRegex(AssertionError, "preserves supplied simplex"):
            check_outputs(l, c, b, a, e, k)

    def test_valid_nonunique_intersection_simplex_is_accepted(self):
        l, c, b, e, k = self.fixture()
        a = {n: v.copy() for n, v in e.items()}
        index = 158
        ss = l["simplex"]
        state = a["State"].reshape(-1, l["simplex_words"])
        self.assertEqual(state[index, ss["dim"]], 3)
        point = (
            state[
                index,
                ss["vertices"] + 3 * l["cso_words"] : ss["vertices"]
                + 4 * l["cso_words"],
            ]
            .view(np.float32)
            .reshape(3, 3)
        )
        angle = 0.001
        x, z = float(point[1, 0]), float(point[1, 2])
        point[1, 0] = np.cos(angle) * x - np.sin(angle) * z
        point[1, 2] = np.sin(angle) * x + np.cos(angle) * z
        point[0] = point[1] - point[2]
        errors = check_outputs(l, c, b, a, e, k)
        self.assertGreater(errors["State"], 1e-5)

    def test_invalid_intersection_geometry_cannot_hide_in_semantic_comparison(self):
        for mutation in ("decomposition", "surface", "index"):
            l, c, b, e, k = self.fixture()
            a = {n: v.copy() for n, v in e.items()}
            index = 158
            ss = l["simplex"]
            row = a["State"].reshape(-1, l["simplex_words"])[index]
            point = (
                row[ss["vertices"] : ss["vertices"] + l["cso_words"]]
                .view(np.float32)
                .reshape(3, 3)
            )
            if mutation == "decomposition":
                point[0, 0] += np.float32(0.1)
            if mutation == "surface":
                point[1] *= 2
                point[0] = point[1] - point[2]
            if mutation == "index":
                row[ss["previous_vertices"]] ^= 1
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                check_outputs(l, c, b, a, e, k)


if __name__ == "__main__":
    unittest.main()
