"""Scoped numerical budgets never replace exact identities or independent physics."""

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scene_numeric as numeric
from general_scene_validation import check_outputs
from replay import HERE


class SceneNumericTests(unittest.TestCase):
    def fixture(self):
        root = (
            Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
            / "general_scene/metal/fixtures"
        )
        d = root / "general_scene/general_triangle-c1-l1-b1-f120"
        l = json.loads((root / "layout.json").read_text())
        c = json.loads((d / "case.json").read_text())
        k = json.loads((HERE / "contracts/general_scene.json").read_text())
        b = {n: np.fromfile(d / f"{n}.bin", np.uint32) for n in c["buffer_words"]}
        e = {
            n: np.fromfile(d / f"{n}.expected.bin", np.uint32)
            if n in k["outputs"]
            else v.copy()
            for n, v in b.items()
        }
        return l, c, b, e, k

    def test_default_and_unknown_fields_remain_strict(self):
        _l, c, _b, _e, k = self.fixture()
        for case, field in [
            (dict(c, name="unlisted"), "Poses"),
            (c, "Params"),
            (c, "TileMaterial"),
        ]:
            with self.subTest(field=field), self.assertRaises(AssertionError):
                numeric.check(case, k, field, np.array([0.0001]), np.array([0.0]))

    def test_policy_cannot_be_reused_for_other_configuration(self):
        _l, c, _b, _e, k = self.fixture()
        for field, value in [
            ("frames", 1),
            ("batches", 2),
            ("body_count", 3),
            ("scene", "general_cylinder"),
        ]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                numeric.tolerances(dict(c, **{field: value}), k, "Poses")

    def test_matching_wrong_coefficients_fail_independent_check(self):
        for field in ("normal_mass", "tangent_k", "coupling"):
            l, c, b, e, k = self.fixture()
            slot = int(e["Indices"][0])
            base = slot // 32 * l["tile_words"] + slot % 32 * 4
            point = base + l["points"]
            off = (
                point + l["point"]["arm_b"] + 3
                if field == "normal_mass"
                else point + l["point"]["friction"] + 2
                if field == "tangent_k"
                else base + l["coupling"]
            )
            e["Tiles"].view(np.float32)[off] += 0.001
            e["OldTiles"] = e["Tiles"].copy()
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(AssertionError, "effective response"),
            ):
                check_outputs(l, c, b, e, e, k)

    def test_matching_nonorthogonal_basis_is_rejected(self):
        l, c, b, e, k = self.fixture()
        slot = int(e["Indices"][0])
        base = slot // 32 * l["tile_words"] + slot % 32 * 4
        off = base + l["header"]["tangent"]
        e["Tiles"].view(np.float32)[off : off + 3] *= 1.001
        e["OldTiles"] = e["Tiles"].copy()
        with self.assertRaisesRegex(AssertionError, "Orthonormal"):
            check_outputs(l, c, b, e, e, k)

    def test_matching_negative_inertia_is_rejected(self):
        l, c, b, e, k = self.fixture()
        slot = int(e["Indices"][0])
        base = slot // 32 * l["tile_words"] + slot % 32 * 4
        e["Tiles"].view(np.float32)[base + l["header"]["diag_b"]] = -1
        e["OldTiles"] = e["Tiles"].copy()
        with self.assertRaisesRegex(AssertionError, "inertia"):
            check_outputs(l, c, b, e, e, k)

    def test_trajectory_budget_still_rejects_excess_error(self):
        l, c, b, e, k = self.fixture()
        a = {n: v.copy() for n, v in e.items()}
        off = 50 * c["trace_stride"] + l["pose_words"] + l["pose_translation"]
        a["Trajectory"].view(np.float32)[off] += 0.001
        with self.assertRaisesRegex(AssertionError, "TrajectoryPoses"):
            check_outputs(l, c, b, a, e, k)

    def test_required_restart_evidence_is_registered(self):
        registry = json.loads((HERE / "parity.json").read_text())
        by_op = {f["operation"]: f for f in registry["features"]}
        for op in ("general_scene", "joint_scene"):
            k = json.loads((HERE / f"contracts/{op}.json").read_text())
            for policy in k["numerical_cases"].values():
                feature = by_op[policy["restart_operation"]]
                restart = json.loads(
                    (HERE / "contracts" / Path(feature["contract"]).name).read_text()
                )
                self.assertEqual(
                    set(policy["required_restarts"]), set(restart["required_cases"])
                )


if __name__ == "__main__":
    unittest.main()
