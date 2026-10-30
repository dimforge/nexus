"""Scoped compact-friction budgets preserve history, exact identity and geometry."""

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scene_numeric
from cpu_oracle import verify_probe_file
from joint_scene_validation import check_joint_rows
from replay import HERE
from runtime_restart_validation import check_outputs, validate_case
from runtime_scene_physics import check as check_physics


class RuntimeRestartTests(unittest.TestCase):
    def fixture(self, frame=4):
        root = (
            Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
            / "runtime_twist_restart/c/fixtures"
        )
        d = root / "runtime_twist_restart" / f"mesh-overlap-unreduced-restart-{frame}"
        l = json.loads((root / "layout.json").read_text())
        c = json.loads((d / "case.json").read_text())
        k = json.loads((HERE / "contracts/runtime_twist_restart.json").read_text())
        b = {n: np.fromfile(d / f"{n}.bin", np.uint32) for n in c["buffer_words"]}
        e = {n: np.fromfile(d / f"{n}.expected.bin", np.uint32) for n in b}
        return l, c, b, e, k

    def test_restart_oracle_receipt_is_bound_to_the_report(self):
        root = (
            Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
            / "runtime_twist_restart/c"
        )
        report = json.loads((root / "report.json").read_text())
        verify_probe_file(
            root / "cpu-oracle-noalias.ll", report["environment"]["cpu_oracle"]
        )

    def test_all_native_snapshots_pass_independent_checks(self):
        for frame in range(8):
            with self.subTest(frame=frame):
                l, c, b, e, k = self.fixture(frame)
                check_outputs(l, c, b, e, e, k)

    def test_past_future_and_counter_corruption_is_rejected(self):
        for mutation in ("past", "future", "counter", "published"):
            l, c, b, e, k = self.fixture()
            if mutation == "counter":
                e["FrameIndex"][0] += 1
            elif mutation == "published":
                e["Trajectory"][c["start_frame"] * c["trace_stride"]] ^= 1
            else:
                offset = (
                    0
                    if mutation == "past"
                    else (c["start_frame"] + 1) * c["trace_stride"]
                )
                e["Trajectory"][offset] ^= 1
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                check_outputs(l, c, b, e, e, k)

    def test_restart_index_and_counts_are_checked(self):
        for mutation in ("index", "count", "iterations"):
            l, c, b, _, _ = self.fixture()
            if mutation == "index":
                b["FrameIndex"][0] += 1
            elif mutation == "iterations":
                c["iterations"] = 2
            else:
                b["PairCount"][0] = c["pair_capacity"] + 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_case(l, c, b)

    def test_joint_allowance_cannot_mask_geometry_or_identity(self):
        for field in ("lin_jac", "inv_lhs", "joint_id", "impulse"):
            l, c, b, e, k = self.fixture()
            a = {n: v.copy() for n, v in e.items()}
            offset = l["joint_constraint"]["elements"] + l["joint_element"][field]
            if field == "joint_id":
                a["JointConstraints"][offset] ^= 1
            else:
                a["JointConstraints"].view(np.float32)[offset] += 0.001
            with (
                self.subTest(field=field),
                self.assertRaises((ValueError, AssertionError)),
            ):
                check_joint_rows(l, c, b, a, e, k)

    def test_matching_corrupt_trajectories_fail_physical_checks(self):
        for mutation in ("penetration", "anchor", "speed", "spin"):
            l, c, b, e, k = self.fixture()
            start = c["start_frame"] * c["trace_stride"]
            if mutation in ("speed", "spin"):
                field = "velocity_linear" if mutation == "speed" else "velocity_angular"
                offset = start + c["body_count"] * l["pose_words"] + l[field]
                e["Trajectory"].view(np.float32)[offset] = 100
            else:
                axis = 1 if mutation == "penetration" else 2
                offset = start + l["pose_translation"] + axis
                e["Trajectory"].view(np.float32)[offset] -= 1
            with (
                self.subTest(mutation=mutation),
                self.assertRaisesRegex(AssertionError, "Runtime"),
            ):
                check_physics(l, c, b, e, k, [c["start_frame"]])

    def test_policy_requires_exact_configuration_and_field(self):
        _, c, _, _, k = self.fixture()
        for field in ("JointConstraints", "Poses", "TileGeometry"):
            with self.subTest(field=field), self.assertRaises(AssertionError):
                scene_numeric.check(c, k, field, np.array([1e-4]), np.array([0.0]))
        with self.assertRaises(ValueError):
            scene_numeric.tolerances(dict(c, contact_reduction=True), k, "JointRhs")
        k = json.loads((HERE / "contracts/runtime_twist_scene.json").read_text())
        p = k["numerical_cases"]["mesh-overlap-unreduced-f8"]
        self.assertEqual(set(p["trajectory_frames"]), {"8"})
        self.assertEqual(p["required_restart_operation"], "runtime_twist_restart")
        registry = json.loads((HERE / "parity.json").read_text())
        feature = next(
            f
            for f in registry["features"]
            if f["operation"] == p["required_restart_operation"]
        )
        contract = json.loads((HERE.parents[1] / feature["contract"]).read_text())
        self.assertEqual(len(contract["required_cases"]), 8)


if __name__ == "__main__":
    unittest.main()
