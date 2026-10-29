"""Joint scene geometry, coloring and continuous diagnostic regressions."""

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from joint_scene_physics import check as check_physics
from joint_scene_validation import BUFFERS, check_outputs, validate_case
from replay import HERE


class JointSceneTests(unittest.TestCase):
    def fixture(self, name="joint_chain-c1-l3-b1-f1"):
        root = Path(
            os.environ.get(
                "NEXUS_TILELANG_JOINT_SCENE_FIXTURES",
                Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
                / "joint_scene/c/fixtures",
            )
        )
        layout = json.loads((root / "layout.json").read_text())
        d = root / "joint_scene" / name
        case = json.loads((d / "case.json").read_text())
        contract = json.loads((HERE / "contracts/joint_scene.json").read_text())
        before = {n: np.fromfile(d / f"{n}.bin", np.uint32) for n in BUFFERS}
        expected = {
            n: np.fromfile(d / f"{n}.expected.bin", np.uint32)
            if n in contract["outputs"]
            else v.copy()
            for n, v in before.items()
        }
        return layout, case, before, expected, contract

    def test_all_native_joint_scenes(self):
        contract = json.loads((HERE / "contracts/joint_scene.json").read_text())
        for name in contract["required_cases"]:
            with self.subTest(case=name):
                l, c, b, e, k = self.fixture(name)
                check_outputs(l, c, b, e, e, k)

    def test_new_buffer_guards_and_inputs_are_exact(self):
        for name in (
            "Builders",
            "JointConstraints",
            "Joints",
            "JointColors",
            "MotorParams",
            "JointBias",
            "JointUnbiased",
            "FrameIndex",
            "Trajectory",
        ):
            l, c, b, e, k = self.fixture()
            a = {n: v.copy() for n, v in e.items()}
            a[name][-1] ^= 1
            with self.subTest(buffer=name), self.assertRaises(AssertionError):
                check_outputs(l, c, b, a, e, k)

    def test_color_conflicts_rejected_before_dispatch(self):
        l, c, b, _e, _k = self.fixture()
        b["JointColors"][:3] = c["joints_per_batch"]
        with self.assertRaisesRegex(ValueError, "overlapping writers"):
            validate_case(l, c, b)

    def test_trajectory_uses_exact_published_state_and_frame_count(self):
        for mutation in ("frame", "last_state", "padding"):
            l, c, b, e, k = self.fixture()
            a = {n: v.copy() for n, v in e.items()}
            if mutation == "frame":
                a["FrameIndex"][0] += 1
            if mutation == "last_state":
                a["Trajectory"][l["pose_words"] + l["pose_translation"]] ^= 1
            if mutation == "padding":
                a["Trajectory"][l["pose_words"] - 1] ^= 1
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                check_outputs(l, c, b, a, e, k)

    def test_all_frame_physics_rejects_matching_corrupt_intermediate_states(self):
        for mutation in ("static", "anchor", "rotation", "energy", "nonfinite"):
            l, c, b, e, k = self.fixture("joint_fixed-c1-l1-b1-f120")
            f = e["Trajectory"].view(np.float32)
            if mutation == "static":
                f[l["pose_translation"]] += np.float32(0.1)
            if mutation == "anchor":
                f[l["pose_words"] + l["pose_translation"]] += np.float32(0.2)
            if mutation == "rotation":
                f[l["pose_words"] + l["pose_rotation"] + 3] = np.float32(2.0)
            if mutation == "energy":
                f[
                    c["body_count"] * l["pose_words"]
                    + l["velocity_words"]
                    + l["velocity_linear"]
                ] = np.float32(20.0)
            if mutation == "nonfinite":
                f[l["pose_words"] + l["pose_translation"]] = np.nan
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                check_physics(l, c, b, e, k)

    def test_matching_joint_impulse_violation_rejected(self):
        l, c, b, e, k = self.fixture("joint_motor-c1-l1-b1-f1")
        ef = l["joint_element"]
        off = l["joint_constraint"]["elements"]
        f = e["JointConstraints"].view(np.float32)
        f[off + ef["impulse"]] = f[off + ef["impulse_bounds"] + 1] + np.float32(1.0)
        with self.assertRaisesRegex(AssertionError, "impulse outside bounds"):
            check_outputs(l, c, b, e, e, k)

    def test_invalid_dimensions_and_control_uniforms(self):
        for name in (
            "joint_count",
            "joints_per_batch",
            "joint_storage",
            "joint_color_count",
            "trace_stride",
        ):
            l, c, b, _e, _k = self.fixture()
            c[name] += 1
            with self.subTest(field=name), self.assertRaises(ValueError):
                validate_case(l, c, b)
        for name in ("FrameIndex", "JointBias", "JointUnbiased"):
            l, c, b, _e, _k = self.fixture()
            b[name][0] += 1
            with self.subTest(buffer=name), self.assertRaises(ValueError):
                validate_case(l, c, b)

    def test_motor_energy_check_cannot_be_changed_by_input_force_limit(self):
        l, c, b, _e, _k = self.fixture("joint_motor-c1-l1-b1-f120")
        offset = (
            l["impulse_joint"]["data"]
            + l["generic_joint"]["motors"]
            + l["motor"]["max_force"]
        )
        b["Joints"].view(np.float32)[offset] = 100.0
        with self.assertRaisesRegex(ValueError, "motor changed"):
            validate_case(l, c, b)


if __name__ == "__main__":
    unittest.main()
