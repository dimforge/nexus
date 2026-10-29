"""Joint ABI, write conflicts, guard and independent impulse regressions."""

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joint_validation as joint
from replay import HERE


class JointTests(unittest.TestCase):
    def fixture(self, op="joint_program", name="n12-b3-z0"):
        root = Path(
            os.environ.get(
                "NEXUS_TILELANG_JOINT_FIXTURES",
                Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
                / op
                / "c/fixtures",
            )
        )
        layout = json.loads((root / "layout.json").read_text())
        directory = root / op / name
        case = json.loads((directory / "case.json").read_text())
        contract = json.loads((HERE / "contracts" / f"{op}.json").read_text())
        before = {
            n: np.fromfile(directory / f"{n}.bin", np.uint32) for n in joint.BUFFERS
        }
        expected = {
            n: np.fromfile(directory / f"{n}.expected.bin", np.uint32)
            if n in contract["outputs"]
            else v.copy()
            for n, v in before.items()
        }
        return layout, case, before, expected, contract

    def test_all_native_joint_cases(self):
        for op in joint.OPERATIONS:
            contract = json.loads((HERE / "contracts" / f"{op}.json").read_text())
            for name in contract["required_cases"]:
                with self.subTest(operation=op, case=name):
                    l, c, b, e, k = self.fixture(op, name)
                    joint.check_outputs(l, c, b, e, e, k)

    def test_every_capacity_guard_and_readonly_buffer_is_exact(self):
        for op in joint.OPERATIONS:
            for name in joint.BUFFERS:
                l, c, b, e, k = self.fixture(op)
                actual = {n: v.copy() for n, v in e.items()}
                actual[name][-1] ^= 1
                with (
                    self.subTest(operation=op, buffer=name),
                    self.assertRaises(AssertionError),
                ):
                    joint.check_outputs(l, c, b, actual, e, k)

    def test_inactive_rows_and_integer_identities_are_exact(self):
        l, c, b, e, k = self.fixture()
        cf, ef = l["joint_constraint"], l["joint_element"]
        for index in (
            cf["elements"] + ef["joint_id"],
            cf["elements"] + 11 * l["joint_element_words"] + ef["rhs"],
            cf["solver_vel_a"],
        ):
            a = {n: v.copy() for n, v in e.items()}
            a["Constraints"][index] ^= 1
            with (
                self.subTest(index=index),
                self.assertRaises((AssertionError, ValueError)),
            ):
                joint.check_outputs(l, c, b, a, e, k)

    def test_static_or_zero_row_shared_writers_are_rejected(self):
        for op in ("joint_solve", "joint_program"):
            l, c, b, _e, _k = self.fixture(op)
            if op == "joint_solve":
                rows = b["Constraints"].reshape(-1, l["joint_constraint_words"])
                cf = l["joint_constraint"]
                rows[3, cf["solver_vel_a"]] = rows[0, cf["solver_vel_a"]]
                rows[3, cf["len"]] = 0
            else:
                rows = b["Joints"].reshape(-1, l["impulse_joint_words"])
                jf = l["impulse_joint"]
                rows[3, jf["body_a"]] = rows[0, jf["body_a"]]
            with (
                self.subTest(operation=op),
                self.assertRaisesRegex(ValueError, "overlapping velocity"),
            ):
                joint.validate_case(l, c, b)

    def test_matching_outputs_cannot_hide_impulses_outside_limits(self):
        l, c, b, e, k = self.fixture("joint_solve")
        row = int(b["ColorGroups"][0]) * c["batches"]
        base = row * l["joint_constraint_words"] + l["joint_constraint"]["elements"]
        ef = l["joint_element"]
        f = e["Constraints"].view(np.float32)
        f[base + ef["impulse"]] = f[base + ef["impulse_bounds"] + 1] + np.float32(1.0)
        with self.assertRaisesRegex(AssertionError, "outside bounds"):
            joint.check_outputs(l, c, b, e, e, k)

    def test_invalid_input_layout_masks_ids_and_parameters(self):
        mutations = (
            "storage",
            "batch",
            "color",
            "body",
            "mask",
            "motor",
            "builder",
            "velocity",
            "length",
        )
        for mutation in mutations:
            l, c, b, _e, _k = self.fixture("joint_solve")
            if mutation == "storage":
                c["buffer_words"]["Constraints"] -= 1
            if mutation == "batch":
                b["BatchIds"][l["batch"]["num_batches"]] += 1
            if mutation == "color":
                b["Color"][0] = 3
            if mutation == "body":
                b["Joints"][l["impulse_joint"]["body_a"]] = c["bodies"]
            if mutation == "mask":
                b["Joints"][
                    l["impulse_joint"]["data"] + l["generic_joint"]["motor_axes"]
                ] = 64
            if mutation == "motor":
                b["Joints"][
                    l["impulse_joint"]["data"]
                    + l["generic_joint"]["motors"]
                    + l["motor"]["model"]
                ] = 2
            if mutation == "builder":
                b["Builders"][
                    l["builder"]["joint"] + l["generic_joint"]["locked_axes"]
                ] = 64
            if mutation == "velocity":
                b["Velocities"].view(np.float32)[l["velocity_linear"]] = np.nan
            if mutation == "length":
                b["Constraints"][l["joint_constraint"]["len"]] = 13
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                joint.validate_case(l, c, b)

    def test_small_float_error_is_accepted_but_large_error_rejected(self):
        for delta in (1e-7, 0.1):
            l, c, b, e, k = self.fixture("joint_update")
            a = {n: v.copy() for n, v in e.items()}
            a["Constraints"].view(np.float32)[
                l["joint_constraint"]["elements"] + l["joint_element"]["rhs"]
            ] += np.float32(delta)
            if delta < 1e-6:
                joint.check_outputs(l, c, b, a, e, k)
            else:
                with self.assertRaises(AssertionError):
                    joint.check_outputs(l, c, b, a, e, k)


if __name__ == "__main__":
    unittest.main()
