"""Exercise guardrails against actual exported Rust buffers, after replay.py."""

import json
import os
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from replay import BUFFERS, check_outputs, validate_case


class ReplayGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = (
            Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
            / "c/fixtures"
        )
        if not root.exists():
            raise RuntimeError(
                "Run replay.py --target c first, or set NEXUS_TILELANG_RESULTS"
            )
        cls.layout = json.loads((root / "layout.json").read_text())
        suite = json.loads((root / "suite.json").read_text())
        cls.case = suite["cases"][-1]
        directory = root / cls.case["name"]
        cls.before = {
            name: np.fromfile(directory / f"{name}.bin", dtype=np.uint32)
            for name in BUFFERS
        }
        cls.actual = {
            name: np.fromfile(directory / f"{name}.actual.bin", dtype=np.uint32)
            for name in BUFFERS
        }
        cls.expected = {
            name: (
                np.fromfile(directory / f"{name}.expected.bin", dtype=np.uint32)
                if name in ("Tiles", "Velocities")
                else cls.before[name]
            )
            for name in BUFFERS
        }
        cls.contract = json.loads((HERE / "contract.json").read_text())

    def copies(self, buffers):
        return {name: value.copy() for name, value in buffers.items()}

    def check(self, actual):
        return check_outputs(
            self.layout, self.case, self.before, actual, self.expected, self.contract
        )

    def test_actual_generated_cpu_result_passes(self):
        self.check(self.actual)

    def test_velocity_race_is_rejected(self):
        arrays = self.copies(self.before)
        offset = self.layout["header"]["diag_a"] + 3
        arrays["Tiles"][offset + 4] = arrays["Tiles"][offset]
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_case(self.layout, self.case, arrays)

    def test_invalid_body_or_contact_count_is_rejected(self):
        for field, value in [("body_a", self.case["bodies"]), ("normal", 5)]:
            with self.subTest(field=field):
                arrays = self.copies(self.before)
                arrays["Tiles"][self.layout["header"][field] + 3] = value
                with self.assertRaises(ValueError):
                    validate_case(self.layout, self.case, arrays)

    def test_parameter_write_cannot_hide_in_float_tolerance(self):
        arrays = self.copies(self.actual)
        arrays["Params"][0] ^= np.uint32(1)
        with self.assertRaisesRegex(AssertionError, "Unexpected write"):
            self.check(arrays)

    def test_inactive_point_write_is_rejected(self):
        arrays = self.copies(self.actual)
        # Lane 4 has zero active points in the generated suite.
        offset = self.layout["points"] + self.layout["point"]["arm_a"] + 4 * 4 + 3
        arrays["Tiles"][offset] ^= np.uint32(1)
        with self.assertRaisesRegex(AssertionError, "Unexpected write"):
            self.check(arrays)

    def test_padding_write_is_rejected(self):
        arrays = self.copies(self.actual)
        arrays["Velocities"][3] ^= np.uint32(1)
        with self.assertRaisesRegex(AssertionError, "Unexpected write"):
            self.check(arrays)

    def test_nonfinite_and_wrong_solver_outputs_are_rejected(self):
        for value in [np.nan, 100.0]:
            with self.subTest(value=value):
                arrays = self.copies(self.actual)
                arrays["Velocities"].view(np.float32)[0] = value
                with self.assertRaises(AssertionError):
                    self.check(arrays)


if __name__ == "__main__":
    unittest.main()
