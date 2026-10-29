"""Live body fixtures and negative checks for exact writes, ABI and float results."""

import copy
import json
import os
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from body_validation import BUFFERS, OPERATIONS, check_outputs, validate_case
from replay import HERE


class BodyTests(unittest.TestCase):
    def load(self, operation="free_step", index=4):
        root = (
            Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
            / operation
            / "c/fixtures"
        )
        self.layout = json.loads((root / "layout.json").read_text())
        self.case = json.loads((root / operation / "suite.json").read_text())["cases"][
            index
        ]
        directory = root / operation / self.case["name"]
        self.contract = json.loads((HERE / f"contracts/{operation}.json").read_text())
        self.before = {
            n: np.fromfile(directory / f"{n}.bin", dtype=np.uint32) for n in BUFFERS
        }
        self.actual = {
            n: np.fromfile(directory / f"{n}.actual.bin", dtype=np.uint32)
            for n in BUFFERS
        }
        self.expected = {
            n: np.fromfile(directory / f"{n}.expected.bin", dtype=np.uint32)
            if n in self.contract["outputs"]
            else self.before[n]
            for n in BUFFERS
        }

    def check(self):
        return check_outputs(
            self.layout,
            self.case,
            self.before,
            self.actual,
            self.expected,
            self.contract,
        )

    def test_all_live_body_stages(self):
        for operation in OPERATIONS:
            for index in range(7):
                with self.subTest(operation=operation, index=index):
                    self.load(operation, index)
                    self.check()

    def test_padding_counter_and_input_writes_rejected(self):
        self.load()
        for name, index in (
            ("Velocities", self.layout["velocity_linear"] + 3),
            ("Counts", 0),
            ("FirstSlot", 1),
            ("HubCounts", 2),
            ("Poses", self.layout["pose_padding"]),
            ("WorldMass", 0),
            ("SolverVelocities", -1),
        ):
            with self.subTest(name=name):
                self.actual[name][index] ^= np.uint32(1)
                with self.assertRaises(AssertionError):
                    self.check()
                self.actual[name][index] ^= np.uint32(1)

    def test_copied_components_are_bitwise(self):
        for op, name, field in (
            ("cleanup", "SolverVelocities", "velocity_linear"),
            ("init_bodies", "SolverPoses", "pose_rotation"),
            ("finalize", "Velocities", "velocity_angular"),
        ):
            with self.subTest(operation=op):
                self.load(op)
                self.actual[name][self.layout[field]] ^= np.uint32(1)
                with self.assertRaises(AssertionError):
                    self.check()

    def test_float_mismatch_and_nan_rejected(self):
        self.load()
        index = self.layout["pose_words"] + self.layout["pose_translation"]
        original = self.actual["Poses"][index]
        for value in (100.0, float("nan")):
            self.actual["Poses"].view(np.float32)[index] = value
            with self.assertRaises(AssertionError):
                self.check()
        self.actual["Poses"][index] = original
        self.check()

    def test_invalid_batch_sizes_and_layout_rejected(self):
        self.load()
        for field, value in (
            ("count", self.case["count"] + 1),
            ("slots", self.case["slots"] + 1),
            ("iterations", 0),
            ("substeps", 0),
            ("storage", self.case["storage"] - 1),
        ):
            broken = dict(self.case, **{field: value})
            with self.assertRaises(ValueError):
                validate_case(self.layout, broken, self.before)
        broken = dict(self.layout, velocity_angular=self.layout["velocity_linear"])
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            validate_case(broken, self.case, self.before)

    def test_invalid_inputs_and_truncated_output_rejected(self):
        self.load()
        for field, value in ((0, 0.0), (1, -1.0), (3, float("nan"))):
            arrays = copy.deepcopy(self.before)
            arrays["Params"].view(np.float32)[field] = value
            with self.assertRaises(ValueError):
                validate_case(self.layout, self.case, arrays)
        self.actual["Poses"] = self.actual["Poses"][:-1]
        with self.assertRaises(ValueError):
            self.check()

    def test_gravity_preserves_rust_whole_vector_predicate(self):
        self.load("init_increments")
        values = (
            self.actual["Increments"]
            .view(np.float32)
            .reshape(-1, self.layout["velocity_words"])
        )
        p = self.before["Params"].view(np.float32)
        # Body 1 has inverse mass (1,0,0); Rust still applies all gravity components.
        np.testing.assert_allclose(
            values[
                1, self.layout["velocity_linear"] : self.layout["velocity_linear"] + 3
            ],
            p[3:6] * p[0],
            rtol=0,
            atol=0,
        )
        np.testing.assert_array_equal(
            values[
                0, self.layout["velocity_linear"] : self.layout["velocity_linear"] + 3
            ],
            np.zeros(3),
        )


if __name__ == "__main__":
    unittest.main()
