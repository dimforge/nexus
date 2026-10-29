"""Real collider fixture checks for indexed parents, readonly inputs and output guards."""

import json
import os
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_replay import BUFFERS, validate_case, check_outputs
from replay import HERE


class PoseTests(unittest.TestCase):
    def load(self, index=4):
        root = (
            Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
            / "sync_colliders/c/fixtures"
        )
        self.layout = json.loads((root / "layout.json").read_text())
        self.case = json.loads((root / "sync_colliders/suite.json").read_text())[
            "cases"
        ][index]
        folder = root / "sync_colliders" / self.case["name"]
        self.before = {
            n: np.fromfile(folder / f"{n}.bin", dtype=np.uint32) for n in BUFFERS
        }
        self.actual = {
            n: np.fromfile(folder / f"{n}.actual.bin", dtype=np.uint32) for n in BUFFERS
        }
        self.expected = dict(
            self.before,
            WorldPoses=np.fromfile(folder / "WorldPoses.expected.bin", dtype=np.uint32),
        )
        self.contract = json.loads((HERE / "contracts/sync_colliders.json").read_text())

    def check(self):
        return check_outputs(
            self.layout,
            self.case,
            self.before,
            self.actual,
            self.expected,
            self.contract,
        )

    def test_all_rust_cases(self):
        for i in range(7):
            self.load(i)
            self.check()

    def test_active_parent_bounds_and_ignored_sentinels(self):
        self.load()
        self.check()
        self.assertEqual(int(self.before["Parents"][-1]), 2**32 - 1)
        self.before["Parents"][0] = self.case["body_count"]
        with self.assertRaises(ValueError):
            validate_case(self.layout, self.case, self.before)

    def test_input_padding_and_inactive_writes_fail(self):
        self.load()
        for name, index in (
            ("Parents", 1),
            ("LocalPoses", 0),
            ("BodyPoses", 0),
            ("WorldPoses", -1),
            ("WorldPoses", self.layout["pose_padding"]),
        ):
            self.actual[name][index] ^= np.uint32(1)
            with self.assertRaises(AssertionError):
                self.check()
            self.actual[name][index] ^= np.uint32(1)

    def test_bad_layout_float_and_buffer_size_fail(self):
        self.load()
        bad = dict(self.layout, pose_translation=self.layout["pose_rotation"])
        with self.assertRaises(ValueError):
            validate_case(bad, self.case, self.before)
        offset = self.layout["pose_translation"]
        original = self.actual["WorldPoses"][offset]
        for value in (123.0, float("nan")):
            self.actual["WorldPoses"].view(np.float32)[offset] = value
            with self.assertRaises(AssertionError):
                self.check()
        self.actual["WorldPoses"][offset] = original
        self.actual["WorldPoses"] = self.actual["WorldPoses"][:-1]
        with self.assertRaises(ValueError):
            self.check()


if __name__ == "__main__":
    unittest.main()
