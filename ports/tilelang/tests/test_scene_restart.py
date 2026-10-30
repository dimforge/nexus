"""Native snapshot replay preserves the past and future trajectory exactly."""

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay import HERE
from scene_restart_validation import check_outputs, validate_case


class SceneRestartTests(unittest.TestCase):
    operations = ("general_restart", "joint_restart")

    def fixture(self, op="general_restart", name="general_triangle-restart-19"):
        root = Path(
            os.environ.get(
                "NEXUS_TILELANG_RESTART_FIXTURES",
                Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
                / f"{op}/c/fixtures",
            )
        )
        d = root / op / name
        l = json.loads((root / "layout.json").read_text())
        c = json.loads((d / "case.json").read_text())
        k = json.loads((HERE / f"contracts/{op}.json").read_text())
        b = {n: np.fromfile(d / f"{n}.bin", np.uint32) for n in c["buffer_words"]}
        e = {
            n: np.fromfile(d / f"{n}.expected.bin", np.uint32)
            if n in k["outputs"]
            else v.copy()
            for n, v in b.items()
        }
        return l, c, b, e, k

    def test_all_native_restarts(self):
        for op in self.operations:
            k = json.loads((HERE / f"contracts/{op}.json").read_text())
            for name in k["required_cases"]:
                with self.subTest(case=name):
                    l, c, b, e, k = self.fixture(op, name)
                    check_outputs(l, c, b, e, e, k)

    def test_snapshot_counter_and_capacity_are_checked(self):
        for name in ["FrameIndex", "Plan", "PairCount"]:
            l, c, b, _e, _k = self.fixture()
            b[name][0] = 0xFFFFFFFF
            with (
                self.subTest(buffer=name),
                self.assertRaises((AssertionError, ValueError)),
            ):
                validate_case(l, c, b)

    def test_previous_contact_graph_is_checked(self):
        l, c, b, _e, _k = self.fixture(name="general_triangle-restart-39")
        b["Links"][l["link"]["solver_body_b"]] = c["body_count"]
        with self.assertRaises(AssertionError):
            validate_case(l, c, b)

    def test_past_future_and_final_trace_are_exact(self):
        for frame in [0, 19, 20]:
            l, c, b, e, k = self.fixture()
            a = {n: v.copy() for n, v in e.items()}
            a["Trajectory"][frame * c["trace_stride"]] ^= 1
            with self.subTest(frame=frame), self.assertRaises(AssertionError):
                check_outputs(l, c, b, a, e, k)

    def test_exact_buffer_guards(self):
        for name in ["Poses", "Tiles", "Contacts", "FrameIndex", "Trajectory"]:
            l, c, b, e, k = self.fixture()
            a = {n: v.copy() for n, v in e.items()}
            a[name][-1] ^= 1
            with self.subTest(buffer=name), self.assertRaises(AssertionError):
                check_outputs(l, c, b, a, e, k)


if __name__ == "__main__":
    unittest.main()
