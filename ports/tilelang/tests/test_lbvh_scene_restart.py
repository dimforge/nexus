"""Identical-state LBVH restarts retain exact geometry, counters and trace guards."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import test_scene_restart as restarts

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay import HERE
from scene_restart_validation import check_outputs, validate_case


class LbvhSceneRestartTests(restarts.SceneRestartTests):
    operations = ("general_lbvh_restart", "joint_lbvh_restart")

    def fixture(self, op="general_lbvh_restart", name="general_triangle-restart-35"):
        # Inherited graph corruption test targets an established contact graph.
        if name == "general_triangle-restart-39":
            name = "general_triangle-restart-38"
        root = Path(
            os.environ.get(
                "NEXUS_TILELANG_LBVH_RESTART_FIXTURES",
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

    def test_snapshot_contracts_select_restart_validation(self):
        for op in (*restarts.SceneRestartTests.operations, *self.operations):
            contract = json.loads((HERE / f"contracts/{op}.json").read_text())
            with self.subTest(operation=op):
                self.assertEqual(contract["comparison"], "scene_restart_fields")

    def test_past_future_and_final_trace_are_exact(self):
        for frame in (0, 35, 36):
            l, c, b, e, k = self.fixture()
            a = {n: v.copy() for n, v in e.items()}
            a["Trajectory"][frame * c["trace_stride"]] ^= 1
            with self.subTest(frame=frame), self.assertRaises(AssertionError):
                check_outputs(l, c, b, a, e, k)

    def test_tree_and_pair_corruption_are_rejected_even_if_reference_matches(self):
        for name, offset, value in (
            ("LbvhTree", "left", 0),
            ("Pairs", None, 0xFFFFFFFF),
        ):
            l, c, b, e, k = self.fixture()
            a = {n: v.copy() for n, v in e.items()}
            a[name][l["lbvh_node"][offset] if offset else 0] = value
            with (
                self.subTest(buffer=name),
                self.assertRaises((AssertionError, ValueError, IndexError)),
            ):
                check_outputs(l, c, b, a, a, k)

    def test_restart_normalization_preserves_inputs_and_guards(self):
        l, c, b, e, k = self.fixture()
        before = {n: v.copy() for n, v in b.items()}
        validate_case(l, c, b)
        check_outputs(l, c, b, e, e, k)
        for name in b:
            np.testing.assert_array_equal(b[name], before[name])
        for name in ("LbvhFrontierLen", "LbvhTree", "LbvhSorted"):
            a = {n: v.copy() for n, v in e.items()}
            a[name][-1] ^= 1
            with self.subTest(buffer=name), self.assertRaises(AssertionError):
                check_outputs(l, c, b, a, e, k)
