"""Bounded contact sensitivity retains strict response and independent checks."""

import copy
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scene_numeric as numeric
from general_scene_physics import check as physics
from lbvh_scene_validation import check_outputs
from replay import HERE
from scene_trajectory import check_trajectory


class LbvhSceneNumericTests(unittest.TestCase):
    def fixture(self):
        root = Path(
            os.environ.get(
                "NEXUS_TILELANG_LBVH_NUMERIC_FIXTURES",
                Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
                / "general_lbvh_scene/c/fixtures",
            )
        )
        directory = root / "general_lbvh_scene/general_triangle-c1-l1-b1-f120"
        l = json.loads((root / "layout.json").read_text())
        c = json.loads((directory / "case.json").read_text())
        k = json.loads((HERE / "contracts/general_lbvh_scene.json").read_text())
        b = {
            n: np.fromfile(directory / f"{n}.bin", np.uint32) for n in c["buffer_words"]
        }
        e = {
            n: np.fromfile(directory / f"{n}.expected.bin", np.uint32)
            if n in k["outputs"]
            else v.copy()
            for n, v in b.items()
        }
        return l, c, b, e, k

    def test_native_and_independent_response(self):
        l, c, b, e, k = self.fixture()
        check_outputs(l, c, b, e, e, k)
        self.assertTrue(numeric.check_friction_response(l, c, e, e, k, 0))

    def test_single_frame_allowance_does_not_spread(self):
        for frame, delta, accepted in (
            (35, 0.01, False),
            (36, 0.14, True),
            (36, 0.16, False),
            (37, 0.01, False),
        ):
            l, c, b, e, k = self.fixture()
            a = {n: v.copy() for n, v in e.items()}
            off = (
                (frame - 1) * c["trace_stride"]
                + c["body_count"] * l["pose_words"]
                + l["velocity_words"]
                + l["velocity_angular"]
            )
            a["Trajectory"].view(np.float32)[off] += delta
            with self.subTest(frame=frame, delta=delta):
                if accepted:
                    check_trajectory(l, c, b, a, e, k)
                else:
                    with self.assertRaises(AssertionError):
                        check_trajectory(l, c, b, a, e, k)

    def test_scoped_frame_policy_rejects_invalid_frames_and_fields(self):
        l, c, b, e, k = self.fixture()
        for override in ({"0": {}}, {"121": {}}, {"036": {}}, {"36": {"Params": {}}}):
            bad = copy.deepcopy(k)
            bad["numerical_cases"][c["name"]]["trajectory_frames"] = override
            with self.subTest(override=override), self.assertRaises(ValueError):
                check_trajectory(l, c, b, e, e, bad)

    def test_policy_cannot_be_reused_with_another_broadphase(self):
        l, c, _b, e, k = self.fixture()
        with self.assertRaises(ValueError):
            numeric.check_friction_response(
                l, dict(c, broadphase="brute_force"), e, e, k, 0
            )

    def test_balanced_friction_redistribution_preserves_body_response(self):
        l, c, b, e, k = self.fixture()
        slot = int(e["Indices"][0])
        base = slot // 32 * l["tile_words"] + slot % 32 * 4
        length = int(e["Tiles"][base + l["header"]["normal"] + 3])
        offsets = [
            base + l["points"] + i * l["point_words"] + l["point"]["friction"] + j
            for i in range(length)
            for j in range(2)
        ]
        zero = {n: v.copy() for n, v in e.items()}
        zero["Tiles"][offsets] = 0
        baseline = numeric.impulse_response(l, zero, 0)
        columns = []
        for off in offsets:
            probe = {n: v.copy() for n, v in zero.items()}
            probe["Tiles"].view(np.float32)[off] = 1
            columns.append((numeric.impulse_response(l, probe, 0) - baseline).ravel())
        _, _, vectors = np.linalg.svd(np.array(columns).T, full_matrices=True)
        a = {n: v.copy() for n, v in e.items()}
        a["Tiles"].view(np.float32)[offsets] += vectors[-1] * 0.0001
        a["OldTiles"] = a["Tiles"].copy()
        self.assertGreater(
            float(
                np.max(
                    abs(
                        a["Tiles"].view(np.float32)[offsets]
                        - e["Tiles"].view(np.float32)[offsets]
                    )
                )
            ),
            2e-6,
        )
        check_outputs(l, c, b, a, e, k)

    def test_unbalanced_friction_and_matching_corrupt_cache_are_rejected(self):
        l, c, _b, e, k = self.fixture()
        slot = int(e["Indices"][0])
        base = slot // 32 * l["tile_words"] + slot % 32 * 4
        a = {n: v.copy() for n, v in e.items()}
        a["Tiles"].view(np.float32)[base + l["points"] + l["point"]["friction"]] += (
            0.0001
        )
        with self.assertRaisesRegex(AssertionError, "Contact body response"):
            numeric.check_friction_response(l, c, a, e, k, 0)
        a = {n: v.copy() for n, v in e.items()}
        cache = (
            slot // 32 * l["tile_words"]
            + l["warmstart"]
            + slot % 32 * 2 * l["velocity_words"]
        )
        a["Tiles"].view(np.float32)[
            cache + l["velocity_words"] + l["velocity_linear"]
        ] += 0.0001
        with self.assertRaisesRegex(
            AssertionError, "Independent cached impulse response"
        ):
            numeric.check_friction_response(l, c, a, a, k, 0)

    def test_matching_corrupt_physics_at_exception_frame_is_rejected(self):
        l, c, b, e, k = self.fixture()
        off = (
            35 * c["trace_stride"]
            + c["body_count"] * l["pose_words"]
            + l["velocity_words"]
            + l["velocity_linear"]
        )
        e["Trajectory"].view(np.float32)[off] = 100
        with self.assertRaises(AssertionError):
            physics(l, c, b, e, k)
