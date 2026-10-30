"""Native capacity upload and resident commit-sequence invariants."""

import copy
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frame_scene_validation import check_stats, validate_variants
from lbvh_scene_validation import check_outputs
from replay import HERE


class FrameSceneTests(unittest.TestCase):
    def fixture(self, op="analytic_frame_retry", name="capacity_growth-c2-l1-b1-f120"):
        root = Path(
            os.environ.get(
                "NEXUS_TILELANG_FRAME_FIXTURES", HERE / f"results/{op}/c/fixtures"
            )
        )
        d = root / op / name
        l = json.loads((root / "layout.json").read_text())
        k = json.loads((HERE / f"contracts/{op}.json").read_text())

        def load(path):
            c = json.loads((path / "case.json").read_text())
            return c, {
                n: np.fromfile(path / f"{n}.bin", np.uint32) for n in c["buffer_words"]
            }

        c, b = load(d)
        e = {
            n: np.fromfile(d / f"{n}.expected.bin", np.uint32)
            if n in k["outputs"]
            else v.copy()
            for n, v in b.items()
        }
        variants = [
            load(d / "variants" / str(i)) for i in range(len(c["capacity_ladder"]))
        ]
        return l, c, b, e, k, variants, d

    def test_native_reference_and_every_initial_variant(self):
        for op in ("analytic_frame_retry", "general_frame_retry", "joint_frame_retry"):
            k = json.loads((HERE / f"contracts/{op}.json").read_text())
            for name in k["required_cases"]:
                with self.subTest(case=name):
                    l, c, b, e, k, variants, d = self.fixture(op, name)
                    validate_variants(l, c, b, variants)
                    check_outputs(l, c, b, e, e, k)
                    self.assertFalse(list((d / "variants").rglob("*.expected.bin")))

    def test_variants_cannot_change_initial_physics_or_schedule(self):
        for name in ("Poses", "Gravity", "RbdParams"):
            l, c, b, _e, _k, variants, _d = self.fixture()
            variants[0][1][name].view(np.float32)[0] += 0.001
            with (
                self.subTest(buffer=name),
                self.assertRaises((AssertionError, ValueError)),
            ):
                validate_variants(l, c, b, variants)
        l, c, b, _e, _k, variants, _d = self.fixture()
        variants[0][0]["frames"] -= 1
        with self.assertRaises(ValueError):
            validate_variants(l, c, b, variants)

    def test_native_storage_capacity_and_default_rotations_are_checked(self):
        l, c, b, _e, _k, variants, _d = self.fixture()
        for vc, vb in variants:
            rows = vb["States"].reshape(-1, l["state_words"])
            for side in ("pose_a", "pose_b"):
                off = l["state"][side] + l["pose_rotation"]
                np.testing.assert_array_equal(
                    rows[:, off : off + 4].view(np.float32),
                    np.tile([0, 0, 0, 1], (len(rows), 1)),
                )
        bad = copy.deepcopy(variants)
        bad[0][1]["BatchIds"][l["batch"]["contacts_capacity"]] += 1
        with self.assertRaises(ValueError):
            validate_variants(l, c, b, bad)
        bad = copy.deepcopy(variants)
        bad[-1][1]["OldTiles"][-1] ^= 1
        with self.assertRaises(AssertionError):
            validate_variants(l, c, b, bad)


class FrameStatsTests(unittest.TestCase):
    def fixture(self):
        c = {
            "frames": 3,
            "capacity_ladder": [1, 9],
            "expected_growth_frames": [2],
            "kind": "analytic_scene",
        }
        stats = [
            {"step": 1, "variant": 0, "retries": 0, "counts": {"pairs": 1}},
            {"step": 2, "variant": 1, "retries": 1, "counts": {"pairs": 2}},
            {"step": 3, "variant": 1, "retries": 0, "counts": {"pairs": 2}},
        ]
        return c, stats

    def test_delayed_commit_sequence(self):
        check_stats(*self.fixture())

    def test_missing_double_unnecessary_and_overflowing_commits_are_rejected(self):
        for edit in (
            lambda s: s.pop(),
            lambda s: s[1].update(step=1),
            lambda s: s[1].update(step=True),
            lambda s: s[1].update(retries=0),
            lambda s: s[1].update(variant=0),
            lambda s: s[1]["counts"].update(pairs=1),
            lambda s: s[1]["counts"].update(pairs=10),
            lambda s: s[2].update(variant=0),
            lambda s: s[1]["counts"].update(deferred_pairs=1),
        ):
            c, s = self.fixture()
            edit(s)
            with self.subTest(stats=s), self.assertRaises(AssertionError):
                check_stats(c, s)
