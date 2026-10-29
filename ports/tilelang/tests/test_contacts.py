"""Preparation guards and regression assertions on actual Rust-exported cases."""

import copy
import json
import os
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contact_replay import BUFFERS, check_outputs, output_masks, validate_case
from replay import HERE


class ContactTests(unittest.TestCase):
    def load(self, name="n129-mixed"):
        root = (
            Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
            / "prepare/c/fixtures"
        )
        self.layout = json.loads((root / "layout.json").read_text())
        self.cases = json.loads((root / "prepare/suite.json").read_text())["cases"]
        self.case = next(c for c in self.cases if c["name"] == name)
        folder = root / "prepare" / name
        self.before = {
            n: np.fromfile(folder / f"{n}.bin", dtype=np.uint32) for n in BUFFERS
        }
        self.actual = {
            n: np.fromfile(folder / f"{n}.actual.bin", dtype=np.uint32) for n in BUFFERS
        }
        self.expected = dict(
            self.before,
            Tiles=np.fromfile(folder / "Tiles.expected.bin", dtype=np.uint32),
        )
        self.contract = json.loads((HERE / "contracts/prepare.json").read_text())

    def check(self):
        return check_outputs(
            self.layout,
            self.case,
            self.before,
            self.actual,
            self.expected,
            self.contract,
        )

    def values(self, field, component=3):
        l = self.layout
        return self.actual["Tiles"].view(np.float32)[
            [
                l["points"] + k * l["point_words"] + l["point"][field] + component
                for k in range(4)
            ]
        ]

    def test_all_live_rust_cases(self):
        self.load()
        self.assertEqual(
            set(self.contract["required_cases"]), {c["name"] for c in self.cases}
        )
        for c in self.cases:
            self.load(c["name"])
            self.check()

    def test_nearest_reordered_points_and_first_tie(self):
        self.load("n1-nearest")
        np.testing.assert_array_equal(
            self.values("arm_a"), np.arange(1, 5, dtype=np.float32) * np.float32(0.15)
        )
        self.load("n1-ties")
        np.testing.assert_array_equal(
            self.values("arm_a"), np.full(4, 0.6, dtype=np.float32)
        )
        for name in ("n1-fresh", "n1-no-match", "n1-empty-old"):
            self.load(name)
            np.testing.assert_array_equal(
                self.values("arm_a"), np.zeros(4, dtype=np.float32)
            )

    def test_rotated_friction_basis(self):
        self.load("n1-rotated")
        np.testing.assert_allclose(self.values("friction", 0), 0.02, atol=2e-6)
        np.testing.assert_allclose(
            self.values("friction", 1), np.arange(1, 5) * 0.01, atol=2e-6
        )

    def test_recycling_uses_old_len_and_body_cache_tags(self):
        self.load("n1-recycled")
        l = self.layout
        self.assertEqual(
            self.before["Contacts"][l["contact"]["contact"] + l["manifold"]["len"]], 1
        )
        self.assertEqual(self.actual["Tiles"][l["header"]["normal"] + 3], 4)
        for d, side in enumerate(("a", "b")):
            tag = self.actual["Tiles"][
                l["warmstart"] + d * l["velocity_words"] + l["velocity_padding1"]
            ]
            self.assertEqual(
                tag, self.before["Links"][l["link"][f"solver_body_{side}"]]
            )
            self.assertNotEqual(
                tag, self.before["Links"][l["link"][f"vel_slot_{side}"]]
            )
        self.check()

    def test_exact_copies_tags_padding_and_readonly_inputs(self):
        self.load("n1-recycled")
        l = self.layout
        for name, index in [
            ("Links", 0),
            ("Contacts", 0),
            ("WorldMass", 0),
            ("SolverPoses", 0),
            ("Velocities", 0),
            ("States", 0),
            ("OldTiles", 0),
            ("Tiles", l["points"] + l["point"]["local_a"] + 3),
            ("Tiles", l["points"] + l["point"]["arm_a"] + 3),
            ("Tiles", l["warmstart"] + l["velocity_padding0"]),
            ("Tiles", l["warmstart"] + l["velocity_padding1"]),
            ("Tiles", -1),
        ]:
            self.actual[name][index] ^= np.uint32(1)
            with self.assertRaises(AssertionError):
                self.check()
            self.actual[name][index] ^= np.uint32(1)

    def test_invalid_inputs_fail_before_dispatch(self):
        self.load()
        l = self.layout
        for field, value in (
            ("contact", self.case["storage"]),
            ("solver_body_a", self.case["bodies"]),
            ("len", 5),
            ("recycled", 2),
        ):
            bad = {k: v.copy() for k, v in self.before.items()}
            bad["Links"][l["link"][field]] = value
            with self.assertRaises(ValueError):
                validate_case(l, self.case, bad)
        # Link 1 has previous data; link 0 is fresh.
        for field, value in (
            ("previous_constraint_index", self.case["old_tiles"] * 32),
            ("previous_constraint", 2**32 - 1),
        ):
            self.load("n1-recycled")
            bad = {k: v.copy() for k, v in self.before.items()}
            bad["Links"][l["link"][field]] = value
            with self.assertRaises(ValueError):
                validate_case(l, self.case, bad)
        self.load()
        bad = copy.deepcopy(l)
        bad["point"]["local_a"] = bad["point"]["local_b"]
        with self.assertRaises(ValueError):
            validate_case(bad, self.case, self.before)
        bad = dict(self.before, States=self.before["States"][:-1])
        with self.assertRaises(ValueError):
            validate_case(l, self.case, bad)
        bad = dict(self.case, count=self.case["storage"] + 1)
        with self.assertRaises(ValueError):
            validate_case(l, bad, self.before)
        bad = {k: v.copy() for k, v in self.before.items()}
        bad["SolverPoses"].view(np.float32)[l["pose_words"] + l["pose_rotation"]] = (
            np.nan
        )
        with self.assertRaises(ValueError):
            validate_case(l, self.case, bad)

    def test_float_tolerance_does_not_hide_corruption(self):
        self.load()
        _, mask = output_masks(self.layout, self.case, self.before)
        index = np.flatnonzero(mask)[0]
        for value in (123.0, float("nan")):
            self.actual["Tiles"].view(np.float32)[index] = value
            with self.assertRaises(AssertionError):
                self.check()

    def test_shared_body_inputs_and_zero_mass_are_supported(self):
        self.load()
        l = self.layout
        links = self.before["Links"].reshape(-1, l["link_words"])
        self.assertLess(
            len(set(links[: self.case["count"], l["link"]["solver_body_b"]])),
            self.case["count"],
        )
        self.check()
        self.load("n1-zero-mass")
        np.testing.assert_array_equal(
            self.values("arm_b"), np.zeros(4, dtype=np.float32)
        )
        self.check()


if __name__ == "__main__":
    unittest.main()
