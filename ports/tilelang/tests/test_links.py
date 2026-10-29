"""Persistence decisions, untouched-state guards and malformed native inputs."""

import copy
import json
import os
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from link_replay import BUFFERS, OPERATIONS, check_outputs, validate_case
from replay import HERE


class LinkTests(unittest.TestCase):
    def load(self, op="contact_persistence", name="n129"):
        root = (
            Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
            / op
            / "c/fixtures"
        )
        self.layout = json.loads((root / "layout.json").read_text())
        self.cases = json.loads((root / op / "suite.json").read_text())["cases"]
        self.case = next(c for c in self.cases if c["name"] == name)
        folder = root / op / name
        self.contract = json.loads((HERE / f"contracts/{op}.json").read_text())
        self.before = {
            n: np.fromfile(folder / f"{n}.bin", dtype=np.uint32) for n in BUFFERS
        }
        self.actual = {
            n: np.fromfile(folder / f"{n}.actual.bin", dtype=np.uint32) for n in BUFFERS
        }
        self.expected = {
            n: np.fromfile(folder / f"{n}.expected.bin", dtype=np.uint32)
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

    def links(self):
        return self.actual["Links"].reshape(-1, self.layout["link_words"])

    def test_device_plan_and_padded_launch_are_checked(self):
        self.load()
        self.assertGreater(self.case["launch_capacity"], self.case["count"])
        self.assertEqual(
            self.before["Plan"][self.layout["plan"]["bound"]], self.case["count"]
        )
        np.testing.assert_array_equal(self.actual["Plan"], self.before["Plan"])
        for index in (0, -1):
            self.load()
            self.actual["Plan"][index] ^= np.uint32(1)
            with self.subTest(guard=index), self.assertRaises(AssertionError):
                self.check()
        self.load()
        self.before["Plan"][self.layout["plan"]["bound"]] += 1
        with self.assertRaisesRegex(ValueError, "plan"):
            validate_case(self.layout, self.case, self.before)
        for capacity in (-1, True, 2**31):
            self.load()
            self.case["launch_capacity"] = capacity
            with (
                self.subTest(capacity=capacity),
                self.assertRaisesRegex(ValueError, "capacity"),
            ):
                validate_case(self.layout, self.case, self.before)

    def test_all_entry_points_and_composition(self):
        for op in OPERATIONS:
            self.load(op)
            self.assertEqual(
                set(self.contract["required_cases"]), {c["name"] for c in self.cases}
            )
            for c in self.cases:
                self.load(op, c["name"])
                self.check()

    def test_initialization_does_not_read_inert_body_ids(self):
        for operation in ("init_links", "contact_persistence"):
            self.load(operation)
            rows = self.before["Contacts"].reshape(-1, self.layout["contact_words"])
            self.assertEqual(rows[10, self.layout["contact"]["bodies"]], 2**32 - 1)
            self.assertEqual(self.links()[10, self.layout["link"]["len"]], 0)
            self.check()
            # The exception is confined to empty manifolds.
            rows[
                10, self.layout["contact"]["contact"] + self.layout["manifold"]["len"]
            ] = 1
            with self.assertRaises(ValueError):
                validate_case(self.layout, self.case, self.before)

    def test_recycling_predicates(self):
        self.load()
        rows = self.links()
        l = self.layout["link"]
        np.testing.assert_array_equal(
            rows[:18, l["recycled"]],
            [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
        )
        # No identity match and two empty/static adjacency lists never recycle.
        self.assertEqual(rows[1, l["previous_constraint"]], 2**32 - 1)
        self.assertEqual(rows[17, l["previous_constraint"]], 2**32 - 1)
        # Row ordering is preserved: duplicate matches in a reverse adjacency list pick its first.
        self.assertEqual(rows[0, l["previous_constraint"]], 1)
        self.assertEqual(rows[2, l["previous_constraint"]], 11)
        self.assertEqual(rows[6, l["previous_constraint"]], 27)
        for i in (0, 2, 6):
            previous = rows[i, l["previous_constraint"]]
            self.assertEqual(
                rows[i, l["previous_constraint_index"]],
                self.before["OldIndices"][previous],
            )

    def test_color_range_and_nonrecycled_matches(self):
        self.load()
        rows = self.links()
        l = self.layout["link"]
        self.assertEqual(self.actual["Colors"][2], 63)
        self.assertEqual(self.actual["Colors"][5], 7)
        self.assertEqual(rows[5, l["recycled"]], 0)
        for i in (0, 1, 3, 4, 10, 11, 17):
            self.assertEqual(self.actual["Colors"][i], self.before["Colors"][i])
            self.assertEqual(self.actual["Colored"][i], self.before["Colored"][i])
        for i in (2, 5, 7, 8):
            self.assertEqual(self.actual["Colored"][i], 1)

    def test_composition_matches_separate_stages(self):
        self.load()
        l = self.layout["link"]
        composed = {k: v.copy() for k, v in self.actual.items()}
        mask = np.zeros(self.case["storage"], dtype=bool)
        mask[: self.case["count"]] = self.links()[: self.case["count"], l["len"]] != 0
        self.load("match_links")
        np.testing.assert_array_equal(
            composed["Links"].reshape(-1, self.layout["link_words"])[
                mask[: self.case["storage"]]
            ],
            self.links()[mask[: self.case["storage"]]],
        )
        np.testing.assert_array_equal(composed["States"], self.actual["States"])
        self.load("seed_colors")
        for key in ("Colors", "Colored"):
            np.testing.assert_array_equal(composed[key], self.actual[key])

    def test_stale_gap_ids_are_not_dereferenced(self):
        for op in ("match_links", "seed_colors"):
            self.load(op)
            l = self.layout["link"]
            stride = self.layout["link_words"]
            self.assertEqual(
                self.before["Links"][10 * stride + l["solver_body_a"]], 2**32 - 1
            )
            self.check()
        self.load("init_links")
        l = self.layout["link"]
        # Both low-bit and high-bit multibody flags exclude their contacts.
        self.assertEqual(self.links()[11, l["len"]], 0)
        self.assertEqual(self.links()[29, l["len"]], 0)

    def test_state_copy_padding_and_swapped_ranges(self):
        for name in ("n129", "n65"):
            self.load(name=name)
            l = self.layout
            s = l["state"]
            rows = self.links()
            for i in (0, 3):
                dst = (self.case["new_base"] + i) * l["state_words"]
                if i == 0:
                    src = (
                        self.case["old_base"]
                        + int(rows[i, l["link"]["previous_constraint"]])
                    ) * l["state_words"]
                    np.testing.assert_array_equal(
                        self.actual["States"][dst : dst + l["state_words"]],
                        self.before["States"][src : src + l["state_words"]],
                    )
                else:
                    im = self.before["Contacts"].reshape(-1, l["contact_words"])[i]
                    collider = int(im[l["contact"]["colliders"]])
                    start = collider * l["pose_words"]
                    np.testing.assert_array_equal(
                        self.actual["States"][
                            dst + s["pose_a"] : dst + s["pose_a"] + l["pose_words"]
                        ],
                        self.before["ColliderPoses"][start : start + l["pose_words"]],
                    )
            self.check()

    def test_invalid_inputs_are_rejected(self):
        self.load("match_links")
        l = self.layout
        for name, index, value in (
            ("OldCounts", 0, 2**32 - 1),
            ("OldIds", 0, 2**32 - 1),
            ("Contacts", l["contact"]["colliders"], self.case["colliders"]),
            ("Links", l["link"]["solver_body_a"], self.case["bodies"]),
            ("Links", l["link"]["len"], 5),
        ):
            bad = {n: a.copy() for n, a in self.before.items()}
            bad[name][index] = value
            with self.assertRaises(ValueError):
                validate_case(l, self.case, bad)
        bad = dict(self.case, new_base=self.case["old_base"])
        with self.assertRaises(ValueError):
            validate_case(l, bad, self.before)
        bad = dict(self.before, OldCounts=self.before["OldCounts"][:-1])
        with self.assertRaises(ValueError):
            validate_case(l, self.case, bad)
        bad = copy.deepcopy(l)
        bad["state"]["pose_b"] = bad["state"]["pose_a"]
        with self.assertRaises(ValueError):
            validate_case(bad, self.case, self.before)
        bad = {n: a.copy() for n, a in self.before.items()}
        bad["ColliderPoses"].view(np.float32)[l["pose_words"] + l["pose_rotation"]] = (
            np.nan
        )
        with self.assertRaises(ValueError):
            validate_case(l, self.case, bad)
        self.load("seed_colors")
        bad = {n: a.copy() for n, a in self.before.items()}
        bad["Links"][l["link"]["previous_constraint"]] = self.case["old_storage"]
        with self.assertRaises(ValueError):
            validate_case(l, self.case, bad)

    def test_every_output_word_is_exact(self):
        self.load()
        l = self.layout
        for name, index in (
            ("Links", l["link"]["previous_constraint"]),
            ("States", self.case["new_base"] * l["state_words"] + l["pose_padding"]),
            ("Colors", 2),
            ("Colored", 2),
            ("Links", -1),
            ("States", -1),
            ("Contacts", 0),
            ("Params", 0),
            ("OldLinks", 0),
        ):
            self.actual[name][index] ^= np.uint32(1)
            with self.assertRaises(AssertionError):
                self.check()
            self.actual[name][index] ^= np.uint32(1)


if __name__ == "__main__":
    unittest.main()
