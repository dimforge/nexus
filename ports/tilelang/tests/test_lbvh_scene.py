"""Connected LBVH scenes require independent spatial correctness and guards."""

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lbvh_scene_validation import check_outputs, validate_case
from replay import HERE


class LbvhSceneTests(unittest.TestCase):
    def fixture(self, kind="general", name="general_mixed-c7-l1-b2-f8"):
        operation = kind + "_lbvh_scene"
        root = Path(
            os.environ.get(
                "NEXUS_TILELANG_LBVH_SCENE_FIXTURES",
                Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
                / operation
                / "c/fixtures",
            )
        )
        directory = root / operation / name
        layout = json.loads((root / "layout.json").read_text())
        case = json.loads((directory / "case.json").read_text())
        contract = json.loads((HERE / "contracts" / (operation + ".json")).read_text())
        before = {
            n: np.fromfile(directory / (n + ".bin"), np.uint32)
            for n in contract["inputs"]
        }
        expected = {
            n: np.fromfile(directory / (n + ".expected.bin"), np.uint32)
            if n in contract["outputs"]
            else v.copy()
            for n, v in before.items()
        }
        return layout, case, before, expected, contract

    def test_native_convex_composite_batched_and_long_joint_scenes(self):
        for kind, name in [
            ("general", "general_capsule-c1-l1-b1-f1"),
            ("general", "general_mesh-c1-l1-b1-f1"),
            ("general", "general_mixed-c7-l1-b2-f8"),
            ("general", "general_capsule-c1-l1-b1-f120"),
            ("joint", "joint_chain-c1-l3-b1-f120"),
        ]:
            with self.subTest(scene=name):
                l, c, b, e, k = self.fixture(kind, name)
                check_outputs(l, c, b, e, e, k)

    def test_matching_tree_corruption_does_not_pass(self):
        for field in ("left", "first_leaf", "escape", "aabb"):
            l, c, b, e, k = self.fixture()
            offset = l["lbvh_node"][field]
            e["LbvhTree"][offset] ^= 1
            with (
                self.subTest(field=field),
                self.assertRaises((AssertionError, ValueError, IndexError)),
            ):
                check_outputs(l, c, b, e, e, k)

    def test_matching_bad_morton_and_sort_output_is_rejected(self):
        for name in ("LbvhKeys", "LbvhSorted", "LbvhSortKeys"):
            l, c, b, e, k = self.fixture()
            e[name][0] ^= 1
            with (
                self.subTest(buffer=name),
                self.assertRaises((AssertionError, ValueError)),
            ):
                check_outputs(l, c, b, e, e, k)

    def test_readonly_and_tail_writes_are_rejected(self):
        for name in (
            "LbvhIdentity",
            "LbvhTree",
            "LbvhFrontier",
            "LbvhPartials",
            "LbvhConfig0",
            "LbvhNumKeys",
        ):
            l, c, b, e, k = self.fixture()
            e[name][-1] ^= 1
            with (
                self.subTest(buffer=name),
                self.assertRaises((AssertionError, ValueError)),
            ):
                check_outputs(l, c, b, e, e, k)

    def test_shape_storage_or_broadphase_selection_mismatch_is_rejected(self):
        l, c, b, _, _ = self.fixture()
        c["broadphase"] = "brute_force"
        with self.assertRaisesRegex(ValueError, "LBVH scheduling"):
            validate_case(l, c, b)
        c["broadphase"] = "lbvh"
        b["LbvhNumKeys"][0] -= 1
        with self.assertRaises(AssertionError):
            validate_case(l, c, b)
