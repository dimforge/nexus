"""General scene ABI, deferred ordering, and continuous physical regressions."""

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from general_scene_physics import check as check_physics
from general_scene_validation import BUFFERS, check_outputs, validate_case
from replay import HERE


class GeneralSceneTests(unittest.TestCase):
    def fixture(self, name="general_mesh-c1-l1-b1-f120"):
        root = Path(
            os.environ.get(
                "NEXUS_TILELANG_GENERAL_SCENE_FIXTURES",
                Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
                / "general_scene/c/fixtures",
            )
        )
        layout = json.loads((root / "layout.json").read_text())
        d = root / "general_scene" / name
        case = json.loads((d / "case.json").read_text())
        contract = json.loads((HERE / "contracts/general_scene.json").read_text())
        before = {n: np.fromfile(d / f"{n}.bin", np.uint32) for n in BUFFERS}
        expected = {
            n: np.fromfile(d / f"{n}.expected.bin", np.uint32)
            if n in contract["outputs"]
            else v.copy()
            for n, v in before.items()
        }
        return layout, case, before, expected, contract

    def test_all_native_general_scenes(self):
        k = json.loads((HERE / "contracts/general_scene.json").read_text())
        for name in k["required_cases"]:
            with self.subTest(case=name):
                l, c, b, e, k = self.fixture(name)
                check_outputs(l, c, b, e, e, k)

    def test_readonly_and_capacity_guards(self):
        for name in (
            "GeometryIndices",
            "PfmPairs",
            "PfmIdentity",
            "PfmOrder",
            "PfmKeys",
            "PfmSortedKeys",
            "Trajectory",
            "FrameIndex",
        ):
            l, c, b, e, k = self.fixture()
            a = {n: v.copy() for n, v in e.items()}
            a[name][-1] ^= 1
            with self.subTest(buffer=name), self.assertRaises(AssertionError):
                check_outputs(l, c, b, a, e, k)

    def test_malformed_geometry_and_fixture_assumptions(self):
        for mutation in (
            "bvh",
            "mass",
            "local_pose",
            "filter",
            "material",
            "combine",
            "identity",
        ):
            l, c, b, _e, _k = self.fixture()
            if mutation == "bvh":
                ib = int(b["Shapes"][l["shape"]["a"] + 1])
                b["GeometryIndices"][ib] = 0
            if mutation == "mass":
                b["LocalMass"].view(np.float32)[
                    l["local_words"] + l["local_inv_inertia"]
                ] = 5
            if mutation == "local_pose":
                b["LocalPoses"].view(np.float32)[l["pose_translation"]] = 0.1
            if mutation == "filter":
                b["Filters"][1] = 1
            if mutation == "material":
                b["Materials"].view(np.float32)[l["material"]["friction"]] = np.nan
            if mutation == "combine":
                b["Materials"][l["material"]["friction_combine_rule"]] = 9
            if mutation == "identity":
                b["PfmIdentity"][1] = 0
            with (
                self.subTest(mutation=mutation),
                self.assertRaises((ValueError, AssertionError)),
            ):
                validate_case(l, c, b)
        l, c, b, _e, _k = self.fixture("general_triangle-c1-l1-b1-f1")
        b["Shapes"].view(np.float32)[l["shape_words"] + l["shape"]["a"]] = -0.4
        with self.assertRaisesRegex(ValueError, "triangle changed"):
            validate_case(l, c, b)

    def test_matching_corrupt_work_order_is_not_a_valid_reference(self):
        for name in ("PfmOrder", "PfmSortedKeys", "PfmPairs"):
            l, c, b, e, k = self.fixture()
            self.assertGreater(int(e["PfmCount"][0]), 1)
            if name == "PfmOrder":
                e[name][1] = e[name][0]
            elif name == "PfmSortedKeys":
                e[name][0] ^= 1
            else:
                e[name][l["pfm_pair"]["pair_index"]] = c["pair_capacity"]
            with self.subTest(buffer=name), self.assertRaises(AssertionError):
                check_outputs(l, c, b, e, e, k)

    def test_matching_bad_contact_geometry_rejected(self):
        for mutation in ("normal", "point"):
            l, c, b, e, k = self.fixture()
            cf, mf = l["contact"], l["manifold"]
            w = l["contact_words"]
            self.assertGreater(e["Contacts"][cf["contact"] + mf["len"]], 0)
            off = cf["contact"] + (
                mf["normal_a"]
                if mutation == "normal"
                else mf["points_a"] + l["manifold_point"]["pt"]
            )
            indices = [("Contacts", 0), ("RawContacts", int(e["InOrder"][0]))]
            for name, row in indices:
                f = e[name].view(np.float32)
                if mutation == "normal":
                    f[row * w + off : row * w + off + 3] *= 2
                else:
                    f[row * w + off] += 100
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                check_outputs(l, c, b, e, e, k)

    def test_intermediate_frame_physics_is_independent_of_reference(self):
        for mutation in ("static", "penetration", "rotation", "energy", "nonfinite"):
            l, c, b, e, k = self.fixture()
            f = e["Trajectory"].view(np.float32)
            p = l["pose_words"] + l["pose_translation"]
            if mutation == "static":
                f[l["pose_translation"]] += 0.1
            if mutation == "penetration":
                f[p + 1] = -1
            if mutation == "rotation":
                f[l["pose_words"] + l["pose_rotation"] + 3] = 2
            if mutation == "energy":
                f[
                    c["body_count"] * l["pose_words"]
                    + l["velocity_words"]
                    + l["velocity_linear"]
                ] = 20
            if mutation == "nonfinite":
                f[p] = np.nan
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                check_physics(l, c, b, e, k)

    def test_final_trajectory_copy_is_exact(self):
        l, c, b, e, k = self.fixture()
        a = {n: v.copy() for n, v in e.items()}
        a["Trajectory"][
            (c["frames"] - 1) * c["trace_stride"]
            + l["pose_words"]
            + l["pose_translation"]
        ] ^= 1
        with self.assertRaises(AssertionError):
            check_outputs(l, c, b, a, e, k)


if __name__ == "__main__":
    unittest.main()
