"""Public scene upload, separate index spaces and native mesh sentinel regressions."""

import copy
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_geometry import lbvh_vertices
from replay import HERE
from runtime_scene_validation import check_outputs, validate_case


class RuntimeSceneTests(unittest.TestCase):
    def fixture(self):
        root = Path(
            os.environ.get(
                "NEXUS_TILELANG_RUNTIME_FIXTURES",
                HERE / "results/runtime_scene/c/fixtures",
            )
        )
        name = os.environ.get("NEXUS_TILELANG_RUNTIME_CASE", "mesh-contact-joint-f1")
        directory = root / "runtime_scene" / name
        l = json.loads((root / "layout.json").read_text())
        c = json.loads((directory / "case.json").read_text())
        b = {
            n: np.fromfile(directory / f"{n}.bin", np.uint32) for n in c["buffer_words"]
        }
        e = {
            n: np.fromfile(directory / f"{n}.expected.bin", np.uint32)
            if (directory / f"{n}.expected.bin").exists()
            else v.copy()
            for n, v in b.items()
        }
        k = json.loads((HERE / "contracts/joint_scene.json").read_text())
        k.pop("numeric_policies", None)
        return l, c, b, e, k

    def test_real_multi_collider_upload_and_expected_outputs(self):
        l, c, b, e, k = self.fixture()
        self.assertNotEqual(c["body_count"], c["collider_count"])
        self.assertGreater(c["batches"], 1)
        check_outputs(l, c, b, e, e, k)

    def test_native_empty_leaf_bounds_are_not_modified(self):
        l, _, b, _, _ = self.fixture()
        original = b["Vertices"].copy()
        normalized = lbvh_vertices(l, b)
        self.assertFalse(np.isfinite(original.view(np.float32)).all())
        self.assertTrue(np.isfinite(normalized.view(np.float32)).all())
        np.testing.assert_array_equal(b["Vertices"], original)

    def test_nonfinite_physical_geometry_remains_rejected(self):
        for mutation in ("vertex", "leaf", "primitive_index"):
            l, _, b, _, _ = self.fixture()
            shapes = b["Shapes"].reshape(-1, l["shape_words"])
            sf = l["shape"]
            mesh = shapes[shapes[:, sf["a"] + 3] == l["shape_types"]["trimesh"]][0]
            vb, ib, nodes = map(int, mesh[sf["a"] : sf["a"] + 3])
            if mutation == "primitive_index":
                b["GeometryIndices"][ib + nodes * 3] = 0xFFFFFFFF
            else:
                row = vb + (2 * nodes if mutation == "vertex" else 0)
                b["Vertices"].view(np.float32)[
                    row * l["vertex_words"] + l["vertex_vector"]
                ] = np.nan
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                lbvh_vertices(l, b)

    def test_runtime_counts_schedule_and_parents_are_checked(self):
        for mutation in ("batch_parent", "body_count", "schedule", "pose", "mass"):
            l, c, b, _, _ = self.fixture()
            if mutation == "batch_parent":
                b["Parents"][0] = 1
            elif mutation == "body_count":
                c["body_count"] += 1
            elif mutation == "schedule":
                c["substeps"] += 1
            elif mutation == "pose":
                b["LocalPoses"].view(np.float32)[l["pose_rotation"] + 3] = 2
            else:
                b["LocalMass"].view(np.float32)[l["local_inv_mass"]] = -1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_case(l, c, b)

    def test_joint_color_cannot_alias_writers(self):
        l, c, b, _, _ = self.fixture()
        c = copy.deepcopy(c)
        n, width = c["joint_count"], l["impulse_joint_words"]
        c["joint_count"] *= 2
        c["joints_per_batch"] *= 2
        b["BatchIds"][l["batch"]["impulse_joints_len"]] *= 2
        b["JointColors"][0] *= 2
        b["Joints"][n * width : 2 * n * width] = b["Joints"][: n * width]
        with self.assertRaisesRegex(ValueError, "aliases a velocity writer"):
            validate_case(l, c, b)

    def test_matching_bad_fixed_pose_and_guard_are_rejected(self):
        for mutation in ("fixed_pose", "guard", "joint_id"):
            l, c, b, e, k = self.fixture()
            if mutation == "fixed_pose":
                # Body 1 in each batch is the native fixed joint endpoint.
                e["Trajectory"].view(np.float32)[
                    c["batches"] * l["pose_words"] + l["pose_translation"]
                ] += 1
            elif mutation == "guard":
                e["Poses"][-1] ^= 1
            else:
                e["JointConstraints"][l["joint_constraint"]["solver_vel_a"]] = c[
                    "body_count"
                ]
            with (
                self.subTest(mutation=mutation),
                self.assertRaises((ValueError, AssertionError)),
            ):
                check_outputs(l, c, b, e, e, k)


if __name__ == "__main__":
    unittest.main()
