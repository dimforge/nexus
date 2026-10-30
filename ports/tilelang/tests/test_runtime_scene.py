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
    def fixture(self, name=None):
        root = Path(
            os.environ.get(
                "NEXUS_TILELANG_RUNTIME_FIXTURES",
                Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
                / "runtime_scene/c/fixtures",
            )
        )
        name = name or os.environ.get(
            "NEXUS_TILELANG_RUNTIME_CASE", "mesh-contact-joint-f1"
        )
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
        k = json.loads((HERE / "contracts/runtime_scene.json").read_text())
        k.pop("numeric_policies", None)
        return l, c, b, e, k

    def test_hub_pool_uses_native_allocation_not_guard_rows(self):
        l, c, b, e, k = self.fixture("large-batch-storage-f1")
        self.assertEqual(c["native_body_slots"], 1026)
        self.assertEqual(c["hub_pool"], 1026)
        self.assertGreater(c["body_storage"], c["hub_pool"])
        check_outputs(l, c, b, e, e, k)
        # The world-AABB sub-check must accept more than 64 independent batches,
        # but the ordinary exact native capacity metadata remains mandatory.
        self.assertEqual(c["batches"], 513)
        c["hub_pool"] = c["body_storage"]
        with self.assertRaises(ValueError):
            validate_case(l, c, b)

    def test_native_split_hubs_and_corrupt_ownership(self):
        for name, expected_hubs in (("split-hub-f8", 1), ("split-hubs-f8", 2)):
            l, c, b, e, k = self.fixture(name)
            self.assertEqual(int(e["HubCounts"][l["hub"]["hubs"]]), expected_hubs)
            self.assertEqual(int(e["HubCounts"][l["hub"]["slots"]]), expected_hubs * 33)
            check_outputs(l, c, b, e, e, k)
        for mutation in (
            "slot_body",
            "slot_constraint",
            "reservation",
            "scale",
            "tile_slot",
            "tail_counts",
            "solver_guard",
        ):
            l, c, b, e, k = self.fixture("split-hubs-f1")
            if mutation == "slot_body":
                e["SlotBody"][0] = c["body_count"]
            elif mutation == "slot_constraint":
                e["SlotConstraint"][0] = c["contact_capacity"]
            elif mutation == "reservation":
                e["FirstSlot"][int(e["HubList"][1])] = e["FirstSlot"][
                    int(e["HubList"][0])
                ]
            elif mutation == "scale":
                e["Links"].view(np.float32)[l["link"]["mass_scale_a"]] = 1
            elif mutation == "tile_slot":
                e["Tiles"][l["header"]["diag_a"] + 3] = 0
            elif mutation == "tail_counts":
                e["Counts"][-1] = 0
            else:
                e["SolverVelocities"][-1] ^= 1
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                check_outputs(l, c, b, e, e, k)

    def test_valid_hub_reservation_permutation(self):
        l, c, b, e, k = self.fixture("split-hubs-f1")
        a = {n: v.copy() for n, v in e.items()}
        base = c["body_storage"]
        degree = 33
        # Exchange two complete reservations, preserving each exact owner and constraint.
        for name, width in (
            ("SlotBody", 1),
            ("SlotConstraint", 1),
            ("SolverVelocities", l["velocity_words"]),
        ):
            rows = a[name].reshape(-1, width)
            start = base if name == "SolverVelocities" else 0
            rows[start : start + 2 * degree] = (
                rows[start : start + 2 * degree]
                .reshape(2, degree, width)[::-1]
                .reshape(2 * degree, width)
            )

        def remap(slot):
            return (
                slot + degree
                if base <= slot < base + degree
                else slot - degree
                if base + degree <= slot < base + 2 * degree
                else slot
            )

        for body in a["HubList"][:2]:
            a["FirstSlot"][body] = remap(int(a["FirstSlot"][body]) + base) - base
        a["HubList"][:2] = a["HubList"][:2][::-1]
        for name in ("Links", "OldLinks", "SortedLinks"):
            rows = a[name].reshape(-1, l["link_words"])
            for row in rows[: int(a["Plan"][l["plan"]["bound"]])]:
                if row[l["link"]["len"]]:
                    for field in ("vel_slot_a", "vel_slot_b"):
                        row[l["link"][field]] = remap(int(row[l["link"][field]]))
        for name in ("Tiles", "OldTiles"):
            for cid in range(int(a["Plan"][l["plan"]["bound"]])):
                slot = int(a["Indices"][cid])
                start = slot // 32 * l["tile_words"] + slot % 32 * 4
                for field in ("diag_a", "off_a"):
                    i = start + l["header"][field] + 3
                    a[name][i] = remap(int(a[name][i]))
        check_outputs(l, c, b, a, e, k)

    def test_native_hub_input_and_body_group_validation(self):
        for mutation in ("body_group", "pool", "solver_storage"):
            l, c, b, _, _ = self.fixture()
            if mutation == "body_group":
                b["BodyGroup"][1] = 0
            elif mutation == "pool":
                c["hub_pool"] -= 1
            else:
                c["solver_storage"] -= 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_case(l, c, b)

    def test_padded_body_storage_does_not_overlaunch_active_body_kernels(self):
        from ordered import validate_launch
        from scene_kernels import make_program

        l, c, _, _, _ = self.fixture("many-batches-f8")
        self.assertEqual(c["body_count"], 48)
        self.assertEqual(c["body_storage"], 120)
        plans = make_program(l, c, "metal")
        body_stages = {
            "update_mprops",
            "cleanup",
            "init_bodies",
            "init_increments",
            "apply_increments",
            "integrate",
            "finalize",
        }
        checked = set()
        for plan in plans:
            kind = plan["name"].rsplit("_", 1)[0]
            if kind in body_stages:
                checked.add(kind)
                self.assertEqual(plan["blocks"], 2 if kind == "cleanup" else 1)
                validate_launch(
                    plan["kernel"], "metal", plan["blocks"], plan["threads"]
                )
        self.assertEqual(checked, body_stages)

    def test_many_batch_prefix_scratch_and_guards(self):
        l, c, b, e, k = self.fixture("many-batches-f8")
        self.assertEqual(c["bucket_strides"], [264, 2, 1, 1])
        self.assertEqual(c["bucket_levels"], 2)
        check_outputs(l, c, b, e, e, k)
        for name in ("BucketAux0", "BucketAux1", "BucketAux2", "AdjAux0"):
            actual = {n: v.copy() for n, v in e.items()}
            actual[name][-1] ^= 1
            with self.subTest(buffer=name), self.assertRaises(AssertionError):
                check_outputs(l, c, b, actual, e, k)

    def test_reduction_mode_changes_composite_contacts(self):
        counts = []
        for mode in ("reduced", "unreduced"):
            l, c, b, e, k = self.fixture(f"mesh-overlap-{mode}-f1")
            self.assertEqual(c["contact_reduction"], mode == "reduced")
            self.assertEqual(c["rounds"], c["color_stride"] - 3)
            check_outputs(l, c, b, e, e, k)
            rows = e["Contacts"].reshape(-1, l["contact_words"])
            length = l["contact"]["contact"] + l["manifold"]["len"]
            counts.append(
                int(
                    np.count_nonzero(rows[: int(e["Plan"][l["plan"]["bound"]]), length])
                )
            )
        self.assertGreater(counts[1], counts[0])
        self.assertGreater(counts[0], 0)

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
        for mutation in (
            "batch_parent",
            "body_count",
            "schedule",
            "pose",
            "mass",
            "rounds",
            "reduction",
        ):
            l, c, b, _, _ = self.fixture()
            if mutation == "batch_parent":
                b["Parents"][0] = 1
            elif mutation == "body_count":
                c["body_count"] += 1
            elif mutation == "schedule":
                c["substeps"] += 1
            elif mutation == "pose":
                b["LocalPoses"].view(np.float32)[l["pose_rotation"] + 3] = 2
            elif mutation == "rounds":
                c["rounds"] += 1
            elif mutation == "reduction":
                c["contact_reduction"] = 1
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
