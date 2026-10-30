"""Connected scene guard, ownership, physical and persistence regressions."""

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scene_replay as scene
from replay import HERE
from scene_physics import check_physics
from scene_validation import READONLY


class SceneTests(unittest.TestCase):
    def fixture(self, name="stack-c1-l3-b1-f1"):
        root = Path(
            os.environ.get(
                "NEXUS_TILELANG_SCENE_FIXTURES",
                Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
                / "analytic_scene/c/fixtures",
            )
        )
        l = json.loads((root / "layout.json").read_text())
        c = json.loads((root / "analytic_scene" / name / "case.json").read_text())
        k = json.loads((HERE / "contracts/analytic_scene.json").read_text())
        d = root / "analytic_scene" / name
        b = {n: np.fromfile(d / f"{n}.bin", np.uint32) for n in k["inputs"]}
        e = {
            n: np.fromfile(d / f"{n}.expected.bin", np.uint32)
            if n in k["outputs"]
            else v.copy()
            for n, v in b.items()
        }
        return l, c, b, e, k

    def test_every_native_scene_satisfies_independent_invariants(self):
        k = json.loads((HERE / "contracts/analytic_scene.json").read_text())
        for name in k["required_cases"]:
            with self.subTest(case=name):
                l, c, b, e, k = self.fixture(name)
                scene.check_outputs(l, c, b, e, e, k)

    def test_matching_corrupt_intermediate_states_fail_physics(self):
        for mutation in ("static", "penetration", "energy", "rotation", "nonfinite"):
            l, c, b, e, k = self.fixture("free-c1-l1-b1-f120")
            f = e["Trajectory"].view(np.float32)
            start = 13 * c["trace_stride"]
            pose = start + l["pose_words"]
            velocity = start + c["body_count"] * l["pose_words"] + l["velocity_words"]
            if mutation == "static":
                f[start + l["pose_translation"]] = 0.1
            if mutation == "penetration":
                f[pose + l["pose_translation"] + 1] = -10
            if mutation == "energy":
                f[velocity + l["velocity_linear"]] = 1000
            if mutation == "rotation":
                f[pose + l["pose_rotation"] + 3] = 2
            if mutation == "nonfinite":
                f[pose + l["pose_translation"]] = np.nan
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                scene.check_outputs(l, c, b, e, e, k)

    def test_trace_cursor_padding_and_final_state_are_exact(self):
        for mutation in ("cursor", "padding", "final", "guard"):
            l, c, b, e, k = self.fixture("free-c1-l1-b1-f120")
            a = {n: v.copy() for n, v in e.items()}
            if mutation == "cursor":
                a["FrameIndex"][0] -= 1
            if mutation == "padding":
                a["Trajectory"][13 * c["trace_stride"] + l["pose_padding"]] ^= 1
            if mutation == "final":
                a["Trajectory"][
                    119 * c["trace_stride"] + l["pose_words"] + l["pose_translation"]
                ] ^= 1
            if mutation == "guard":
                a["Trajectory"][-1] ^= 1
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                scene.check_outputs(l, c, b, a, e, k)

    def test_readonly_and_capacity_guards_remain_exact(self):
        mutations = [(name, 0) for name in READONLY] + [
            (name, -1)
            for name in (
                "Tiles",
                "OldTiles",
                "Poses",
                "ColliderPoses",
                "SolverVelocities",
                "Contacts",
                "RawContacts",
                "Links",
                "OldLinks",
                "SortedLinks",
                "Indices",
                "OldIndices",
                "Colors",
                "OldColors",
                "States",
                "Pairs",
                "PairCount",
                "PfmCount",
                "Plan",
                "AdjAux0",
                "BucketAux0",
                "Reduced",
                "RadixCounts",
            )
        ]
        for name, index in mutations:
            with self.subTest(buffer=name, index=index):
                l, c, b, e, k = self.fixture()
                a = {n: v.copy() for n, v in e.items()}
                a[name][index] ^= 1
                with self.assertRaises(AssertionError):
                    scene.check_outputs(l, c, b, a, e, k)

    def test_exact_identity_and_membership_errors_are_not_float_tolerances(self):
        for mutation in (
            "pair",
            "contact",
            "adjacency",
            "bucket",
            "index",
            "state_id",
            "tile_id",
            "copy",
        ):
            with self.subTest(mutation=mutation):
                l, c, b, e, k = self.fixture()
                a = {n: v.copy() for n, v in e.items()}
                if mutation == "pair":
                    a["Pairs"][0] = c["body_count"]
                if mutation == "contact":
                    a["Contacts"][l["contact"]["bodies"]] ^= 1
                if mutation == "adjacency":
                    a["SortedIds"][0] ^= 1
                    a["OldIds"] = a["SortedIds"].copy()
                if mutation == "bucket":
                    a["Buckets"][1] += 1
                if mutation == "index":
                    a["Indices"][0] = c["contact_capacity"]
                    a["OldIndices"] = a["Indices"].copy()
                if mutation == "state_id":
                    off = l["state"]["colliders"]
                    for half in (0, c["contact_storage"]):
                        a["States"][half * l["state_words"] + off] ^= 1
                if mutation == "tile_id":
                    a["Tiles"][l["header"]["body_a"] + 3] ^= 1
                    a["OldTiles"] = a["Tiles"].copy()
                if mutation == "copy":
                    a["OldLinks"][l["link"]["contact"]] ^= 1
                with self.assertRaises((AssertionError, ValueError, KeyError)):
                    scene.check_outputs(l, c, b, a, e, k)

    def test_invalid_schedule_and_abi_rejected_before_launch(self):
        for field in (
            "body_storage",
            "contact_storage",
            "pair_capacity",
            "frames",
            "substeps",
            "pgs",
            "rounds",
            "new_base",
            "batches",
        ):
            l, c, b, _e, _k = self.fixture()
            c[field] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                scene.validate_case(l, c, b)
        for buffer, index in (
            ("BatchIds", 0),
            ("Mode", 0),
            ("Color1", 0),
            ("Config2", 0),
            ("AdjStride0", 0),
        ):
            l, c, b, _e, _k = self.fixture()
            b[buffer][index] += 1
            with self.subTest(buffer=buffer), self.assertRaises(ValueError):
                scene.validate_case(l, c, b)

    def test_physics_rejects_matching_but_unphysical_oracle_and_target(self):
        for mutation in ("static", "energy", "penetration", "rotation", "nonfinite"):
            l, c, b, e, k = self.fixture()
            if mutation == "static":
                e["Poses"][l["pose_translation"]] = np.float32(0.1).view(np.uint32)
            if mutation == "energy":
                e["Velocities"][l["velocity_words"] + l["velocity_linear"]] = (
                    np.float32(20.0).view(np.uint32)
                )
            if mutation == "penetration":
                e["Poses"][l["pose_words"] + l["pose_translation"] + 1] = np.float32(
                    0.3
                ).view(np.uint32)
            if mutation == "rotation":
                e["Poses"][l["pose_words"] + l["pose_rotation"] + 3] = np.float32(
                    2.0
                ).view(np.uint32)
            if mutation == "nonfinite":
                e["Velocities"][l["velocity_words"] + l["velocity_linear"]] = (
                    np.float32(np.nan).view(np.uint32)
                )
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                check_physics(l, c, b, e, k)

    def test_palette_exhaustion_cannot_pass_as_completed_scene(self):
        l, c, b, e, k = self.fixture()
        e["Colors"][0] = 0
        e["OldColors"][0] = 0
        with self.assertRaisesRegex(AssertionError, "unsolved"):
            scene.check_outputs(l, c, b, e, e, k)

    def test_small_arithmetic_roundoff_is_accepted(self):
        l, c, b, e, k = self.fixture()
        a = {n: v.copy() for n, v in e.items()}
        index = l["pose_words"] + l["pose_translation"]
        a["Poses"].view(np.float32)[index] += np.float32(1e-7)
        a["Trajectory"][(c["frames"] - 1) * c["trace_stride"] + index] = a["Poses"][
            index
        ]
        scene.check_outputs(l, c, b, a, e, k)

    def test_free_checkpoint_has_independent_gravity_solution(self):
        l, c, b, e, _k = self.fixture("free-c1-l1-b1-f1")
        dt = float(b["RbdParams"].view(np.float32)[l["rbd_params"]["dt"]])
        steps = c["substeps"]
        gravity = b["Gravity"].view(np.float32)[:3].astype(np.float64)
        po = l["pose_words"] + l["pose_translation"]
        vo = l["velocity_words"] + l["velocity_linear"]
        initial = b["Poses"].view(np.float32)[po : po + 3].astype(np.float64)
        np.testing.assert_allclose(
            e["Velocities"].view(np.float32)[vo : vo + 3],
            gravity * dt * steps,
            atol=2e-6,
            rtol=2e-5,
        )
        np.testing.assert_allclose(
            e["Poses"].view(np.float32)[po : po + 3],
            initial + gravity * dt * dt * steps * (steps + 1) / 2,
            atol=2e-6,
            rtol=2e-5,
        )

    def test_atomic_bucket_permutation_is_accepted_by_identity(self):
        from scene_outputs import tile_words

        l, c, b, e, k = self.fixture("friction-c3-l1-b1-f1")
        a = {n: v.copy() for n, v in e.items()}
        left, right = map(int, a["Indices"][:2])
        links = a["SortedLinks"].reshape(-1, l["link_words"])
        links[[left, right]] = links[[right, left]]
        a["Indices"][[0, 1]] = a["Indices"][[1, 0]]
        a["OldIndices"] = a["Indices"].copy()
        li, ri = list(tile_words(l, left)), list(tile_words(l, right))
        a["Tiles"][li + ri] = a["Tiles"][ri + li]
        a["OldTiles"] = a["Tiles"].copy()
        scene.check_outputs(l, c, b, a, e, k)

    def test_cache_roundoff_policy_does_not_cover_large_or_single_frame_errors(self):
        for frames in (1, 120):
            l, c, b, e, k = self.fixture(f"boxes-c2-l3-b1-f{frames}")
            a = {n: v.copy() for n, v in e.items()}
            slot = int(a["Indices"][0])
            off = (
                slot // 32 * l["tile_words"]
                + l["warmstart"]
                + slot % 32 * 2 * l["velocity_words"]
                + l["velocity_words"]
                + l["velocity_angular"]
                + 2
            )
            a["Tiles"].view(np.float32)[off] += np.float32(
                3e-6 if frames == 1 else 2e-5
            )
            a["OldTiles"] = a["Tiles"].copy()
            with self.subTest(frames=frames), self.assertRaises(AssertionError):
                scene.check_outputs(l, c, b, a, e, k)
