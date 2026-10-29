"""Bounds, batch layouts, filters and atomic overflow semantics."""

import json, os, sys, unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay import HERE
import broadphase_replay as broad


class BroadPhaseTests(unittest.TestCase):
    def fixture(self, op="brute_pairs", name="n17-b2-overflow"):
        root = (
            Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
            / op
            / "c/fixtures"
        )
        l = json.loads((root / "layout.json").read_text())
        c = next(
            c
            for c in json.loads((root / op / "suite.json").read_text())["cases"]
            if c["name"] == name
        )
        d = root / op / name
        k = json.loads((HERE / f"contracts/{op}.json").read_text())
        b = {n: np.fromfile(d / f"{n}.bin", dtype=np.uint32) for n in k["inputs"]}
        e = {
            n: np.fromfile(d / f"{n}.expected.bin", dtype=np.uint32)
            if n in k["outputs"]
            else a.copy()
            for n, a in b.items()
        }
        return l, c, b, e, k

    def test_all_broad_phase_fixtures(self):
        for op in broad.OPERATIONS:
            k = json.loads((HERE / f"contracts/{op}.json").read_text())
            for name in k["required_cases"]:
                with self.subTest(operation=op, case=name):
                    l, c, b, e, k = self.fixture(op, name)
                    broad.check_outputs(l, c, b, e, e, k)

    def test_overflow_allows_legal_alternate_reservations(self):
        l, c, b, e, k = self.fixture()
        allowed = sorted(broad.candidates(l, c, b, e["Aabbs"]))
        self.assertGreater(len(allowed), c["pair_capacity"])
        a = {n: v.copy() for n, v in e.items()}
        pairs = a["Pairs"].reshape(-1, l["pair_words"])
        off = l["pair_colliders"]
        pairs[: c["pair_capacity"], off : off + 2] = allowed[-c["pair_capacity"] :][
            ::-1
        ]
        broad.check_outputs(l, c, b, a, e, k)

    def test_duplicate_illegal_and_missing_pairs_are_rejected(self):
        for mutation in ("duplicate", "cross_batch", "self_pair", "truncated_counter"):
            l, c, b, e, k = self.fixture()
            a = {n: v.copy() for n, v in e.items()}
            p = a["Pairs"].reshape(-1, l["pair_words"])
            off = l["pair_colliders"]
            if mutation == "duplicate":
                p[1] = p[0]
            if mutation == "cross_batch":
                p[0, off + 1] ^= 1
            if mutation == "self_pair":
                p[0, off + 1] = p[0, off]
            if mutation == "truncated_counter":
                a["PairCount"][0] = c["pair_capacity"]
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                broad.check_outputs(l, c, b, a, e, k)

    def test_capacity_zero_still_counts_pairs_without_writes(self):
        l, c, b, e, k = self.fixture(name="n17-b2-zero-capacity")
        self.assertGreater(int(e["PairCount"][0]), 0)
        np.testing.assert_array_equal(e["Pairs"], b["Pairs"])
        broad.check_outputs(l, c, b, e, e, k)

    def test_append_preserves_prefix_and_all_trailing_guards(self):
        for name, index in [
            ("Pairs", 0),
            ("Pairs", -1),
            ("PairCount", -1),
            ("Groups", 0),
            ("Filters", 0),
            ("RbdParams", 0),
            ("BatchIds", 0),
        ]:
            l, c, b, e, k = self.fixture(name="n5-b2-append")
            a = {n: v.copy() for n, v in e.items()}
            a[name][index] ^= 1
            with (
                self.subTest(buffer=name, index=index),
                self.assertRaises(AssertionError),
            ):
                broad.check_outputs(l, c, b, a, e, k)

    def test_interaction_mode_enum_and_batch_uniforms_are_checked(self):
        for mutation in (
            "mode",
            "batches",
            "capacity",
            "negative_prediction",
            "counter",
            "shape",
        ):
            l, c, b, e, k = self.fixture()
            if mutation == "mode":
                b["Groups"][l["groups"]["test_mode"]] = 2
            if mutation == "batches":
                b["BatchIds"][l["batch"]["num_batches"]] += 1
            if mutation == "capacity":
                b["BatchIds"][l["batch"]["collision_pairs_capacity"]] += 1
            if mutation == "negative_prediction":
                b["RbdParams"][l["rbd_params"]["normalized_prediction_distance"]] = (
                    np.float32(-0.1).view(np.uint32)
                )
            if mutation == "counter":
                b["PairCount"][0] += 1
            if mutation == "shape":
                b["Aabbs"] = b["Aabbs"][:-1]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                broad.validate_case(l, c, b)

    def test_and_mode_takes_priority_over_or(self):
        l, c, b, e, k = self.fixture(name="n7-b3-groups")
        groups = b["Groups"].reshape(-1, l["group_words"])
        g = l["groups"]
        self.assertTrue(np.any(groups[: c["count"], g["test_mode"]] == 0))
        self.assertTrue(np.any(groups[: c["count"], g["test_mode"]] == 1))
        original = broad.candidates(l, c, b, e["Aabbs"])
        modified = {n: v.copy() for n, v in b.items()}
        modified["Groups"].reshape(groups.shape)[:, g["test_mode"]] = 1
        either = broad.candidates(l, c, modified, e["Aabbs"])
        self.assertTrue(original < either)

    def test_unknown_shape_has_default_origin_bounds(self):
        l, c, b, e, k = self.fixture("world_aabbs", "n1-b2-shape-9")
        boxes = e["Aabbs"].reshape(-1, l["aabb_words"])
        a = l["aabb"]
        for batch in range(c["batches"]):
            box = boxes[batch * c["capacity"]]
            for field in ("mins", "maxs"):
                np.testing.assert_array_equal(
                    box[a[field] : a[field] + 3], np.zeros(3, dtype=np.uint32)
                )
        self.assertTrue(
            np.any(
                b["Poses"].view(np.float32)[
                    l["pose_translation"] : l["pose_translation"] + 3
                ]
                != 0
            )
        )

    def test_bounds_padding_capacity_gaps_and_input_vertices_are_exact(self):
        for name, index in [
            ("Aabbs", -1),
            ("Aabbs", 7),
            ("Vertices", 0),
            ("Shapes", 0),
            ("Poses", 0),
        ]:
            l, c, b, e, k = self.fixture("world_aabbs", "n33-b3-mixed")
            a = {n: v.copy() for n, v in e.items()}
            if name == "Aabbs" and index == 7:
                index = l["aabb"]["padding1"]
            a[name][index] ^= 1
            with (
                self.subTest(buffer=name, index=index),
                self.assertRaises(AssertionError),
            ):
                broad.check_outputs(l, c, b, a, e, k)

    def test_convex_ranges_and_shape_geometry_are_validated(self):
        for mutation in (
            "first",
            "end",
            "nan_vertex",
            "rotation",
            "negative_radius",
            "inverted_root",
        ):
            name = (
                "n1-b2-shape-7"
                if mutation in ("first", "end", "nan_vertex", "rotation")
                else "n1-b2-shape-0"
                if mutation == "negative_radius"
                else "n1-b2-shape-6"
            )
            l, c, b, e, k = self.fixture("world_aabbs", name)
            f = l["shape"]
            if mutation == "first":
                b["Shapes"][f["a"]] = c["vertices"]
            if mutation == "end":
                b["Shapes"][f["a"] + 1] = c["vertices"] + 1
            if mutation == "nan_vertex":
                b["Vertices"][l["vertex_words"] + l["vertex_vector"]] = 0x7FC00000
            if mutation == "rotation":
                b["Poses"][l["pose_rotation"] + 3] = np.float32(2).view(np.uint32)
            if mutation == "negative_radius":
                b["Shapes"][f["a"]] = np.float32(-1).view(np.uint32)
            if mutation == "inverted_root":
                b["Shapes"][f["b"]] = np.float32(99).view(np.uint32)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                broad.validate_case(l, c, b)
