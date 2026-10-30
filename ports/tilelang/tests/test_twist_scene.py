"""Independent compact geometry, integer ownership and ABI regressions in native scenes."""

import copy
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay import HERE
from runtime_scene_validation import check_outputs, validate_case
from twist_scene_validation import check_constraints


class TwistSceneTests(unittest.TestCase):
    def fixture(self):
        root = Path(
            os.environ.get(
                "NEXUS_TILELANG_TWIST_SCENE_FIXTURES",
                Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
                / "runtime_twist_scene/c/fixtures",
            )
        )
        name = os.environ.get(
            "NEXUS_TILELANG_TWIST_SCENE_CASE", "mesh-contact-joint-f1"
        )
        d = root / "runtime_twist_scene" / name
        l = json.loads((root / "layout.json").read_text())
        c = json.loads((d / "case.json").read_text())
        b = {n: np.fromfile(d / f"{n}.bin", np.uint32) for n in c["buffer_words"]}
        e = {
            n: np.fromfile(d / f"{n}.expected.bin", np.uint32)
            if (d / f"{n}.expected.bin").exists()
            else v.copy()
            for n, v in b.items()
        }
        k = json.loads((HERE / "contracts/runtime_twist_scene.json").read_text())
        return l, c, b, e, k

    def contact(self, l, e):
        count = int(e["Plan"][l["plan"]["bound"]])
        cid = next(
            i
            for i in range(count)
            if e["Links"][i * l["link_words"] + l["link"]["len"]]
        )
        slot = int(e["Indices"][cid])
        return count, slot // 32 * l["tile_words"] + slot % 32 * 4, slot

    def test_native_expected_passes_full_scene_checks(self):
        l, c, b, e, k = self.fixture()
        check_outputs(l, c, b, e, e, k)

    def test_model_and_layout_must_agree(self):
        l, c, b, _, _ = self.fixture()
        b["RbdParams"][l["rbd_params"]["friction_model"]] = l["friction_models"][
            "coulomb"
        ]
        with self.assertRaisesRegex(ValueError, "ABI differs"):
            validate_case(l, c, b)

    def test_shared_wrong_response_is_independently_rejected(self):
        for field in ("center", "mass", "radius", "twist_mass", "tangent_mass"):
            l, c, _, e, k = self.fixture()
            count, base, _ = self.contact(l, e)
            off = {
                "center": l["twist"]["center_a"],
                "mass": l["twist"]["arm_a"] + 3,
                "radius": l["twist"]["radius"],
                "twist_mass": l["header"]["diag_b"] + 3,
                "tangent_mass": l["twist"]["offset_b"] + 3,
            }[field]
            e["Tiles"].view(np.float32)[base + off] += 0.1
            with self.subTest(field=field), self.assertRaises(AssertionError):
                check_constraints(l, c, e, e, k, count)

    def test_wrong_cache_tags_and_padding_fail_even_in_both_oracles(self):
        for field in ("tag", "padding", "impulse"):
            l, c, _, e, k = self.fixture()
            count, base, slot = self.contact(l, e)
            if field == "tag":
                cache = (
                    slot // 32 * l["tile_words"]
                    + l["warmstart"]
                    + slot % 32 * 2 * l["velocity_words"]
                )
                e["Tiles"][cache + l["velocity_padding1"]] = 0xFFFFFFFF
            elif field == "padding":
                e["Tiles"][base + l["twist"]["friction"] + 3] = 1
            else:
                e["Tiles"].view(np.float32)[base + l["twist"]["friction"]] = 1e6
            with self.subTest(field=field), self.assertRaises(AssertionError):
                check_constraints(l, c, e, e, k, count)

    def test_outer_tile_guard_and_published_state_are_exact(self):
        l, c, b, e, k = self.fixture()
        for name in ("OldTiles", "Poses"):
            a = copy.deepcopy(e)
            a[name][-1] ^= 1
            with self.subTest(name=name), self.assertRaises(AssertionError):
                check_outputs(l, c, b, a, e, k)


if __name__ == "__main__":
    unittest.main()
