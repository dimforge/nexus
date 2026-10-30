"""Compact twist preparation regression tests using current native fixture exports."""

import copy
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contact_replay import BUFFERS, check_outputs, validate_case
from replay import HERE


class TwistPrepareTests(unittest.TestCase):
    def fixture(self, name="n1-recycled"):
        root = Path(
            os.environ.get(
                "NEXUS_TILELANG_TWIST_PREPARE_FIXTURES",
                Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
                / "twist_prepare/c/fixtures",
            )
        )
        directory = root / "twist_prepare" / name
        l = json.loads((root / "layout.json").read_text())
        c = next(
            c
            for c in json.loads((root / "twist_prepare/suite.json").read_text())[
                "cases"
            ]
            if c["name"] == name
        )
        b = {n: np.fromfile(directory / f"{n}.bin", np.uint32) for n in BUFFERS}
        e = {
            n: np.fromfile(directory / f"{n}.expected.bin", np.uint32)
            if (directory / f"{n}.expected.bin").exists()
            else a.copy()
            for n, a in b.items()
        }
        k = json.loads((HERE / "contracts/twist_prepare.json").read_text())
        return l, c, b, e, k

    def test_all_current_oracles_validate(self):
        contract = json.loads((HERE / "contracts/twist_prepare.json").read_text())
        for name in contract["required_cases"]:
            l, c, b, e, k = self.fixture(name)
            check_outputs(l, c, b, e, e, k)

    def test_recycled_geometry_impulses_and_tags_are_exact(self):
        l, c, b, e, k = self.fixture()
        for off in [
            l["twist"]["arm_a"],
            l["twist"]["frame_a"],
            l["twist"]["normal_impulse"],
            l["warmstart"] + l["velocity_padding1"],
            l["twist"]["friction"] + 3,
        ]:
            bad = copy.deepcopy(e)
            bad["Tiles"][off] ^= np.uint32(1)
            with self.assertRaises(AssertionError):
                check_outputs(l, c, b, bad, e, k)

    def test_empty_recycled_constraint_keeps_zero_cached_velocity(self):
        l, c, b, e, k = self.fixture("n1-recycled-empty")
        check_outputs(l, c, b, e, e, k)
        self.assertEqual(int(e["Tiles"][l["header"]["normal"] + 3]), 0)
        cache = (
            e["Tiles"]
            .view(np.float32)[l["warmstart"] : l["warmstart"] + 2 * l["velocity_words"]]
            .reshape(2, -1)
        )
        for field in ("velocity_linear", "velocity_angular"):
            np.testing.assert_array_equal(
                cache[:, l[field] : l[field] + 3], np.zeros((2, 3), np.float32)
            )

    def test_invalid_recycled_frame_is_rejected(self):
        l, c, b, _, _ = self.fixture()
        b["OldTiles"].view(np.float32)[
            l["twist"]["frame_a"] : l["twist"]["frame_a"] + 4
        ] = 0
        with self.assertRaises(ValueError):
            validate_case(l, c, b)

    def test_nearest_anchor_preserves_distinct_normal_impulses(self):
        l, _, _, e, _ = self.fixture("n1-nearest")
        actual = e["Tiles"].view(np.float32)[
            l["twist"]["normal_impulse"] : l["twist"]["normal_impulse"] + 4
        ]
        np.testing.assert_array_equal(
            actual, np.asarray([0.15, 0.3, 0.45000002, 0.6], np.float32)
        )

    def test_fresh_target_velocity_uses_link_restitution_and_compact_offset(self):
        l, _, b, e, _ = self.fixture("n1-fresh")
        h, t, link = l["header"], l["twist"], l["link"]
        raw = e["Tiles"]
        f = raw.view(np.float32).astype(float)
        v = (
            b["Velocities"]
            .view(np.float32)
            .reshape(-1, l["velocity_words"])
            .astype(float)
        )
        a = v[int(raw[h["body_a"] + 3])]
        bb = v[int(raw[h["body_b"] + 3])]
        normal = f[h["normal"] : h["normal"] + 3]
        restitution = float(b["Links"].view(np.float32)[link["restitution"]])
        for k in range(int(raw[h["normal"] + 3])):
            arm = t["arm_a"] + k * l["twist_arm_words"]
            ra = f[arm : arm + 3]
            rb = ra + f[t["offset_b"] : t["offset_b"] + 3]
            expected = restitution * np.dot(
                a[l["velocity_linear"] : l["velocity_linear"] + 3]
                + np.cross(a[l["velocity_angular"] : l["velocity_angular"] + 3], ra)
                - bb[l["velocity_linear"] : l["velocity_linear"] + 3]
                - np.cross(bb[l["velocity_angular"] : l["velocity_angular"] + 3], rb),
                normal,
            )
            np.testing.assert_allclose(
                f[t["normal_vel"] + k], expected, atol=2e-6, rtol=2e-5
            )


if __name__ == "__main__":
    unittest.main()
