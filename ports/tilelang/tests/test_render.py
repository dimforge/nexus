"""Native renderer ABI and mutation guards, including exact negative-zero colors."""

import copy
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from render_replay import BUFFERS, check_outputs, validate_case
from replay import HERE


class RenderTests(unittest.TestCase):
    def fixture(self, name="n65"):
        root = Path(
            os.environ.get(
                "NEXUS_TILELANG_RENDER_FIXTURES",
                HERE / "results-render-0097/prep_render/c/fixtures",
            )
        )
        d = root / "prep_render" / name
        l = json.loads((root / "layout.json").read_text())
        c = next(
            c
            for c in json.loads((root / "prep_render/suite.json").read_text())["cases"]
            if c["name"] == name
        )
        k = json.loads((HERE / "contracts/prep_render.json").read_text())
        b = {n: np.fromfile(d / f"{n}.bin", np.uint32) for n in BUFFERS}
        e = {
            n: np.fromfile(d / f"{n}.expected.bin", np.uint32)
            if n in k["outputs"]
            else v.copy()
            for n, v in b.items()
        }
        return l, c, b, e, k

    def test_every_native_case(self):
        for n in (0, 1, 31, 32, 33, 63, 64, 65, 129):
            l, c, b, e, k = self.fixture(f"n{n}")
            check_outputs(l, c, b, e, e, k)

    def test_indices_rotations_and_counts_are_checked(self):
        for mutation in ("body", "quaternion", "count", "scale"):
            l, c, b, _, _ = self.fixture()
            if mutation == "body":
                b["Descriptors"][l["descriptor"]["pose_index"]] = c["bodies"]
            elif mutation == "quaternion":
                b["Poses"][:] = 0
            elif mutation == "count":
                b["Count"][0] += 1
            else:
                b["Descriptors"][l["descriptor"]["scale"]] = 0x7FC00000
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_case(l, c, b)

    def test_colors_and_guard_bits_are_exact(self):
        l, c, b, e, k = self.fixture()
        self.assertEqual(e["Colors"][0], 0x80000000)
        for name, index in (
            ("Colors", 0),
            ("Positions", c["count"] * 3),
            ("Count", 1),
            ("Descriptors", 0),
        ):
            a = copy.deepcopy(e)
            a[name][index] ^= 1
            with self.subTest(name=name), self.assertRaises(AssertionError):
                check_outputs(l, c, b, a, e, k)

    def test_both_wrong_deformations_fail_independent_geometry(self):
        l, c, b, e, k = self.fixture()
        e["Deformations"].view(np.float32)[4] += 0.2
        with self.assertRaisesRegex(AssertionError, "Scaled orthogonal"):
            check_outputs(l, c, b, e, e, k)


if __name__ == "__main__":
    unittest.main()
