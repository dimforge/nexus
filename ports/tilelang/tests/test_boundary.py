"""CFM branch policy rejects invented ambiguity and uncoupled outcomes."""

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from boundary_validation import (
    BUFFERS,
    classify,
    indices,
    validate_case,
    variants,
    vector,
)
from replay import HERE

root = Path(
    os.environ.get(
        "NEXUS_TILELANG_BOUNDARY_FIXTURES",
        Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
        / "solve_boundary/c/fixtures",
    )
)


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global l, k
        l = json.loads((root / "layout.json").read_text())
        k = json.loads(
            (
                Path(__file__).resolve().parents[1] / "contracts/solve_boundary.json"
            ).read_text()
        )

    def fixture(self):
        return self.load("n129-b1-f1-i3-s42")

    def load(self, name):
        d = root / "solve_boundary" / name
        c = json.loads((d / "case.json").read_text())
        b = {n: np.fromfile(d / f"{n}.bin", np.uint32) for n in BUFFERS}
        e = {
            n: np.fromfile(d / f"{n}.expected.bin", np.uint32)
            if n in ("Tiles", "Velocities")
            else v.copy()
            for n, v in b.items()
        }
        return c, b, {n: v.copy() for n, v in e.items()}, e

    def test_all_native_branch_cases(self):
        for c in json.loads((root / "solve_boundary/suite.json").read_text())["cases"]:
            with self.subTest(case=c["name"]):
                c, b, a, e = self.load(c["name"])
                classify(l, c, b, a, e, k)

    def test_false_branch_ambiguity_is_rejected(self):
        c, b, _, _ = self.fixture()
        b["Uncertain"][0] ^= 1
        with self.assertRaisesRegex(AssertionError, "Independent boundary mask"):
            validate_case(l, c, b)

    def test_fabricated_nonambiguous_outcome_is_rejected(self):
        c, b, a, e = self.fixture()
        for i in range(c["count"]):
            ids = indices(l, c, b, i)
            if b["Uncertain"][i] == 0 and ids["Tiles"]:
                index = ids["Tiles"][0]
                b["SoftTiles"].view(np.float32)[index] += 0.1
                a["SoftTiles"] = b["SoftTiles"].copy()
                e["SoftTiles"] = b["SoftTiles"].copy()
                break
        with self.assertRaisesRegex(AssertionError, "Nonambiguous native branch"):
            classify(l, c, b, a, e, k)

    def test_mixed_manifold_outcomes_do_not_form_a_third_branch(self):
        c, b, a, e = self.fixture()
        options = variants(b)
        for i in range(c["count"]):
            ids = indices(l, c, b, i)
            if not np.allclose(
                vector(options["Soft"], ids),
                vector(options["Hard"], ids),
                atol=k["atol"],
                rtol=k["rtol"],
            ):
                for name, ix in ids.items():
                    a[name][ix] = options["Soft" if name == "Tiles" else "Hard"][name][
                        ix
                    ]
                break
        else:
            self.fail("Fixture requires distinct branch outcomes")
        with self.assertRaisesRegex(AssertionError, "complete allowed CFM branch"):
            classify(l, c, b, a, e, k)

    def test_outside_interval_never_opens_a_branch_choice(self):
        c, b, _, _ = self.fixture()
        uncertain, width = validate_case(l, c, b)
        self.assertLess(width, 4e-7)
        self.assertEqual(np.count_nonzero(uncertain), 65)
        # Move every known cancellation point far enough to make its sign stable.
        p = l["point"]
        for i, m in enumerate(uncertain):
            if m:
                j = int(m).bit_length() - 1
                offset = (
                    (i // 32) * l["tile_words"]
                    + l["points"]
                    + j * l["point_words"]
                    + (i % 32) * 4
                    + p["local_a"]
                    + 3
                )
                b["Tiles"].view(np.float32)[offset] += 0.01
        with self.assertRaisesRegex(AssertionError, "Independent boundary mask"):
            validate_case(l, c, b)

    def test_readonly_guards_and_impulse_bounds_remain_exact(self):
        for name in [
            "Tiles",
            "Velocities",
            "Poses",
            "Params",
            "SoftTiles",
            "HardVelocities",
            "Uncertain",
        ]:
            c, b, a, e = self.fixture()
            a[name][-1] ^= 1
            with self.subTest(name=name), self.assertRaises(AssertionError):
                classify(l, c, b, a, e, k)


if __name__ == "__main__":
    unittest.main()
