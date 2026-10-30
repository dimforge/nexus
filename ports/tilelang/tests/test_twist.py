"""Guard and physics-validation regressions on the current native twist fixtures."""

import copy
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay import HERE
from twist_validation import BUFFERS, OPERATIONS, check_outputs, validate_case


class TwistTests(unittest.TestCase):
    def fixture(self, op="twist_solve"):
        root = (
            Path(
                os.environ.get(
                    "NEXUS_TILELANG_TWIST_RESULTS", HERE / "results-twist-0089"
                )
            )
            / op
            / "c/fixtures"
        )
        name = "n65-b1-f1-i3" if op == "twist_solve" else "n65-b1-f1-i1"
        directory = root / op / name
        l = json.loads((root / "layout.json").read_text())
        c = json.loads((directory / "case.json").read_text())
        b = {n: np.fromfile(directory / f"{n}.bin", np.uint32) for n in BUFFERS}
        e = {
            n: np.fromfile(directory / f"{n}.expected.bin", np.uint32)
            if (directory / f"{n}.expected.bin").exists()
            else a.copy()
            for n, a in b.items()
        }
        k = json.loads((HERE / f"contracts/{op}.json").read_text())
        return l, c, b, e, k

    def test_native_oracle_passes_each_operation(self):
        for op in OPERATIONS:
            l, c, b, e, k = self.fixture(op)
            check_outputs(l, c, b, e, e, k)

    def test_input_ownership_and_range_rejections(self):
        l, c, b, _, _ = self.fixture()
        h = l["header"]
        for field, value in (
            ("normal", 5),
            ("body_a", c["bodies"]),
            ("diag_a", c["solver_storage"]),
            ("off_a", int(b["Tiles"][h["diag_a"] + 3])),
        ):
            bad = copy.deepcopy(b)
            bad["Tiles"][h[field] + 3] = value
            with self.assertRaises(ValueError):
                validate_case(l, c, bad)

    def test_immutable_ids_padding_and_inactive_rows_are_exact(self):
        l, c, b, e, k = self.fixture()
        indices = [
            l["header"]["body_a"] + 3,
            l["twist"]["friction"] + 3,
            l["warmstart"],
            l["tile_words"] * 2 + 4,
        ]
        for index in indices:
            bad = copy.deepcopy(e)
            bad["Tiles"][index] ^= np.uint32(1)
            with self.assertRaises(AssertionError):
                check_outputs(l, c, b, bad, e, k)
        bad = copy.deepcopy(e)
        bad["Velocities"][l["velocity_padding0"]] ^= np.uint32(1)
        with self.assertRaises(AssertionError):
            check_outputs(l, c, b, bad, e, k)

    def test_readonly_inputs_are_exact(self):
        l, c, b, e, k = self.fixture()
        for n in ("Poses", "Params"):
            bad = copy.deepcopy(e)
            bad[n][0] ^= np.uint32(1)
            with self.assertRaises(AssertionError):
                check_outputs(l, c, b, bad, e, k)

    def test_independent_limits_reject_even_corrupt_matching_oracles(self):
        l, c, b, e, k = self.fixture()
        t, h = l["twist"], l["header"]
        for mode in ("normal", "tangent", "twist", "single"):
            bad = copy.deepcopy(e)
            f = bad["Tiles"].view(np.float32)
            if mode == "normal":
                f[4 + t["normal_impulse"]] = -1.0
            if mode == "tangent":
                f[8 + t["friction"]] = 1e5
            if mode == "twist":
                f[8 + h["off_b"] + 3] = 1e5
            if mode == "single":
                f[4 + h["off_b"] + 3] = 1e-8
            with self.assertRaises(AssertionError):
                check_outputs(l, c, b, bad, bad, k)

    def test_invalid_native_frames_and_abi_are_rejected(self):
        l, c, b, _, _ = self.fixture()
        bad = copy.deepcopy(b)
        bad["Tiles"].view(np.float32)[
            l["twist"]["frame_a"] : l["twist"]["frame_a"] + 4
        ] = 0
        with self.assertRaises(ValueError):
            validate_case(l, c, bad)
        l["twist"]["center_a"] = l["twist"]["offset_b"]
        with self.assertRaises(ValueError):
            validate_case(l, c, b)


if __name__ == "__main__":
    unittest.main()
