"""Native color ranges and parameter ABI safety."""

import json
import os
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay import HERE
import colored_replay as colored
import coefficient_replay as coefficients


class ColoredTests(unittest.TestCase):
    def fixture(self, op="colored_program", name="w33-b3-m1-s21"):
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

    def test_every_native_color_case(self):
        for op in colored.OPERATIONS:
            k = json.loads((HERE / f"contracts/{op}.json").read_text())
            for name in k["required_cases"]:
                with self.subTest(operation=op, case=name):
                    l, c, b, e, k = self.fixture(op, name)
                    colored.check_outputs(l, c, b, e, e, k)

    def test_overlapping_velocity_writes_are_rejected_per_color(self):
        l, c, b, e, k = self.fixture()
        h = l["header"]
        nb = c["batches"]
        first = int(b["Buckets"][nb - 1])
        second = first + 1
        a = first // 32 * l["tile_words"] + first % 32 * 4
        z = second // 32 * l["tile_words"] + second % 32 * 4
        b["Tiles"][z + h["diag_a"] + 3] = b["Tiles"][a + h["diag_a"] + 3]
        with self.assertRaisesRegex(ValueError, "overlap"):
            colored.validate_case(l, c, b)

    def test_input_uniform_guard_and_unsolved_tiles_are_preserved(self):
        for name, index in [
            ("Mode", 0),
            ("Color", 0),
            ("RbdParams", 0),
            ("Velocities", -1),
            ("Tiles", -1),
            ("Buckets", -1),
        ]:
            l, c, b, e, k = self.fixture()
            a = {n: v.copy() for n, v in e.items()}
            a[name][index] ^= np.uint32(1)
            with self.subTest(buffer=name), self.assertRaises(AssertionError):
                colored.check_outputs(l, c, b, a, e, k)
        l, c, b, e, k = self.fixture()
        start = int(b["Buckets"][4 * c["batches"] - 1])
        base = start // 32 * l["tile_words"] + start % 32 * 4
        a = {n: v.copy() for n, v in e.items()}
        a["Tiles"][base + l["points"] + l["point"]["arm_a"] + 3] ^= np.uint32(1)
        with self.assertRaises(AssertionError):
            colored.check_outputs(l, c, b, a, e, k)

    def test_cache_tags_are_independent_of_the_oracle(self):
        l, c, b, e, k = self.fixture()
        i = int(b["Buckets"][c["batches"] - 1])
        cache = (
            i // 32 * l["tile_words"]
            + l["warmstart"]
            + i % 32 * 2 * l["velocity_words"]
        )
        e["Tiles"][cache + l["velocity_padding1"]] = 0xFFFFFFFF
        with self.assertRaisesRegex(AssertionError, "cache body tag"):
            colored.check_outputs(l, c, b, e, e, k)

    def test_invalid_ranges_modes_and_parameters_fail(self):
        for mutation in ("range", "mode", "spring", "color", "body", "shape"):
            l, c, b, e, k = self.fixture()
            if mutation == "range":
                b["Buckets"][-1] += 1
            if mutation == "mode":
                b["Mode"][0] += 1
            if mutation == "spring":
                b["RbdParams"][l["rbd_params"]["contact_damping_ratio"]] = 0
            if mutation == "color":
                b["Color"][0] = 0
            if mutation == "body":
                b["Tiles"][l["header"]["body_a"] + 3] = 0xFFFFFFFF
            if mutation == "shape":
                b["Tiles"] = b["Tiles"][:-1]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                colored.validate_case(l, c, b)

    def test_coefficients_include_exact_rust_sentinel_and_adjacent_values(self):
        k = json.loads((HERE / "contracts/contact_coefficients.json").read_text())
        for name in k["required_cases"]:
            l, c, b, e, k = self.fixture("contact_coefficients", name)
            coefficients.check_outputs(l, c, b, e, e, k)
        l, c, b, e, k = self.fixture("contact_coefficients", "p4")
        self.assertEqual(e["Coefficients"][1], l["constants"]["max_flt_bits"])
        self.assertNotEqual(e["Coefficients"][1], 0x7F7FFFFF)
        bad = {n: v.copy() for n, v in e.items()}
        bad["Coefficients"][1] = 0x7F7FFFFF
        with self.assertRaises(AssertionError):
            coefficients.check_outputs(l, c, b, bad, bad, k)
        for name in ("p8", "p9"):
            l, c, b, e, k = self.fixture("contact_coefficients", name)
            self.assertGreater(e["Coefficients"].view(np.float32)[1], 1.5e18)

    def test_coefficient_guards_and_zero_timestep(self):
        l, c, b, e, k = self.fixture("contact_coefficients", "p1")
        self.assertEqual(list(e["Coefficients"][[0, 2, 4]]), [0, 0, 0])
        for name, index in (("Coefficients", 8), ("RbdParams", 0)):
            bad = {n: v.copy() for n, v in e.items()}
            bad[name][index] ^= np.uint32(1)
            with self.assertRaises(AssertionError):
                coefficients.check_outputs(l, c, b, bad, e, k)
