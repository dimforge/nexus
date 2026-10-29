"""Exact scan hierarchy, ordered ABI, guard words and invalid fixture checks."""

import copy
import json
import os
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay import HERE
import scan_replay as scan
from ordered import validate_program


class ScanTests(unittest.TestCase):
    def fixture(self, op="prefix_scan", name="n65537-b2-mixed"):
        root = (
            Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
            / op
            / "c/fixtures"
        )
        layout = json.loads((root / "layout.json").read_text())
        suite = json.loads((root / op / "suite.json").read_text())
        case = next(c for c in suite["cases"] if c["name"] == name)
        directory = root / op / name
        before = {
            n: np.fromfile(directory / f"{n}.bin", dtype=np.uint32)
            for n in scan.BUFFERS
        }
        contract = json.loads((HERE / f"contracts/{op}.json").read_text())
        expected = {
            n: np.fromfile(directory / f"{n}.expected.bin", dtype=np.uint32)
            if n in contract["outputs"]
            else a.copy()
            for n, a in before.items()
        }
        program = json.loads((directory / "bindings.json").read_text())
        return layout, case, before, expected, contract, program

    def test_all_cases_and_independent_scan_invariant(self):
        for op in scan.OPERATIONS:
            contract = json.loads((HERE / f"contracts/{op}.json").read_text())
            for name in contract["required_cases"]:
                with self.subTest(op=op, name=name):
                    l, c, b, e, k, p = self.fixture(op, name)
                    scan.check_outputs(l, c, b, e, e, k)
                    validate_program(p)

    def test_large_scan_requires_three_sweeps_and_two_adds(self):
        _, c, _, _, _, p = self.fixture()
        self.assertEqual(c["levels"], 3)
        self.assertEqual(
            [s["name"] for s in p["stages"]],
            [
                "prefix_sweep_0",
                "prefix_sweep_1",
                "prefix_sweep_2",
                "prefix_add_1",
                "prefix_add_0",
            ],
        )
        self.assertEqual(p["stages"][0]["bindings"]["blocks"], 514)
        self.assertEqual(p["stages"][1]["buffer_map"]["Data"], "Aux0")

    def test_corrupt_data_scratch_uniform_and_guard_fail(self):
        for name, index in (
            ("Data", 256),
            ("Data", 65537),
            ("Aux0", 256),
            ("Aux1", 0),
            ("Aux3", -1),
            ("Data", -1),
            ("Stride0", 1),
        ):
            l, c, b, e, k, _ = self.fixture()
            actual = {n: a.copy() for n, a in e.items()}
            actual[name][index] ^= np.uint32(1)
            with (
                self.subTest(name=name, index=index),
                self.assertRaises(AssertionError),
            ):
                scan.check_outputs(l, c, b, actual, e, k)

    def test_malformed_dimensions_uniform_and_overflow_rejected(self):
        for mutation in (
            "batches",
            "stride",
            "levels",
            "uniform",
            "storage",
            "overflow",
        ):
            l, c, b, _, _, _ = self.fixture()
            if mutation == "batches":
                c["batches"] = 0
            if mutation == "stride":
                c["stride"] = -1
            if mutation == "levels":
                c["levels"] = 1
            if mutation == "uniform":
                b["Stride0"][0] += 1
            if mutation == "storage":
                b["Aux0"] = b["Aux0"][:-1]
            if mutation == "overflow":
                b["Data"][:2] = 0xFFFFFFFF
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                scan.validate_case(l, c, b)

    def test_ordered_abi_rejects_aliases_missing_storage_and_unsafe_sources(self):
        *_, program = self.fixture()
        for mutation in (
            "alias",
            "missing",
            "size",
            "source",
            "duplicate",
            "empty",
            "iterations",
            "schema",
        ):
            p = copy.deepcopy(program)
            s = p["stages"][0]
            if mutation == "alias":
                s["buffer_map"]["Aux"] = s["buffer_map"]["Data"]
            if mutation == "missing":
                del s["buffer_map"]["Data"]
            if mutation == "size":
                p["buffer_bytes"]["Data"] += 4
            if mutation == "source":
                s["source"] = "../bad"
            if mutation == "duplicate":
                p["stages"].append(copy.deepcopy(s))
            if mutation == "empty":
                p["stages"] = []
            if mutation == "iterations":
                p["iterations"] = 0
            if mutation == "schema":
                p["schema"] = 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_program(p)
