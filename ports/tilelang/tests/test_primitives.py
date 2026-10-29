"""Guard irregular integer reads, writes and strict comparisons using Rust fixtures."""

import json
import os
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from primitives import BUFFERS, HERE, check_outputs, validate_case


class PrimitiveTests(unittest.TestCase):
    def setUp(self):
        root = (
            Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
            / "stabilize/c/fixtures/stabilize"
        )
        self.case = json.loads((root / "suite.json").read_text())["cases"][-1]
        directory = root / self.case["name"]
        self.before = {
            n: np.fromfile(directory / f"{n}.bin", dtype=np.uint32) for n in BUFFERS
        }
        self.actual = {
            n: np.fromfile(directory / f"{n}.actual.bin", dtype=np.uint32)
            for n in BUFFERS
        }
        self.expected = {
            n: np.fromfile(directory / f"{n}.expected.bin", dtype=np.uint32)
            if n == "Sorted"
            else self.before[n]
            for n in BUFFERS
        }
        self.contract = json.loads((HERE / "contracts/stabilize.json").read_text())

    def check(self):
        check_outputs(
            {}, self.case, self.before, self.actual, self.expected, self.contract
        )

    def test_live_rust_result_and_high_integer_ids(self):
        self.assertTrue(np.any(self.before["Ids"] > 2**24))
        self.check()

    def test_invalid_row_ends_fail_before_dispatch(self):
        self.before["Counts"][-1] = self.case["storage"] + 1
        with self.assertRaisesRegex(ValueError, "row ends"):
            validate_case(self.case, self.before)
        self.before["Counts"][-1] = self.case["count"]
        self.before["Counts"][1] = self.case["count"]
        self.before["Counts"][2] = 0
        with self.assertRaisesRegex(ValueError, "row ends"):
            validate_case(self.case, self.before)

    def test_input_and_sentinel_writes_fail(self):
        self.actual["Ids"][0] ^= np.uint32(1)
        with self.assertRaises(AssertionError):
            self.check()
        self.actual["Ids"] = self.before["Ids"].copy()
        self.actual["Sorted"][-1] ^= np.uint32(1)
        with self.assertRaises(AssertionError):
            self.check()

    def test_one_bit_sort_error_is_rejected(self):
        self.actual["Sorted"][2] ^= np.uint32(1)
        with self.assertRaises(AssertionError):
            self.check()


if __name__ == "__main__":
    unittest.main()
