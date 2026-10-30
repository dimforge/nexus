"""Prevent scratch underallocation at hierarchy boundaries in native scenes."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scan_layout import validate_scan


class RuntimeScanLayoutTests(unittest.TestCase):
    def test_boundaries_and_corruption(self):
        for strides, levels in [
            ([0, 0, 0, 0, 0], 1),
            ([256, 1, 1, 1, 1], 1),
            ([264, 2, 1, 1, 1], 2),
            ([65537, 257, 2, 1, 1], 3),
            ([16777217, 65537, 257, 2, 1], 4),
        ]:
            c = {"strides": strides[:4], "levels": levels}
            b = {}
            for i in range(4):
                b[f"AdjAux{i}"] = np.zeros(strides[i + 1] + 3, np.uint32)
                b[f"AdjStride{i}"] = np.array([strides[i], 123], np.uint32)
            validate_scan(c, b, "Adj", strides[0], "strides", "levels")
            for mutation in ("levels", "strides", "scratch", "uniform"):
                cc = {**c, "strides": c["strides"].copy()}
                bb = {n: v.copy() for n, v in b.items()}
                if mutation == "levels":
                    cc["levels"] += 1
                elif mutation == "strides":
                    cc["strides"][0] += 1
                elif mutation == "scratch":
                    bb["AdjAux1"] = bb["AdjAux1"][:-1]
                else:
                    bb["AdjStride0"][0] += 1
                with (
                    self.subTest(count=strides[0], mutation=mutation),
                    self.assertRaises(ValueError),
                ):
                    validate_scan(cc, bb, "Adj", strides[0], "strides", "levels")


if __name__ == "__main__":
    unittest.main()
