"""Resident LBVH has no host sorting or intermediate physics substitution."""

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lbvh_program_validation import BUFFERS, check_outputs, validate_case
from replay import HERE


class LbvhProgramTests(unittest.TestCase):
    def fixture(self, name="n257-b2-g3-mixed"):
        root = Path(
            os.environ.get(
                "NEXUS_TILELANG_LBVH_PROGRAM_FIXTURES",
                Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
                / "lbvh_program/c/fixtures",
            )
        )
        d = root / "lbvh_program" / name
        l = json.loads((root / "layout.json").read_text())
        c = json.loads((d / "case.json").read_text())
        k = json.loads((HERE / "contracts/lbvh_program.json").read_text())
        b = {n: np.fromfile(d / f"{n}.bin", np.uint32) for n in BUFFERS}
        e = {
            n: np.fromfile(d / f"{n}.expected.bin", np.uint32)
            if n in k["outputs"]
            else v.copy()
            for n, v in b.items()
        }
        return l, c, b, e, k

    def test_all_native_programs(self):
        k = json.loads((HERE / "contracts/lbvh_program.json").read_text())
        for name in k["required_cases"]:
            with self.subTest(case=name):
                l, c, b, e, k = self.fixture(name)
                check_outputs(l, c, b, e, e, k)

    def test_bad_input_and_sort_uniforms(self):
        for name in ("Identity", "Config3", "NumKeys", "FrontierLen"):
            l, c, b, _e, _k = self.fixture()
            b[name][0] ^= 1
            with (
                self.subTest(buffer=name),
                self.assertRaises((AssertionError, ValueError)),
            ):
                validate_case(l, c, b)

    def test_matching_corrupt_sort_scratch_is_rejected(self):
        for name in (
            "Keys",
            "SortKeys",
            "Sorted",
            "PongValues",
            "RadixCounts",
            "Reduced",
            "SortGrid",
        ):
            l, c, b, e, k = self.fixture()
            e[name][0] ^= 1
            with (
                self.subTest(buffer=name),
                self.assertRaises((AssertionError, ValueError)),
            ):
                check_outputs(l, c, b, e, e, k)

    def test_matching_corrupt_tree_is_rejected(self):
        for field, value in [
            ("left", 0),
            ("escape", 0),
            ("refit_count", 1),
            ("first_leaf", 9),
        ]:
            l, c, b, e, k = self.fixture()
            e["Tree"][l["lbvh_node"][field]] = value
            with (
                self.subTest(field=field),
                self.assertRaises((AssertionError, ValueError)),
            ):
                check_outputs(l, c, b, e, e, k)
        l, c, b, e, k = self.fixture()
        e["Tree"].view(np.float32)[l["lbvh_node"]["aabb"] + l["aabb"]["maxs"]] += 1
        with self.assertRaises(AssertionError):
            check_outputs(l, c, b, e, e, k)

    def test_overflow_keeps_requested_count_and_written_subset(self):
        l, c, b, e, k = self.fixture("n33-b2-g1-overflow")
        self.assertGreater(e["PairCount"][0], c["pair_capacity"])
        e["PairCount"][0] = c["pair_capacity"]
        with self.assertRaises(AssertionError):
            check_outputs(l, c, b, e, e, k)

    def test_frontier_order_is_semantic(self):
        l, c, b, e, k = self.fixture()
        a = {n: v.copy() for n, v in e.items()}
        for batch in range(c["batches"]):
            start = batch * c["capacity"]
            count = int(a["FrontierLen"][batch])
            a["Frontier"][start : start + count] = a["Frontier"][start : start + count][
                ::-1
            ]
        check_outputs(l, c, b, a, e, k)

    def test_exact_guards_and_readonly_inputs(self):
        for name in BUFFERS:
            l, c, b, e, k = self.fixture("n2-b1-g1-mixed")
            a = {n: v.copy() for n, v in e.items()}
            a[name][-1] ^= 1
            with (
                self.subTest(buffer=name),
                self.assertRaises((AssertionError, ValueError)),
            ):
                check_outputs(l, c, b, a, e, k)


if __name__ == "__main__":
    unittest.main()
