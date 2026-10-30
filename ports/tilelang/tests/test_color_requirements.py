"""Reject lost contacts, wraparound, and matching corrupt control-oracle outputs."""

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import color_requirement_replay as replay
from replay import HERE


class ColorRequirementTests(unittest.TestCase):
    def fixture(self, name="n65-pending"):
        root = (
            Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
            / "color_requirements/c/fixtures"
        )
        l = json.loads((root / "layout.json").read_text())
        k = json.loads((HERE / "contracts/color_requirements.json").read_text())
        c = next(
            c
            for c in json.loads((root / "color_requirements/suite.json").read_text())[
                "cases"
            ]
            if c["name"] == name
        )
        d = root / "color_requirements" / name
        b = {n: np.fromfile(d / f"{n}.bin", np.uint32) for n in replay.BUFFERS}
        e = {
            n: (
                np.fromfile(d / f"{n}.expected.bin", np.uint32)
                if n in k["outputs"]
                else a.copy()
            )
            for n, a in b.items()
        }
        return l, c, b, e, k

    def test_every_native_case(self):
        k = json.loads((HERE / "contracts/color_requirements.json").read_text())
        for name in k["required_cases"]:
            with self.subTest(case=name):
                l, c, b, e, k = self.fixture(name)
                replay.check_outputs(l, c, b, e, e, k)

    def test_matching_wrong_feedback_fails(self):
        for case, word in [
            ("n65-pending", 0),
            ("n65-pending", 1),
            ("n1-invalid", 0),
            ("n1-max-rounds", 1),
            ("n65-inactive", 0),
        ]:
            l, c, b, e, k = self.fixture(case)
            bad = {n: a.copy() for n, a in e.items()}
            bad["Requirements"][word] ^= np.uint32(1)
            with self.subTest(case=case, word=word), self.assertRaises(AssertionError):
                replay.check_outputs(l, c, b, bad, bad, k)

    def test_inputs_and_tail_are_exact(self):
        l, c, b, e, k = self.fixture()
        for name in replay.BUFFERS:
            bad = {n: a.copy() for n, a in e.items()}
            bad[name][-1] ^= np.uint32(1)
            with self.subTest(buffer=name), self.assertRaises(AssertionError):
                replay.check_outputs(l, c, b, bad, bad, k)

    def test_plan_cannot_escape_storage(self):
        l, c, b, _e, _k = self.fixture()
        b["Plan"][l["plan"]["bound"]] = c["count"] + 1
        with self.assertRaises(ValueError):
            replay.validate_case(l, c, b)


if __name__ == "__main__":
    unittest.main()
