"""Focused optimized schedules, with fresh noalias-disabled Rust outputs."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpu_oracle import export_fixtures
from ordered import CompilationSession, execute_c_program, lower_program
from replay import HERE, build_native, run
import canonical_kernels
import canonical_replay as canonical


class ExecutionOptimizations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="nexus-execution-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.target = os.environ.get("NEXUS_TILELANG_TEST_TARGET", "c")
        cls.session = CompilationSession()
        cls.runner = build_native() if cls.target == "metal" else None
        cls.exports = {}
        cls.sequence = 0

    def fixture(self, oracle, operation, name):
        if oracle not in self.exports:
            directory = self.root / oracle
            export_fixtures(oracle, directory)
            self.exports[oracle] = directory
        root = self.exports[oracle]
        layout = json.loads((root / "layout.json").read_text())
        suite = json.loads((root / operation / "suite.json").read_text())
        case = next(c for c in suite["cases"] if c["name"] == name)
        contract = json.loads((HERE / f"contracts/{operation}.json").read_text())
        directory = root / operation / name
        before = {n: np.fromfile(directory / f"{n}.bin", np.uint32)
                  for n in contract["inputs"]}
        expected = {n: np.fromfile(directory / f"{n}.expected.bin", np.uint32)
                    if n in contract["outputs"] else a.copy()
                    for n, a in before.items()}
        return layout, case, before, expected, contract

    def execute(self, plans, before, iterations=1):
        self.sequence += 1
        directory = self.root / f"execution-{self.sequence}"
        directory.mkdir()
        sources, program = lower_program(plans, before, self.target,
                                         iterations=iterations, session=self.session)
        actual = {n: a.copy() for n, a in before.items()}
        if self.target == "c":
            execute_c_program(sources, program, directory, actual, session=self.session)
        else:
            for n, a in before.items():
                a.tofile(directory / f"{n}.bin")
            for n, source in sources.items():
                (directory / n).write_text(source)
            (directory / "bindings.json").write_text(json.dumps(program))
            run([self.runner, directory])
            actual = {n: np.fromfile(directory / f"{n}.actual.bin", np.uint32)
                      for n in before}
        return actual, sources

    def test_bounded_sort_matches_rust_order_and_exact_workspace(self):
        for name in ("n65-duplicates-s7", "n65-two-fields-s7"):
            with self.subTest(case=name):
                l, c, b, e, k = self.fixture("canonical", "canonical_contacts", name)
                canonical.check_outputs(l, c, b, e, e, k)
                rows = b["Contacts"].reshape(-1, l["contact_words"])[:c["count"]]
                live = rows[:, l["contact"]["contact"] + l["manifold"]["len"]] != 0
                offset = l["contact"]["colliders"]
                bound = int(rows[live, offset:offset + 2].max()) + 1
                optimized = dict(c, collider_count=bound)
                plans = canonical_kernels.make_program(l, optimized, self.target)
                self.assertLess(len(plans), len(canonical_kernels.make_program(l, c, self.target)))
                actual, _ = self.execute(plans, b)
                spec = canonical.expected_outputs(l, optimized, b)
                canonical.check_outputs(l, optimized, b, actual, spec, k)
                for field in ("Output", "InOrder"):
                    np.testing.assert_array_equal(actual[field], e[field], err_msg=field)
                with self.assertRaisesRegex(ValueError, "declared bound"):
                    canonical.validate_case(l, dict(c, collider_count=bound - 1), b)

    def test_sort_digit_boundaries_reserve_inert_sentinel(self):
        for count, digits in ((0, 2), (255, 2), (256, 4), (65535, 4),
                              (65536, 6), (2**24, 8), (2**32 - 1, 8)):
            for selector in (1, 2, 5, 6):
                self.assertEqual(canonical_kernels.radix_digits(selector, count), digits)
                self.assertGreaterEqual(2**(digits * 4) - 1, count)
            for selector in (0, 3, 4, 7):
                self.assertEqual(canonical_kernels.radix_digits(selector, count), 8)
        self.assertEqual(canonical_kernels.radix_digits(1), 8)
        for invalid in (-1, 2**32, True, 3.5):
            with self.assertRaises(ValueError):
                canonical_kernels.radix_digits(1, invalid)
