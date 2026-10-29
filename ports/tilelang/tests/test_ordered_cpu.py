"""Repeated native stages reuse compilation while preserving all dispatches."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import replay
from ordered import execute_c_program


class OrderedCpuTests(unittest.TestCase):
    def test_repeated_kernels_keep_buffer_mappings_and_iteration_counts(self):
        bindings = {
            "schema": 1,
            "entry": "step",
            "buffers": ["A"],
            "storage_buffers": ["A"],
            "buffer_bytes": {"A": 4},
            "blocks": 1,
            "threads": 1,
            "iterations": 1,
        }
        stages = []
        for name, source, physical, iterations in [
            ("x", "one.c", "X", 1),
            ("y", "one.c", "Y", 2),
            ("add-two", "two.c", "X", 1),
        ]:
            b = copy.deepcopy(bindings)
            b["iterations"] = iterations
            stages.append(
                {
                    "name": name,
                    "source": source,
                    "bindings": b,
                    "buffer_map": {"A": physical},
                }
            )
        program = {
            "schema": 2,
            "storage_buffers": ["X", "Y"],
            "buffer_bytes": {"X": 4, "Y": 4},
            "iterations": 2,
            "stages": stages,
        }
        sources = {
            n: '#include <cstdint>\nextern "C" int32_t step(uint32_t* a) {a[0]+='
            + str(i)
            + ";return 0;}"
            for n, i in [("one.c", 1), ("two.c", 2)]
        }
        arrays = {
            "X": np.array([0], dtype=np.uint32),
            "Y": np.array([10], dtype=np.uint32),
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(replay, "compile_c", wraps=replay.compile_c) as compile_kernel,
        ):
            execute_c_program(sources, program, Path(directory), arrays)
            self.assertEqual(compile_kernel.call_count, 2)
        self.assertEqual(arrays["X"][0], 6)
        self.assertEqual(arrays["Y"][0], 14)
