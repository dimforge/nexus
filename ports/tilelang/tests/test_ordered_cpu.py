"""Repeated native stages reuse compilation while preserving all dispatches."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import replay
from ordered import CompilationSession, execute_c_program


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
            session = CompilationSession()
            execute_c_program(
                sources, program, Path(directory), arrays, session=session
            )
            self.assertEqual(compile_kernel.call_count, 2)
            other = {"X": np.array([100], np.uint32), "Y": np.array([200], np.uint32)}
            execute_c_program(sources, program, Path(directory), other, session=session)
            self.assertEqual(compile_kernel.call_count, 2)
            self.assertEqual((int(other["X"][0]), int(other["Y"][0])), (106, 204))
            sources["two.c"] = sources["two.c"].replace("a[0]+=2", "a[0]+=3")
            execute_c_program(sources, program, Path(directory), other, session=session)
            self.assertEqual(compile_kernel.call_count, 3)
            self.assertEqual((int(other["X"][0]), int(other["Y"][0])), (114, 208))
        self.assertEqual(arrays["X"][0], 6)
        self.assertEqual(arrays["Y"][0], 14)

    def test_full_ir_and_target_key_lowering_reuse(self):
        import tilelang
        from primitive_kernels import make_kernel

        first = make_kernel({"rows": 2, "storage": 70}, "c")
        same = make_kernel({"rows": 2, "storage": 70}, "c")
        different = make_kernel({"rows": 3, "storage": 70}, "c")
        session = CompilationSession(capacity=2)
        with patch.object(tilelang, "lower", wraps=tilelang.lower) as lower:
            a = session.lower(first, "c")
            self.assertIs(session.lower(same, "c"), a)
            changed = session.lower(different, "c")
            self.assertNotEqual(a.kernel_source, changed.kernel_source)
            self.assertEqual(lower.call_count, 2)
            # The target is part of the key even for identical IR.
            session.lower(first, "metal")
            self.assertEqual(lower.call_count, 3)
            session.lower(first, "c")
            self.assertEqual(lower.call_count, 4)  # bounded cache evicted oldest IR
