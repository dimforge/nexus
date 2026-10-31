"""Virtual-grid semantics on real Metal, with source-only CUDA/Ascend ABI checks."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tilelang.language as T
from cpu_oracle import export_fixtures
from device_compiler import ascend_wrapper
from ordered import lower_program
from replay import HERE, build_native, run
from virtual_dispatch import virtualize


def reverse_kernel(target):
    @T.macro
    def body(X, group, lane):
        shared = T.alloc_shared((64,), "uint32")
        shared[lane] = X[group * 64 + lane]
        T.sync_threads()
        X[group * 64 + lane] = shared[63 - lane] + T.uint32(1)

    if target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(X: T.Tensor((1024,), "uint32")):
            with A.Kernel(16) as group, A.SimtVF(threads=64):
                body(X, group, A.get_thread_binding())
    else:

        @T.prim_func
        def kernel(X: T.Tensor((1024,), "uint32")):
            with T.Kernel(16, threads=64) as group:
                body(X, group, T.get_thread_binding())

    return kernel


class VirtualDispatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="nexus-virtual-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name)

    def plan(self, target):
        arrays = {
            "X": np.arange(1024, dtype=np.uint32),
            "Grid": np.array([16, 1, 1], np.uint32),
        }
        plan = {
            "name": "reverse",
            "kernel": reverse_kernel(target),
            "blocks": 16,
            "threads": 64,
            "buffer_map": {"X": "X"},
            "indirect": {"buffer": "Grid", "word": 0},
        }
        return arrays, virtualize(plan, arrays, physical_limit=3)

    def execute(self, sources, program, arrays, directory):
        directory.mkdir()
        for name, source in sources.items():
            (directory / name).write_text(source)
        for name, words in arrays.items():
            words.tofile(directory / f"{name}.bin")
        (directory / "bindings.json").write_text(json.dumps(program))
        run([build_native(), directory])
        return {
            n: np.fromfile(directory / f"{n}.actual.bin", np.uint32) for n in arrays
        }

    def test_shared_reuse_zero_tail_and_capacity_clamp_on_metal(self):
        arrays, plan = self.plan("metal")
        sources, program = lower_program([plan], arrays, "metal")
        self.assertEqual(program["stages"][0]["bindings"]["blocks"], 3)
        for count in (0, 1, 4, 15, 16, 100):
            with self.subTest(count=count):
                arrays["Grid"][0] = count
                actual = self.execute(
                    sources, program, arrays, self.root / f"count-{count}"
                )
                expected = arrays["X"].reshape(16, 64).copy()
                live = min(count, 16)
                expected[:live] = expected[:live, ::-1] + 1
                np.testing.assert_array_equal(actual["X"], expected.reshape(-1))
                np.testing.assert_array_equal(actual["Grid"], arrays["Grid"])

    def test_cuda_and_ascend_source_preserve_shared_memory_and_virtual_binding(self):
        for target in ("cuda", "ascend"):
            with self.subTest(target=target):
                arrays, plan = self.plan(target)
                sources, program = lower_program([plan], arrays, target)
                stage = program["stages"][0]
                self.assertNotIn("indirect", stage)
                self.assertEqual(stage["virtual_grid"]["capacity_blocks"], 16)
                self.assertIn("NexusDispatch", stage["bindings"]["buffers"])
                self.assertEqual(stage["bindings"]["blocks"], 3)
                if target == "cuda":
                    self.assertEqual(stage["bindings"]["shared_memory_bytes"], 256)
                else:
                    self.assertIn(
                        "nexus_launch",
                        ascend_wrapper(next(iter(sources.values())), stage["bindings"]),
                    )

    def test_virtual_lbvh_matches_fresh_rust_with_partial_groups(self):
        import lbvh_program_kernels as kernels
        import lbvh_program_validation as validation

        directory = self.root / "rust"
        export_fixtures("lbvh", directory)
        operation, name = "lbvh_program", "n129-b1-g1-duplicates"
        layout = json.loads((directory / "layout.json").read_text())
        case = next(
            c
            for c in json.loads((directory / operation / "suite.json").read_text())[
                "cases"
            ]
            if c["name"] == name
        )
        contract = json.loads((HERE / f"contracts/{operation}.json").read_text())
        fixtures = directory / operation / name
        before = {
            n: np.fromfile(fixtures / f"{n}.bin", np.uint32) for n in contract["inputs"]
        }
        expected = {
            n: np.fromfile(fixtures / f"{n}.expected.bin", np.uint32)
            if n in contract["outputs"]
            else a.copy()
            for n, a in before.items()
        }
        plans = kernels.make_program(layout, dict(case, groups=None), "metal")
        before["Grid"] = np.array([3, 1, 1], np.uint32)
        virtual = []
        for plan in plans:
            if plan["name"] == "lbvh_pairs":
                plan = virtualize(
                    dict(plan, indirect={"buffer": "Grid", "word": 0}),
                    before,
                    physical_limit=2,
                )
            virtual.append(plan)
        sources, program = lower_program(virtual, before, "metal")
        actual = self.execute(sources, program, before, self.root / "lbvh")
        before.pop("Grid")
        actual.pop("Grid")
        validation.check_outputs(layout, case, before, actual, expected, contract)
