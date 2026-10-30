"""LBVH topology, geometry, capacity and synchronization regressions."""

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lbvh_validation import BUFFERS, OPERATIONS, check_outputs, validate_case
from replay import HERE


class LbvhTests(unittest.TestCase):
    def fixture(self, op="lbvh_refit_frontier", name="n257-b2-g3-mixed"):
        root = Path(
            os.environ.get(
                "NEXUS_TILELANG_LBVH_FIXTURES",
                Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
                / "lbvh_domain/c/fixtures",
            )
        )
        d = root / op / name
        l = json.loads((root / "layout.json").read_text())
        c = json.loads((d / "case.json").read_text())
        k = json.loads((HERE / "contracts" / f"{op}.json").read_text())
        b = {n: np.fromfile(d / f"{n}.bin", np.uint32) for n in BUFFERS}
        e = {
            n: np.fromfile(d / f"{n}.expected.bin", np.uint32)
            if n in k["outputs"]
            else v.copy()
            for n, v in b.items()
        }
        return l, c, b, e, k

    def test_all_native_stages(self):
        for op in OPERATIONS:
            k = json.loads((HERE / "contracts" / f"{op}.json").read_text())
            for name in k["required_cases"]:
                with self.subTest(operation=op, case=name):
                    l, c, b, e, k = self.fixture(op, name)
                    check_outputs(l, c, b, e, e, k)

    def test_malformed_dispatch_and_buffers(self):
        for mutation in ("uniform", "capacity", "dtype", "pose", "sort", "cycle"):
            l, c, b, _e, _k = self.fixture()
            if mutation == "uniform":
                b["BatchIds"][l["batch"]["num_batches"]] += 1
            if mutation == "capacity":
                c["capacity"] += 1
            if mutation == "dtype":
                b["Tree"] = b["Tree"].astype(np.int32)
            if mutation == "pose":
                b["Poses"].view(np.float32)[l["pose_rotation"] + 3] = 2
            if mutation == "sort":
                b["Sorted"][1] = b["Sorted"][0]
            if mutation == "cycle":
                b["Tree"][l["lbvh_node"]["left"]] = 0
            with (
                self.subTest(mutation=mutation),
                self.assertRaises((AssertionError, ValueError)),
            ):
                validate_case(l, c, b)

    def test_matching_corrupt_topology_and_escape_rejected(self):
        for op, field, value in [
            ("lbvh_build", "left", 0),
            ("lbvh_build", "last_leaf", 0),
            ("lbvh_escapes", "escape", 0),
        ]:
            l, c, b, e, k = self.fixture(op)
            e["Tree"][l["lbvh_node"][field]] = value
            with (
                self.subTest(operation=op, field=field),
                self.assertRaises(AssertionError),
            ):
                check_outputs(l, c, b, e, e, k)

    def test_matching_wrong_morton_endpoint_rejected(self):
        l, c, b, e, k = self.fixture("lbvh_morton", "n129-b1-g1-duplicates")
        e["Keys"][35] ^= 1
        with self.assertRaisesRegex(AssertionError, "Morton quantization"):
            check_outputs(l, c, b, e, e, k)

    def test_matching_bad_refit_geometry_rejected(self):
        l, c, b, e, k = self.fixture()
        e["Tree"].view(np.float32)[l["lbvh_node"]["aabb"] + l["aabb"]["maxs"]] += 1
        with self.assertRaises(AssertionError):
            check_outputs(l, c, b, e, e, k)
        l, c, b, e, k = self.fixture("lbvh_leaves")
        off = (
            (c["colliders"] - 1) * l["lbvh_node_words"]
            + l["lbvh_node"]["aabb"]
            + l["aabb"]["mins"]
        )
        e["Tree"].view(np.float32)[off] += 10
        with self.assertRaises(AssertionError):
            check_outputs(l, c, b, e, e, k)

    def test_frontier_is_a_unique_complete_tree_cut(self):
        l, c, b, e, k = self.fixture("lbvh_refit_chunks")
        self.assertGreater(e["FrontierLen"][0], 1)
        e["Frontier"][1] = e["Frontier"][0]
        with self.assertRaisesRegex(AssertionError, "cut membership"):
            check_outputs(l, c, b, e, e, k)

    def test_overflow_keeps_exact_requested_count_and_legal_subset(self):
        l, c, b, e, k = self.fixture("lbvh_pairs", "n33-b2-g1-overflow")
        self.assertGreater(e["PairCount"][0], c["pair_capacity"])
        e["PairCount"][0] = c["pair_capacity"]
        with self.assertRaises(AssertionError):
            check_outputs(l, c, b, e, e, k)
        l, c, b, e, k = self.fixture("lbvh_pairs", "n33-b2-g1-overflow")
        e["Pairs"][:2] = e["Pairs"][2:4]
        with self.assertRaises(AssertionError):
            check_outputs(l, c, b, e, e, k)

    def test_guard_storage_and_readonly_inputs_remain_exact(self):
        for name in BUFFERS:
            l, c, b, e, k = self.fixture()
            a = {n: v.copy() for n, v in e.items()}
            a[name][-1] ^= 1
            with self.subTest(buffer=name), self.assertRaises(AssertionError):
                check_outputs(l, c, b, a, e, k)

    def test_ascend_pair_scratch_is_private(self):
        try:
            import tilelang.ascend.language
        except ImportError:
            self.skipTest("Ascend TileLang is not installed")
        import tilelang

        if "ascend" not in tilelang.tvm.target.Target.list_kinds():
            self.skipTest("Compiler was built without the Ascend target")
        from lbvh_kernels import dimensions, make_kernel
        from ordered import lower_program

        l, c, b, _e, _k = self.fixture("lbvh_pairs", "n33-b3-g1-mixed")
        blocks, threads, _ = dimensions(c)
        sources, _ = lower_program(
            [
                {
                    "name": "pairs",
                    "kernel": make_kernel(l, c, "ascend"),
                    "buffer_map": {n: n for n in BUFFERS},
                    "blocks": blocks,
                    "threads": threads,
                }
            ],
            b,
            "ascend",
        )
        source = next(iter(sources.values()))
        self.assertIn("uint32_t WgPairs[16]", source)
        self.assertGreaterEqual(source.count("asc_syncthreads();"), 3)

    def test_metal_refit_has_device_memory_workgroup_fence(self):
        try:
            import tilelang  # noqa: F401
        except ImportError:
            self.skipTest("TileLang is not installed")
        from lbvh_kernels import dimensions, make_kernel
        from ordered import lower_program

        l, c, b, _e, _k = self.fixture("lbvh_refit_chunks", "n2-b1-g1-mixed")
        blocks, threads, _ = dimensions(c)
        sources, _ = lower_program(
            [
                {
                    "name": "refit",
                    "kernel": make_kernel(l, c, "metal"),
                    "buffer_map": {n: n for n in BUFFERS},
                    "blocks": blocks,
                    "threads": threads,
                }
            ],
            b,
            "metal",
        )
        source = next(iter(sources.values()))
        self.assertIn(
            "threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup)",
            source,
        )
        self.assertIn("nexus_lbvh_refit_barrier();", source)


if __name__ == "__main__":
    unittest.main()
