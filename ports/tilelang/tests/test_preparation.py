"""Prepared code must preserve fresh state, exact IR and full qualification gates."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import program_cache
from kernel_cache import KernelCache
from ordered import CompilationSession


class PreparationTests(unittest.TestCase):
    def test_import_graph_tracks_helpers_packages_new_imports_not_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pkg").mkdir()
            (root / "main.py").write_text("from pkg import kernel\n")
            (root / "pkg/__init__.py").write_text("version = 1\n")
            (root / "pkg/kernel.py").write_text(
                "def build():\n from .helper import value\n return value\n"
            )
            (root / "pkg/helper.py").write_text("value = 1\n")

            def signature():
                return program_cache.builder_sources(root, ("main",))

            first = signature()
            self.assertEqual(
                set(first),
                {"main.py", "pkg/__init__.py", "pkg/kernel.py", "pkg/helper.py"},
            )
            (root / "test_unrelated.py").write_text("assert True\n")
            (root / "README.md").write_text("changed docs")
            self.assertEqual(signature(), first)
            (root / "pkg/helper.py").write_text("value = 2\n")
            self.assertNotEqual(signature(), first)
            (root / "pkg/kernel.py").write_text("import new_helper\n")
            (root / "new_helper.py").write_text("value = 3\n")
            self.assertIn("new_helper.py", signature())
            (root / "new_helper.py").unlink()
            self.assertNotIn("new_helper.py", signature())

    def test_program_cache_does_not_store_state_and_invalidates_code_or_case(self):
        from test_frame_cpu import variant

        v = variant(8)
        frame = v["manifest"]
        original = copy.deepcopy(v["initial"])
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                program_cache, "builder_sources", return_value={"kernel": "one"}
            ) as sources,
            patch("scene_kernels.make_program", return_value=[]) as make,
            patch.object(
                program_cache,
                "lower_program",
                side_effect=lambda *a, **kw: (
                    copy.deepcopy(v["sources"]),
                    copy.deepcopy(frame["program"]),
                ),
            ),
            patch.object(
                program_cache,
                "from_program",
                side_effect=lambda l, c, p, program: dict(
                    copy.deepcopy(frame), program=program
                ),
            ),
        ):
            cache = KernelCache(directory, compiler_identity={})
            first = program_cache.prepare_program({}, {}, original, "c", cache)
            original["State"][0] = 99
            second = program_cache.prepare_program({}, {}, original, "c", cache)
            self.assertEqual(first, second)
            self.assertEqual(int(original["State"][0]), 99)
            self.assertEqual(make.call_count, 1)
            self.assertEqual(cache.stats["program_hits"], 1)
            payload = next((Path(directory) / "program").glob("*/artifact"))
            self.assertEqual(
                set(json.loads(payload.read_bytes())), {"frame", "sources"}
            )
            payload.write_bytes(b"corrupted")
            program_cache.prepare_program({}, {}, original, "c", cache)
            self.assertEqual(cache.stats["invalid_entries"], 1)
            sources.return_value = {"kernel": "changed"}
            program_cache.prepare_program({}, {}, original, "c", cache)
            program_cache.prepare_program({}, {"rows": 2}, original, "c", cache)
            self.assertEqual(make.call_count, 4)

    def test_source_compaction_preserves_all_dispatches_and_bindings(self):
        from frame_cpu import CpuFrameSession
        from test_frame_cpu import variant

        v = variant(8)
        stage = copy.deepcopy(v["manifest"]["program"]["stages"][-1])
        stage.update(name="again", source="again.c")
        v["manifest"]["program"]["stages"].append(stage)
        v["sources"]["again.c"] = v["sources"]["commit.c"]
        v["sources"] = program_cache.compact_sources(
            v["sources"], v["manifest"]["program"]
        )
        self.assertEqual(len(v["sources"]), 2)
        self.assertEqual(len(v["manifest"]["program"]["stages"]), 3)
        with tempfile.TemporaryDirectory() as directory:
            session = CpuFrameSession([v], Path(directory))
            session.step()
            self.assertEqual(int(session.storage["State"][0]), 44)
            session.reset()
            session.step()
            self.assertEqual(int(session.storage["State"][0]), 44)

    def test_parallel_lowering_matches_serial_and_warm_hits_start_no_workers(self):
        from primitive_kernels import make_kernel

        kernels = [make_kernel({"rows": n, "storage": 70}, "metal") for n in (2, 3)]
        expected = [
            CompilationSession().lower(k, "metal").kernel_source for k in kernels
        ]
        with tempfile.TemporaryDirectory() as directory:
            cache = KernelCache(directory, compiler_identity={})
            parallel = CompilationSession(persistent_cache=cache, workers=2)
            actual = parallel.lower_many(kernels, "metal")
            self.assertEqual([actual[id(k)].kernel_source for k in kernels], expected)
            self.assertEqual(cache.stats["lowering_misses"], 2)
            with patch(
                "concurrent.futures.ProcessPoolExecutor",
                side_effect=AssertionError("warm worker"),
            ):
                warm = CompilationSession(persistent_cache=cache, workers=2).lower_many(
                    kernels, "metal"
                )
            self.assertEqual([warm[id(k)].kernel_source for k in kernels], expected)
            self.assertEqual(cache.stats["lowering_hits"], 2)

    def test_artifact_identity_is_independent_of_parity_baseline(self):
        import kernel_cache

        with (
            patch.object(kernel_cache, "package_files", return_value={}),
            patch(
                "parity.evidence_inputs",
                side_effect=AssertionError("evidence is not compiler input"),
            ),
        ):
            identity = kernel_cache.identity()
        self.assertIn("lowering_worker.py", identity["adapters"])
        self.assertIn("compiler_intrinsics.py", identity["adapters"])
        self.assertNotIn("source", identity)


if __name__ == "__main__":
    unittest.main()
