"""Persistent compiler cache must never substitute state or stale executable code."""

import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_cache import KernelCache
from ordered import CompilationSession


class KernelCacheTests(unittest.TestCase):
    def test_identity_key_corruption_and_incomplete_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            a = KernelCache(directory, compiler_identity={"compiler": "a"})
            self.assertEqual(
                a.artifact("c", {"source": "one", "abi": 1}, lambda: b"one"), b"one"
            )
            fresh = KernelCache(directory, compiler_identity={"compiler": "a"})
            self.assertEqual(
                fresh.artifact(
                    "c",
                    {"source": "one", "abi": 1},
                    lambda: self.fail("compiled on hit"),
                ),
                b"one",
            )
            self.assertEqual(fresh.stats["binary_hits"], 1)
            for key in ({"source": "two", "abi": 1}, {"source": "one", "abi": 2}):
                self.assertEqual(fresh.artifact("c", key, lambda: b"new"), b"new")
            changed = KernelCache(directory, compiler_identity={"compiler": "b"})
            self.assertEqual(
                changed.artifact(
                    "c", {"source": "one", "abi": 1}, lambda: b"compiler-b"
                ),
                b"compiler-b",
            )
            for path in Path(directory).rglob("artifact"):
                path.write_bytes(b"damaged")
            self.assertEqual(
                fresh.artifact("c", {"source": "one", "abi": 1}, lambda: b"repaired"),
                b"repaired",
            )
            self.assertEqual(fresh.stats["invalid_entries"], 1)
            for path in Path(directory).rglob("entry.json"):
                path.unlink()
            self.assertEqual(
                fresh.artifact("c", {"source": "one", "abi": 1}, lambda: b"complete"),
                b"complete",
            )
            with self.assertRaisesRegex(RuntimeError, "builder failed"):
                fresh.artifact(
                    "c",
                    "failed",
                    lambda: (_ for _ in ()).throw(RuntimeError("builder failed")),
                )
            self.assertFalse(list(Path(directory).rglob(".building-*")))

    def test_parallel_publication_is_complete(self):
        with tempfile.TemporaryDirectory() as directory:

            def build(_):
                cache = KernelCache(directory, compiler_identity={})
                return cache.artifact("c", "same", lambda: b"kernel" * 1000)

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                self.assertEqual(
                    list(pool.map(build, range(16))), [b"kernel" * 1000] * 16
                )
            cache = KernelCache(directory, compiler_identity={})
            self.assertEqual(
                cache.artifact("c", "same", lambda: self.fail("miss")), b"kernel" * 1000
            )

    def test_lowering_survives_process_exit_and_keys_full_ir_target_and_compiler(self):
        import tilelang
        from primitive_kernels import make_kernel

        with tempfile.TemporaryDirectory() as directory:
            code = """import sys
from kernel_cache import KernelCache
from ordered import CompilationSession
from primitive_kernels import make_kernel
cache=KernelCache(sys.argv[1], compiler_identity={"test":1})
CompilationSession(persistent_cache=cache).lower(make_kernel({"rows":2,"storage":70},"c"),"c")
"""
            import os

            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
            subprocess.run(
                [sys.executable, "-c", code, directory],
                env=env,
                check=True,
                capture_output=True,
            )
            cache = KernelCache(directory, compiler_identity={"test": 1})
            with patch.object(tilelang, "lower", wraps=tilelang.lower) as lower:
                a = CompilationSession(persistent_cache=cache).lower(
                    make_kernel({"rows": 2, "storage": 70}, "c"), "c"
                )
                self.assertEqual(lower.call_count, 0)
                b = CompilationSession(persistent_cache=cache).lower(
                    make_kernel({"rows": 3, "storage": 70}, "c"), "c"
                )
                self.assertNotEqual(a.kernel_source, b.kernel_source)
                CompilationSession(persistent_cache=cache).lower(
                    make_kernel({"rows": 2, "storage": 70}, "metal"), "metal"
                )
                different = KernelCache(directory, compiler_identity={"test": 2})
                CompilationSession(persistent_cache=different).lower(
                    make_kernel({"rows": 2, "storage": 70}, "c"), "c"
                )
                self.assertEqual(lower.call_count, 3)

    def test_repeated_stages_load_identical_libraries_only_once(self):
        import copy
        import frame_cpu
        from test_frame_cpu import variant

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v = variant(8)
            stage = copy.deepcopy(v["manifest"]["program"]["stages"][-1])
            stage.update(name="again", source="again.c")
            v["manifest"]["program"]["stages"].append(stage)
            v["sources"]["again.c"] = v["sources"]["commit.c"]
            payload = KernelCache(root / "cache", compiler_identity={}).c_library(
                v["sources"]["commit.c"], stage["bindings"]
            )
            v["libraries"] = {}
            for name in ("commit.c", "again.c"):
                path = root / (name + ".native")
                path.write_bytes(payload)
                v["libraries"][name] = path
            with patch.object(
                frame_cpu, "load_c_library", wraps=frame_cpu.load_c_library
            ) as load:
                session = frame_cpu.CpuFrameSession([v], root)
                session.step()
                self.assertEqual(session.storage["State"][0], 44)
                session.reset()
                session.step()
                self.assertEqual(session.storage["State"][0], 44)
                self.assertEqual(load.call_count, 1)

    def test_cached_c_library_executes_with_fresh_state_and_changed_source(self):
        import numpy as np
        import replay

        source = '#include <cstdint>\nextern "C" int32_t step(uint32_t* a) {a[0]+=3;return 0;}'
        abi = {"schema": 1, "entry": "step", "buffers": ["A"]}
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            cache = KernelCache(root / "cache", compiler_identity={"test": 1})
            with patch.object(replay, "compile_c", wraps=replay.compile_c) as compile_c:
                payload = cache.c_library(source, abi)
                fresh = KernelCache(root / "cache", compiler_identity={"test": 1})
                self.assertEqual(fresh.c_library(source, abi), payload)
                self.assertEqual(compile_c.call_count, 1)
                library = root / "loaded.dylib"
                library.write_bytes(payload)
                native, function, _ = replay.load_c_library(library, abi, {})
                a, b = np.array([10], np.uint32), np.array([100], np.uint32)
                self.assertEqual(function(a.ctypes.data), 0)
                self.assertEqual(function(b.ctypes.data), 0)
                self.assertEqual((int(a[0]), int(b[0])), (13, 103))
                fresh.c_library(source.replace("+=3", "+=7"), abi)
                self.assertEqual(compile_c.call_count, 2)
                del native, function
            for path in (root / "cache").rglob("entry.json"):
                metadata = json.loads(path.read_text())
                self.assertNotIn("initial", metadata)


class CompilerSelectionTests(unittest.TestCase):
    def test_reference_build_only_selects_its_requested_binary(self):
        import replay

        with patch.object(replay, "run") as run:
            for reference, frame, expected in (
                (False, False, "native-replay"),
                (False, True, "native-frame"),
                (True, False, "rust-reference"),
            ):
                binary = replay.build_native(reference, frame=frame)
                command = run.call_args.args[0]
                self.assertEqual(command[command.index("--bin") + 1], expected)
                self.assertEqual(binary.name, expected)
                self.assertEqual("--features" in command, reference)

    def test_missing_metal_toolchain_keeps_runtime_compilation_fallback(self):
        import kernel_cache

        with tempfile.TemporaryDirectory() as directory:
            cache = KernelCache(directory, compiler_identity={})
            with (
                patch.object(
                    kernel_cache.subprocess,
                    "run",
                    return_value=type("Probe", (), {"stdout": "/tool/metal"})(),
                ),
                patch.object(
                    kernel_cache,
                    "executable_identity",
                    side_effect=OSError("SDK missing"),
                ) as probe,
            ):
                self.assertIsNone(cache.metal_library("source"))
                self.assertIsNone(cache.metal_library("source"))
                self.assertEqual(probe.call_count, 1)
                self.assertEqual(cache.stats["binary_misses"], 0)

    def test_native_compiler_identity_includes_sdk_selection(self):
        import kernel_cache

        with (
            patch.object(kernel_cache.platform, "system", return_value="Darwin"),
            patch.object(
                kernel_cache, "apple_sdk_identity", return_value={"sdk": "one"}
            ),
        ):
            first = kernel_cache.executable_identity(["c++"])
        with (
            patch.object(kernel_cache.platform, "system", return_value="Darwin"),
            patch.object(
                kernel_cache, "apple_sdk_identity", return_value={"sdk": "two"}
            ),
        ):
            second = kernel_cache.executable_identity(["c++"])
        self.assertNotEqual(first, second)
