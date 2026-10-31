"""Build command/cache contracts with a fake compiler; no vendor compilation claim."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from device_compiler import DeviceCompiler


class DeviceCompilerTests(unittest.TestCase):
    def test_strict_flags_cache_reuse_corruption_and_failed_build_logs(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ):
            os.environ.pop("NEXUS_TILELANG_DEVICE_CACHE", None)
            root = Path(tmp)
            for target in ("cuda", "ascend"):
                (root / "src/tl_templates" / target).mkdir(parents=True)
            executable = root / "fake-compiler"
            executable.write_text(
                f"#!{sys.executable}\n"
                + """import sys
from pathlib import Path
if '--version' in sys.argv:
 print('TEST ONLY compiler 1');raise SystemExit(0)
source=next(Path(a) for a in sys.argv if a.endswith(('.cu','.asc')))
if 'FAIL_TEST' in source.read_text():
 print('TEST ONLY intentional compile failure');raise SystemExit(2)
Path(sys.argv[sys.argv.index('-o')+1]).write_bytes(b'TEST ONLY binary')
"""
            )
            executable.chmod(0o755)
            compiler = DeviceCompiler("cuda", executable=executable, templates=root)
            bindings = {"entry": "kernel", "buffers": ["X"], "shared_memory_bytes": 256}
            binary, receipt = compiler.compile("example", bindings, root / "cache")
            original = binary.stat().st_mtime_ns
            self.assertEqual(receipt["device_execution"], "not_run")
            compiler.compile("example", bindings, root / "cache")
            self.assertEqual(binary.stat().st_mtime_ns, original)
            binary.write_bytes(b"corrupt")
            compiler.compile("example", bindings, root / "cache")
            self.assertEqual(binary.read_bytes(), b"TEST ONLY binary")
            second, _ = compiler.compile(
                "example", dict(bindings, shared_memory_bytes=512), root / "cache"
            )
            self.assertNotEqual(binary, second)
            with self.assertRaisesRegex(RuntimeError, "source and log"):
                compiler.compile("FAIL_TEST", bindings, root / "cache")
            failed = list((root / "cache").glob("*.failed"))
            self.assertEqual(len(failed), 1)
            self.assertEqual((failed[0] / "kernel.cu").read_text(), "FAIL_TEST")
            self.assertIn(
                "intentional compile failure", (failed[0] / "build.json").read_text()
            )
            command = compiler.command(Path("x.cu"), Path("x.ptx"))
            for flag in (
                "--fmad=false",
                "--ftz=false",
                "--prec-div=true",
                "--prec-sqrt=true",
                "-arch=compute_80",
            ):
                self.assertIn(flag, command)
            ascend = DeviceCompiler("ascend", executable=executable, templates=root)
            command = ascend.command(Path("x.asc"), Path("x.so"))
            self.assertIn("--npu-arch=dav-3510", command)
            self.assertIn("-ffp-contract=off", command)
