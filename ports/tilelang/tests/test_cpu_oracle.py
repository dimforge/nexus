"""Keep an unsafe or unverified khal CPU oracle out of accepted evidence."""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cpu_oracle


class CpuOracleTests(unittest.TestCase):
    def test_effective_configuration_is_explicit(self):
        inherited = {
            "PATH": "unchanged",
            "RUSTC_BOOTSTRAP": "0",
            "RUSTUP_TOOLCHAIN": "other",
        }
        env = cpu_oracle.environment(inherited)
        self.assertEqual(env["PATH"], "unchanged")
        self.assertEqual(env["RUSTC_BOOTSTRAP"], "1")
        self.assertEqual(env["RUSTUP_TOOLCHAIN"], cpu_oracle.toolchain())
        self.assertEqual(env["CARGO_ENCODED_RUSTFLAGS"], "-Zmutable-noalias=no")
        self.assertNotIn("RUSTFLAGS", env)
        self.assertNotIn("CARGO_ENCODED_RUSTFLAGS", inherited)

    def test_encoded_flags_cannot_silently_enable_noalias(self):
        for key in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS"):
            for value in (
                "-Zmutable-noalias=yes",
                "-Copt-level=3",
                "-Zmutable-noalias=no -Zmutable-noalias=yes",
            ):
                with (
                    self.subTest(key=key, value=value),
                    self.assertRaisesRegex(ValueError, "custom"),
                ):
                    cpu_oracle.environment({key: value})
        env = cpu_oracle.environment(
            {
                "RUSTFLAGS": "-Zmutable-noalias=yes",
                "CARGO_ENCODED_RUSTFLAGS": "-Zmutable-noalias=no",
            }
        )
        self.assertNotIn("RUSTFLAGS", env)

    def test_compiler_wrappers_cannot_bypass_the_probe(self):
        for name in ("RUSTC", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                cpu_oracle.environment({name: "unrecorded-wrapper"})

    def test_llvm_guard_checks_both_reference_kinds(self):
        good = "define i32 @mutable_reference_probe(ptr %x, ptr %y) {\ndefine i32 @shared_reference_probe(ptr noalias %x) {\n"
        cpu_oracle.check_llvm(good)
        for bad in (
            good.replace("ptr %x", "ptr noalias %x"),
            good.replace("noalias", ""),
            good.replace("mutable_reference_probe", "missing"),
        ):
            with self.assertRaises(ValueError):
                cpu_oracle.check_llvm(bad)

    def test_live_compiler_attributes(self):
        with tempfile.TemporaryDirectory() as directory:
            result = cpu_oracle.verify(directory, cpu_oracle.environment())
            self.assertFalse(result["mutable_reference_noalias"])
            self.assertTrue(result["shared_reference_noalias"])
            self.assertEqual(result["rustflags"], list(cpu_oracle.FLAGS))
            self.assertTrue((Path(directory) / "cpu-oracle-noalias.ll").is_file())


if __name__ == "__main__":
    unittest.main()
