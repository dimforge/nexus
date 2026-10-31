import os
import re
from pathlib import Path
import subprocess
import tarfile
import tempfile
import tomllib
import unittest



ROOT = Path(__file__).resolve().parents[1]


def published_manifest(source):
    with tempfile.TemporaryDirectory() as directory:
        manifest = Path(directory) / "Cargo.toml"
        manifest.write_text(source)
        result = subprocess.run(["bash", str(ROOT / "scripts/package_tilelang_viewer.sh"), str(manifest)], text=True, capture_output=True)
        if result.returncode:
            if manifest.read_text() != source:
                raise AssertionError("Failed transform modified its input")
            raise ValueError(result.stderr)
        return manifest.read_text()


class ViewerPackagingTests(unittest.TestCase):
    def test_real_manifest_keeps_normal_backends(self):
        before = tomllib.loads((ROOT / "crates/nexus_viewer3d/Cargo.toml").read_text())
        after = tomllib.loads(published_manifest((ROOT / "crates/nexus_viewer3d/Cargo.toml").read_text()))
        self.assertNotIn("tilelang", after["features"])
        self.assertEqual(before["features"]["cpu"], after["features"]["cpu"])
        self.assertEqual(before["features"]["default"], after["features"]["default"])
        self.assertEqual(before["dependencies"], after["dependencies"])
        expected = before
        del expected["features"]["tilelang"]
        expected["features"]["metal"].remove("nexus-tilelang-native?/metal")
        del expected["target"]['cfg(not(target_arch = "wasm32"))']["dependencies"]["nexus-tilelang-native"]
        self.assertEqual(expected, after)

    def test_multiline_change_requires_review_and_preserves_input(self):
        source = (ROOT / "crates/nexus_viewer3d/Cargo.toml").read_text()
        source = source.replace('tilelang = ["cpu", "dep:nexus-tilelang-native"]', 'tilelang = [\n"cpu",\n"dep:nexus-tilelang-native",\n]')
        with self.assertRaises(ValueError):
            published_manifest(source)

    def test_unrecognized_manifest_fails_without_silent_changes(self):
        with self.assertRaises(ValueError):
            published_manifest('[package]\nname="another-crate"\n')

    def test_publish_staging_restores_manifests_and_root_lock_on_failure(self):
        # Exercise the actual staging/cleanup script in a disposable tree. Fake
        # cargo/curl prevent any registry access or publication; CARGO_HOME also
        # confines the script's cache cleanup to this fixture.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            script = (ROOT / "publish.sh").read_text()
            (root / "publish.sh").write_text(script)
            (root / "scripts/package_tilelang_viewer.sh").write_text((ROOT / "scripts/package_tilelang_viewer.sh").read_text())
            (root / "Cargo.toml").write_text('[workspace.package]\nversion = "0.7.0"\n')
            (root / "Cargo.lock").write_text("original locked graph\n")
            entries = re.search(r"CRATES=\(\n(.*?)\n\)", script, re.S).group(1).split()
            originals = {}
            for entry in entries:
                name, shared = entry.split(":")
                folder = root / "crates" / name
                folder.mkdir(parents=True)
                (root / shared).mkdir(exist_ok=True)
                (root / shared / "lib.rs").touch()
                text = (ROOT / "crates/nexus_viewer3d/Cargo.toml").read_text() if name == "nexus_viewer3d" else f'[package]\nname = "{name}"\n[lib]\npath = "../../{shared}/lib.rs"\n'
                (folder / "Cargo.toml").write_text(text)
                originals[name] = text
            fake_bin = root / "fake-bin"
            fake_bin.mkdir()
            commands = {
                "git": "exit 0\n",
                "curl": "exit 22\n",
                "cargo": "cp crates/nexus_viewer3d/Cargo.toml staged-viewer.toml\nprintf 'changed graph\\n' > Cargo.lock\nexit 23\n",
            }
            for name, body in commands.items():
                path = fake_bin / name
                path.write_text("#!/bin/bash\n" + body)
                path.chmod(0o755)
            env = dict(os.environ, PATH=str(fake_bin) + os.pathsep + os.environ["PATH"], CARGO_HOME=str(root / "cargo-home"))
            result = subprocess.run(["bash", str(root / "publish.sh"), "--dry-run", "--no-verify"], env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
            self.assertEqual((root / "Cargo.lock").read_text(), "original locked graph\n")
            for entry in entries:
                name, shared = entry.split(":")
                folder = root / "crates" / name
                self.assertEqual((folder / "Cargo.toml").read_text(), originals[name])
                self.assertFalse((folder / shared).is_symlink())
            staged = tomllib.loads((root / "staged-viewer.toml").read_text())
            self.assertNotIn("tilelang", staged["features"])
            self.assertNotIn("nexus-tilelang-native", str(staged))

    def test_cargo_packages_viewer_while_private_testbed_keeps_its_feature(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Cargo.toml").write_text('[workspace]\nmembers=["viewer","example"]\nresolver="2"\n')
            viewer = '''[package]
name = "nexus_viewer3d"
version = "0.7.0"
edition = "2024"
description = "Local packaging regression fixture"
license = "MIT"
[features]
cpu = []
metal = ["cpu", "nexus-tilelang-native?/metal"]
tilelang = ["cpu", "dep:nexus-tilelang-native"]
[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
nexus-tilelang-native = { path = "../unpublished", optional = true }
'''
            manifests = {
                "viewer": published_manifest(viewer),
                "example": '[package]\nname="testbed-fixture"\nversion="0.1.0"\nedition="2024"\npublish=false\n[dependencies]\nnexus_viewer3d={path="../viewer", version="0.7.0"}\n[features]\ntilelang=["nexus_viewer3d/tilelang"]\n',
            }
            for name, text in manifests.items():
                folder = root / name
                (folder / "src").mkdir(parents=True)
                (folder / "Cargo.toml").write_text(text)
                (folder / "src/lib.rs").write_text("")
            result = subprocess.run(['cargo', 'package', '--offline', '--no-verify', '--allow-dirty', '-p', 'nexus_viewer3d'], cwd=root, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            archive = root / 'target/package/nexus_viewer3d-0.7.0.crate'
            with tarfile.open(archive) as crate:
                packaged = tomllib.loads(crate.extractfile('nexus_viewer3d-0.7.0/Cargo.toml').read().decode())
            self.assertNotIn('tilelang', packaged['features'])
            self.assertNotIn('nexus-tilelang-native', str(packaged))


if __name__ == '__main__':
    unittest.main()
