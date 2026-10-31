"""Opt-in cache of immutable compiler artifacts, never simulation buffers/results.

Entries are local trusted executable code, like the explicitly selected compiler.
An integrity failure is a miss. Publication uses a complete temporary directory;
concurrent writers may duplicate compilation but cannot expose partial entries.
"""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import tempfile
import uuid
from types import SimpleNamespace

from parity import evidence_inputs
from replay import ROOT

SCHEMA = 1
ENVIRONMENT = (
    "PATH",
    "CXX",
    "DEVELOPER_DIR",
    "SDKROOT",
    "MACOSX_DEPLOYMENT_TARGET",
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "LIBRARY_PATH",
    "COMPILER_PATH",
)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def package_files(directory):
    """Include compiler Python, shared libraries and generated-C include files."""
    directory = Path(directory)
    return {
        str(p.relative_to(directory)): digest(p.read_bytes())
        for p in sorted(directory.rglob("*"))
        if p.is_file()
        and p.suffix in (".py", ".so", ".dylib", ".dll", ".h", ".hpp")
        and "__pycache__" not in p.parts
    }


def identity():
    import tilelang
    import tvm_ffi

    return {
        "schema": SCHEMA,
        "source": evidence_inputs(ROOT),
        "host": [
            platform.system(),
            platform.release(),
            platform.version(),
            platform.machine(),
        ],
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("tilelang", "apache-tvm-ffi", "numpy")
        },
        "compiler_files": {
            "tilelang": package_files(Path(tilelang.__file__).parent),
            "tvm": package_files(Path(tilelang.tvm.__file__).parent),
            "ffi": package_files(Path(tvm_ffi.__file__).parent),
        },
        "environment": {name: os.environ.get(name) for name in ENVIRONMENT},
    }


def executable_identity(command):
    path = shutil.which(command[0])
    if path is None:
        raise FileNotFoundError(command[0])
    path = Path(path).resolve()
    version = subprocess.run(
        [*command, "--version"], check=True, capture_output=True, text=True
    ).stdout
    return {
        "command": command,
        "executable": str(path),
        "sha256": digest(path.read_bytes()),
        "version": version,
    }


class KernelCache:
    def __init__(self, directory, *, compiler_identity=None):
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity = identity() if compiler_identity is None else compiler_identity
        self.identity_hash = digest(encoded(self.identity))
        self.stats = {
            "lowering_hits": 0,
            "lowering_misses": 0,
            "binary_hits": 0,
            "binary_misses": 0,
            "invalid_entries": 0,
        }
        self._c_compiler = None
        self._metal_compiler = None
        self._metal_probed = False

    def artifact(self, kind, key, build):
        category = "lowering" if kind == "lowering" else "binary"
        signature = {
            "schema": SCHEMA,
            "identity": self.identity_hash,
            "kind": kind,
            "key": key,
        }
        entry = self.directory / kind / digest(encoded(signature))
        try:
            if entry.is_symlink():
                raise ValueError("Cache entry is a symlink")
            manifest_path, payload_path = entry / "entry.json", entry / "artifact"
            if manifest_path.is_symlink() or payload_path.is_symlink():
                raise ValueError("Cache artifact is a symlink")
            metadata = json.loads(manifest_path.read_text())
            payload = payload_path.read_bytes()
            if metadata != {
                "signature": signature,
                "sha256": digest(payload),
                "bytes": len(payload),
            }:
                raise ValueError("Cache integrity mismatch")
            self.stats[category + "_hits"] += 1
            return payload
        except (OSError, ValueError, TypeError):
            if entry.exists() or entry.is_symlink():
                self.stats["invalid_entries"] += 1
                # Move the invalid entry aside before replacing it. A concurrent
                # reader either has complete bytes already or observes a miss.
                quarantine = entry.with_name(
                    entry.name + ".invalid-" + uuid.uuid4().hex
                )
                try:
                    entry.rename(quarantine)
                except FileNotFoundError:
                    pass
                else:
                    if quarantine.is_symlink() or quarantine.is_file():
                        quarantine.unlink()
                    else:
                        shutil.rmtree(quarantine)
        self.stats[category + "_misses"] += 1
        payload = build()
        if not isinstance(payload, bytes):
            raise TypeError("Cache builders must return bytes")
        entry.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=entry.parent))
        try:
            (temporary / "artifact").write_bytes(payload)
            (temporary / "entry.json").write_bytes(
                encoded(
                    {
                        "signature": signature,
                        "sha256": digest(payload),
                        "bytes": len(payload),
                    }
                )
            )
            try:
                temporary.rename(entry)
            except OSError:
                if not entry.is_dir():
                    raise
                # Another complete entry won publication. This caller uses its
                # own freshly compiled bytes, never an unchecked competing entry.
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return payload

    def lower(self, kernel, target, build):
        import tilelang

        tvm = tilelang.tvm
        ir = tvm.ir.save_json(kernel)

        def encode_artifact():
            artifact = build()
            return encoded(
                {
                    "device_mod": tvm.ir.save_json(artifact.device_mod),
                    "kernel_source": artifact.kernel_source,
                }
            )

        payload = json.loads(
            self.artifact("lowering", {"target": target, "ir": ir}, encode_artifact)
        )
        return SimpleNamespace(
            device_mod=tvm.ir.load_json(payload["device_mod"]),
            kernel_source=payload["kernel_source"],
        )

    def c_library(self, source, bindings):
        from replay import compile_c

        if self._c_compiler is None:
            self._c_compiler = executable_identity(
                shlex.split(os.environ.get("CXX", "c++"))
            )
        key = {"source": source, "abi": bindings, "compiler": self._c_compiler}

        def build():
            with tempfile.TemporaryDirectory(prefix="nexus-c-kernel-") as name:
                directory = Path(name)
                # The ordinary uncached compiler remains the numerical reference
                # for compilation flags and headers. Its source is in identity.
                native, function, _ = compile_c(source, bindings, directory)
                payload = Path(native._name).read_bytes()
                del function, native
                return payload

        return self.artifact("c", key, build)

    def metal_library(self, source):
        if not self._metal_probed:
            self._metal_probed = True
            try:
                tool = subprocess.run(
                    ["xcrun", "--find", "metal"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                self._metal_compiler = executable_identity([tool])
                self._metal_compiler["sdk"] = subprocess.run(
                    ["xcrun", "--sdk", "macosx", "--show-sdk-build-version"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            except (OSError, subprocess.CalledProcessError):
                return None
        if self._metal_compiler is None:
            return None
        flags = ["-fno-fast-math", "-ffp-contract=off"]
        key = {
            "source": source,
            "compiler": self._metal_compiler,
            "flags": flags,
            "sdk": "macosx",
        }

        def build():
            with tempfile.TemporaryDirectory(prefix="nexus-metal-kernel-") as name:
                directory = Path(name)
                source_path, library = (
                    directory / "kernel.metal",
                    directory / "kernel.metallib",
                )
                source_path.write_text(source)
                subprocess.run(
                    [
                        "xcrun",
                        "-sdk",
                        "macosx",
                        "metal",
                        *flags,
                        str(source_path),
                        "-o",
                        str(library),
                    ],
                    check=True,
                    capture_output=True,
                )
                return library.read_bytes()

        return self.artifact("metal", key, build)
