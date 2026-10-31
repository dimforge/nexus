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
import threading
import uuid
from types import SimpleNamespace

from replay import ROOT

SCHEMA = 2
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
        # Kernel builders/layouts are represented by the full input IR. Reference
        # code, tests, docs and evidence policies are not compiler inputs. Keep
        # code-generation hooks and compiler adapters in this artifact identity;
        # parity.evidence_inputs remains deliberately broad and unchanged.
        "adapters": {
            name: digest((Path(__file__).parent / name).read_bytes())
            for name in (
                "kernel_cache.py",
                "ordered.py",
                "lowering_worker.py",
                "compiler_intrinsics.py",
                "replay.py",
            )
        },
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


def apple_sdk_identity():
    return {
        option: subprocess.run(
            ["xcrun", "--sdk", "macosx", option],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        for option in ("--show-sdk-path", "--show-sdk-build-version")
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
        "sdk": apple_sdk_identity() if platform.system() == "Darwin" else None,
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
            "program_hits": 0,
            "program_misses": 0,
            "lowering_hits": 0,
            "lowering_misses": 0,
            "binary_hits": 0,
            "binary_misses": 0,
            "invalid_entries": 0,
        }
        self._stats_lock = threading.Lock()
        self._compiler_lock = threading.Lock()
        self._c_compiler = None
        self._metal_compiler = None
        self._metal_probed = False

    def _entry(self, kind, key):
        signature = {
            "schema": SCHEMA,
            "identity": self.identity_hash,
            "kind": kind,
            "key": key,
        }
        return self.directory / kind / digest(encoded(signature)), signature

    def _increment(self, name):
        with self._stats_lock:
            self.stats[name] += 1

    def lookup(self, kind, key):
        category = kind if kind in ("program", "lowering") else "binary"
        entry, signature = self._entry(kind, key)
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
            self._increment(category + "_hits")
            return payload
        except (OSError, ValueError, TypeError):
            if entry.exists() or entry.is_symlink():
                self._increment("invalid_entries")
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
        self._increment(category + "_misses")
        return None

    def publish(self, kind, key, payload):
        entry, signature = self._entry(kind, key)
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

    def artifact(self, kind, key, build):
        payload = self.lookup(kind, key)
        return self.publish(kind, key, build()) if payload is None else payload

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

        with self._compiler_lock:
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
        with self._compiler_lock:
            if not self._metal_probed:
                self._metal_probed = True
                try:
                    tool = subprocess.run(
                        ["xcrun", "--find", "metal"],
                        check=True,
                        capture_output=True,
                        text=True,
                    ).stdout.strip()
                    # Publish only after every probe succeeds. A partial probe must
                    # consistently use runtime source compilation on later stages.
                    compiler = executable_identity([tool])
                    self._metal_compiler = compiler
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
