"""Compile immutable CUDA PTX / Ascend launch libraries on the target host.

No driver is loaded here. Compilation can run on a host with the vendor toolkit
and no accelerator; only successful device execution establishes execution evidence.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def launch_metadata(function):
    """Retain compiler-owned dynamic shared-memory demand in the exported ABI."""
    attrs = function.attrs
    shared = int(attrs.get("dyn_shared_memory_buf", 0))
    if shared < 0 or shared > 0xFFFFFFFF:
        raise ValueError("Invalid generated shared-memory size")
    if (
        attrs.get("use_cooperative_groups", False)
        or attrs.get("cluster_dims") is not None
    ):
        raise ValueError(
            "Cooperative/cluster launches need an explicit execution contract"
        )
    return {"shared_memory_bytes": shared}


def ascend_wrapper(source, bindings):
    """Follow the pinned TLAscendSourceWrapper triple-chevron ABI.

    Exported Nexus tensors are uint32 storage words. Reject a different generated
    signature rather than guessing argument types or relying on implicit casts.
    """
    entry = bindings["entry"]
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", entry):
        raise ValueError("Invalid kernel entry")
    declaration = re.search(
        r'extern\s+"C"\s+__global__\s+__vector__\s+void\s+'
        + re.escape(entry)
        + r"\s*\(([^)]*)\)",
        source,
    )
    if declaration is None:
        raise ValueError("Missing generated Ascend entry declaration")
    parameters = [p.strip() for p in declaration[1].split(",") if p.strip()]
    if len(parameters) != len(bindings["buffers"]):
        raise ValueError("Ascend argument count differs from binding manifest")
    arguments = []
    for i, parameter in enumerate(parameters):
        # Lowered parameters may be const/restrict; every public buffer is words.
        if not re.fullmatch(
            r"(?:const\s+)?__gm__\s+(?:const\s+)?uint32_t\s*\*\s*(?:__restrict__\s+)?[A-Za-z_]\w*",
            parameter,
        ):
            raise ValueError(f"Unsupported generated Ascend parameter: {parameter}")
        arguments.append(f"reinterpret_cast<uint32_t*>(args[{i}])")
    shared = bindings.get("shared_memory_bytes", 0)
    return (
        source
        + f"""
// Host adapter follows pinned TileLang Ascend 950 SIMT launch semantics.
extern "C" int32_t nexus_launch(uint32_t blocks, void* stream, void** args) {{
  {entry}<<<blocks, {shared}, stream>>>({", ".join(arguments)});
  return static_cast<int32_t>(aclrtGetLastError(ACL_RT_THREAD_LEVEL));
}}
"""
    )


class DeviceCompiler:
    def __init__(self, target, *, architecture=None, executable=None, templates=None):
        if target not in ("cuda", "ascend"):
            raise ValueError("Unsupported device compiler")
        self.target = target
        self.architecture = architecture or (
            "compute_80" if target == "cuda" else "dav-3510"
        )
        if not re.fullmatch(
            r"compute_[0-9]+a?" if target == "cuda" else r"dav-[0-9]+",
            self.architecture,
        ):
            raise ValueError("Invalid device architecture")
        name = "nvcc" if target == "cuda" else "bisheng"
        candidate = (
            executable
            or os.environ.get("NEXUS_TILELANG_" + name.upper())
            or shutil.which(name)
        )
        if candidate is None:
            raise RuntimeError(
                f"{name} not found: install the target toolkit or set NEXUS_TILELANG_{name.upper()}; no CPU fallback"
            )
        self.executable = str(Path(candidate).resolve())
        if templates is None:
            import tilelang

            templates = Path(tilelang.__file__).resolve().parent
        self.templates = Path(templates)
        if not (self.templates / "src/tl_templates" / target).is_dir():
            raise ValueError("Selected TileLang installation lacks target headers")
        version = subprocess.check_output(
            [self.executable, "--version"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        headers = {}
        for base in (
            "src/tl_templates",
            "3rdparty/cutlass/include",
            "3rdparty/cutlass/tools/util/include",
        ):
            for p in sorted((self.templates / base).rglob("*")):
                if p.is_file() and p.suffix in (".h", ".hpp", ".cuh"):
                    headers[str(p.relative_to(self.templates))] = sha256(p.read_bytes())
        self.identity = {
            "schema": 1,
            "target": target,
            "architecture": self.architecture,
            "compiler": self.executable,
            "version": version,
            "compiler_sha256": sha256(Path(self.executable).read_bytes()),
            "headers_sha256": sha256(json.dumps(headers, sort_keys=True).encode()),
            "environment": {
                n: os.environ.get(n)
                for n in (
                    "CUDA_HOME",
                    "CUDA_PATH",
                    "ASCEND_HOME_PATH",
                    "BISHENG_HOME",
                    "CPATH",
                    "CPLUS_INCLUDE_PATH",
                    "LIBRARY_PATH",
                )
            },
        }

    def command(self, source, output):
        includes = ["-I" + str(self.templates / "src")]
        if self.target == "cuda":
            includes += [
                "-I" + str(self.templates / p)
                for p in (
                    "3rdparty/cutlass/include",
                    "3rdparty/cutlass/tools/util/include",
                )
            ]
            return [
                self.executable,
                "-x",
                "cu",
                "--ptx",
                "-std=c++17",
                "-O3",
                "--fmad=false",
                "--prec-div=true",
                "--prec-sqrt=true",
                "--ftz=false",
                "-arch=" + self.architecture,
                *includes,
                str(source),
                "-o",
                str(output),
            ]
        toolkit = os.environ.get("ASCEND_HOME_PATH")
        if toolkit:
            includes += [
                "-I" + str(Path(toolkit) / "include"),
                "-L" + str(Path(toolkit) / "lib64"),
                "-Wl,-rpath," + str(Path(toolkit) / "lib64"),
            ]
        return [
            self.executable,
            "-O2",
            "-fPIC",
            "-std=c++20",
            "-ffp-contract=off",
            "-mllvm",
            "-cce-aicore-dcpreload-args=false",
            "--npu-arch=" + self.architecture,
            "--shared",
            *includes,
            str(source),
            "-lascendcl",
            "-o",
            str(output),
        ]

    def compile(self, source, bindings, directory):
        code = ascend_wrapper(source, bindings) if self.target == "ascend" else source
        abi = {k: bindings[k] for k in ("entry", "buffers")}
        abi["shared_memory_bytes"] = bindings.get("shared_memory_bytes", 0)
        key = {
            **self.identity,
            "source_sha256": sha256(code.encode()),
            "abi": abi,
            "flags": "strict-f32-v1",
        }
        key_hash = sha256(json.dumps(key, sort_keys=True).encode())
        root = Path(os.environ.get("NEXUS_TILELANG_DEVICE_CACHE", directory)).resolve()
        root.mkdir(parents=True, exist_ok=True)
        destination = root / key_hash
        filename = "kernel.ptx" if self.target == "cuda" else "kernel.so"
        try:
            receipt = json.loads((destination / "build.json").read_text())
            binary = destination / filename
            if receipt["key"] == key and receipt["binary_sha256"] == sha256(
                binary.read_bytes()
            ):
                return binary, receipt
        except (OSError, ValueError, KeyError):
            pass
        with tempfile.TemporaryDirectory(prefix="build-", dir=root) as tmp:
            tmp = Path(tmp)
            src = tmp / ("kernel.cu" if self.target == "cuda" else "kernel.asc")
            src.write_text(code)
            binary = tmp / filename
            command = self.command(src, binary)
            try:
                result = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=180,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                result = subprocess.CompletedProcess(command, -1, str(error))
            if result.returncode or not binary.is_file():
                failed = root / (key_hash + ".failed")
                failed.mkdir(exist_ok=True)
                (failed / src.name).write_text(code)
                (failed / "build.json").write_text(
                    json.dumps(
                        {
                            "key": key,
                            "command": command,
                            "log": result.stdout,
                            "returncode": result.returncode,
                            "level": "device_compile_failed",
                        },
                        indent=2,
                    )
                    + "\n"
                )
                raise RuntimeError(
                    f"{self.target} device compilation failed ({result.returncode}); source and log: {failed}:\n{result.stdout}"
                )
            receipt = {
                "key": key,
                "command": command,
                "log": result.stdout,
                "binary_sha256": sha256(binary.read_bytes()),
                "level": "device_compiled",
                "device_execution": "not_run",
            }
            (tmp / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
            # Cache is trusted local executable storage, never simulation state.
            destination.mkdir(exist_ok=True)
            for name in (filename, src.name, "build.json"):
                os.replace(tmp / name, destination / name)
        return destination / filename, receipt
