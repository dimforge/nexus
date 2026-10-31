#!/usr/bin/env python3
"""Compile a prepared CUDA/Ascend scene with vendor compilers in device-free Docker.

The host keeps Python/TileLang; the Linux container only sees compiler headers and
an isolated build directory. No accelerator, CPU physics or matrix execution.
"""

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from device_compiler import DeviceCompiler
from device_replay import load_bundle

CUDA_IMAGE = "nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6"

ASCEND_IMAGE = "quay.io/ascend/cann:9.2.1-950-ubuntu22.04-py3.12@sha256:1fa0e2b6754034d9683c5718dfc12a666fc340dafb15966bdf4641ed1045c1b1"


def command(args, **kwargs):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--target", choices=("cuda", "ascend"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--image",
        default=None,
        help="Explicit compiler-image override; recorded in receipt",
    )
    args = parser.parse_args()
    args.image = args.image or (CUDA_IMAGE if args.target == "cuda" else ASCEND_IMAGE)
    if not 1 <= args.jobs <= 16:
        parser.error("--jobs must be 1..16")
    import tilelang

    templates = Path(tilelang.__file__).resolve().parent
    variants, _ = load_bundle(args.directory, args.target)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    cache = root / "cache"
    cache.mkdir()
    report = {
        "schema": 1,
        "suite": "device_vendor_compile",
        "target": args.target,
        "status": "running",
        "image": args.image,
        "device_execution": "not_run",
        "full_matrix": "not_run",
        "kernels": [],
    }
    path = root / "report.json"
    started = time.monotonic()
    container = "nexus-device-compile-" + uuid.uuid4().hex[:12]
    running = False
    old_cache = os.environ.get("NEXUS_TILELANG_DEVICE_CACHE")
    old_toolkit = os.environ.get("ASCEND_HOME_PATH")
    try:
        details = json.loads(
            command(["docker", "image", "inspect", args.image], timeout=30)
        )[0]
        report["image_identity"] = {
            k: details[k] for k in ("Id", "RepoDigests", "Architecture", "Os")
        }
        command(
            [
                "docker",
                "run",
                "--rm",
                "-d",
                "--pull=never",
                "--network=none",
                "--cpus",
                str(args.jobs),
                "--memory",
                "8g",
                "--name",
                container,
                "--mount",
                f"type=bind,source={templates},target={templates},readonly",
                "--mount",
                f"type=bind,source={root},target={root}",
                "--entrypoint",
                "sleep",
                args.image,
                "infinity",
            ],
            timeout=60,
        )
        running = True
        tool = "nvcc" if args.target == "cuda" else "bisheng"
        wrapper = root / tool
        invocation = ["docker", "exec", container]
        if args.target == "cuda":
            invocation += ["/usr/local/cuda/bin/nvcc"]
        else:
            invocation += ["/bin/bash", "-lc", 'exec bisheng "$@"', "bash"]
            toolkit = command(
                ["docker", "exec", container, "printenv", "ASCEND_HOME_PATH"],
                timeout=30,
            ).strip()
            if not toolkit.startswith("/") or "\n" in toolkit:
                raise ValueError("Invalid container CANN installation path")
            os.environ["ASCEND_HOME_PATH"] = toolkit
            report["cann_home"] = toolkit
        wrapper.write_text("#!/bin/sh\nexec " + shlex.join(invocation) + ' "$@"\n')
        wrapper.chmod(0o755)
        os.environ["NEXUS_TILELANG_DEVICE_CACHE"] = str(cache)
        compiler = DeviceCompiler(args.target, executable=wrapper, templates=templates)
        report["compiler"] = compiler.identity
        jobs = {}
        stages = 0
        for variant in variants:
            for stage in variant["manifest"]["program"]["stages"]:
                stages += 1
                source = variant["sources"][stage["source"]]
                bindings = stage["bindings"]
                key = (
                    source,
                    bindings["entry"],
                    tuple(bindings["buffers"]),
                    bindings.get("shared_memory_bytes", 0),
                )
                if key not in jobs:
                    jobs[key] = (source, bindings, [])
                jobs[key][2].append(stage["name"])
        report["stages"] = stages
        report["unique_kernels"] = len(jobs)
        path.write_text(json.dumps(report, indent=2) + "\n")

        def build(job):
            source, bindings, names = job
            row = {
                "stages": names,
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "shared_memory_bytes": bindings.get("shared_memory_bytes", 0),
            }
            try:
                binary, receipt = compiler.compile(source, bindings, cache)
                if args.target == "ascend":
                    symbols = command(
                        ["docker", "exec", container, "nm", "-D", str(binary)],
                        timeout=30,
                    )
                    if not any(
                        line.split()[-2:] == ["T", "nexus_launch"]
                        for line in symbols.splitlines()
                    ):
                        raise ValueError(
                            "Bisheng library lacks the public host launch symbol"
                        )
                    sections = command(
                        ["docker", "exec", container, "readelf", "-SW", str(binary)],
                        timeout=30,
                    )
                    device_code = re.search(
                        r"\]\s+\.aicore_binary\s+PROGBITS\s+\w+\s+\w+\s+([0-9a-fA-F]+)\s",
                        sections,
                    )
                    if device_code is None or int(device_code[1], 16) == 0:
                        raise ValueError(
                            "Bisheng library contains no Ascend device code"
                        )
                    row.update(
                        status="passed",
                        device_binary_bytes=int(device_code[1], 16),
                        binary=str(binary.relative_to(root)),
                        binary_sha256=receipt["binary_sha256"],
                        elf_sections=sections,
                    )
                else:
                    cubin = binary.with_suffix(".cubin")
                    arguments = [
                        "docker",
                        "exec",
                        container,
                        "/usr/local/cuda/bin/ptxas",
                        "-arch=sm_80",
                        "-v",
                        str(binary),
                        "-o",
                        str(cubin),
                    ]
                    output = command(arguments, timeout=180)
                    (binary.parent / "ptxas.log").write_text(output)
                    row.update(
                        status="passed",
                        ptx=str(binary.relative_to(root)),
                        ptx_sha256=receipt["binary_sha256"],
                        cubin=str(cubin.relative_to(root)),
                        cubin_sha256=hashlib.sha256(cubin.read_bytes()).hexdigest(),
                        ptxas_command=arguments,
                        ptxas_log=output,
                    )
            except (
                ValueError,
                RuntimeError,
                subprocess.SubprocessError,
                OSError,
            ) as error:
                row.update(
                    status="failed",
                    error=str(error),
                    log=getattr(error, "output", None),
                )
            return row

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for row in pool.map(build, jobs.values()):
                report["kernels"].append(row)
                path.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    f"Compiled {len(report['kernels'])}/{len(jobs)}: {row['status']}",
                    flush=True,
                )
        report["status"] = (
            "passed"
            if all(r["status"] == "passed" for r in report["kernels"])
            else "failed"
        )
        report["level"] = (
            "device_compiled"
            if report["status"] == "passed"
            else "device_compile_failed"
        )
    except Exception as error:
        report.update(
            status="failed", error=str(error), log=getattr(error, "output", None)
        )
        raise
    finally:
        if old_cache is None:
            os.environ.pop("NEXUS_TILELANG_DEVICE_CACHE", None)
        else:
            os.environ["NEXUS_TILELANG_DEVICE_CACHE"] = old_cache
        if old_toolkit is None:
            os.environ.pop("ASCEND_HOME_PATH", None)
        else:
            os.environ["ASCEND_HOME_PATH"] = old_toolkit
        if running:
            try:
                cleanup = subprocess.run(
                    ["docker", "stop", "--time", "5", container],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30,
                )
                if cleanup.returncode:
                    report.update(status="failed", cleanup_error=cleanup.stderr)
            except subprocess.SubprocessError as error:
                report.update(status="failed", cleanup_error=str(error))
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Vendor compilation {report['status']}: {path}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
