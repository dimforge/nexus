# Linux ARM64 compiler qualification

The production compiler.py helper initially failed in the CANN 9.2.1 Ubuntu
22.04 container because its system CMake was too old. The helper now installs
pinned CMake/Ninja and Linux patchelf, prepends their venv directory to PATH, and
records the build-tool requirements. The same clean-source build passed after
that fix; the compiler checkout was restored clean after the CUDA patch.

compiler-receipt.json records the exact source/submodule revisions, wheel hash,
Python and platform. The wheel remains at:
`/private/tmp/nexus-linux-compiler-0153/wheels/` on this machine. It is a Linux
ARM64 wheel, not an x86 or macOS package. Use compiler.py to build on those hosts.

The pinned image is:
`quay.io/ascend/cann:9.2.1-950-ubuntu22.04-py3.12@sha256:1fa0e2b6754034d9683c5718dfc12a666fc340dafb15966bdf4641ed1045c1b1`.
The build used six compiler jobs; the source probe used four workers. No device
was exposed. The helper command inside the container was:

```sh
python3 /nexus-port/compiler.py /private/tmp/nexus-linux-compiler-0153 \
  --source /private/tmp/nexus-linux-compiler-0153/source --cuda-codegen-only --jobs 6
```

After installing the resulting wheel with pip --no-deps, source-probe.py re-lowered
both full resident schedules from verified pre-existing native input archives.
It compares all 92 unique sources and the complete 416-stage frame manifest per
target. Both CUDA and Ascend matched the macOS-generated sources that passed real
NVCC/PTXAS and Bisheng compilation. It does not export fresh Rust answers, execute
physics, simulate a device, or run the full parity matrix. The report records its
exact source snapshot, before the subsequent documentation update.

Reproduce the comparison with the installed compiler interpreter:

```sh
python source-probe.py REPOSITORY PREPARED_ROOT NEW_OUTPUT
```

PREPARED_ROOT must contain cuda/ and ascend/ specializations from device_prepare.py.
The recorded run used results-device-0149. The 13 focused host ABI/compiler/frame/
replay regressions also passed on the newly installed Linux wheel (1.559 seconds).
These use explicitly identified host test shims; they are not hardware execution.
