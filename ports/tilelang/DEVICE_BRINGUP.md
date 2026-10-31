# CUDA and Ascend bring-up

CUDA and Ascend have execution adapters, native `TileLangSimulation` backend
selection, testbed selection, resident frame transactions and contract replay
runners. They are ready for **initial hardware bring-up**, not qualified devices.
This Mac can lower their source, test host ABI/control behavior, and run both
real vendor compilers in Linux containers. Neither accelerator is available.
Metal execution tests the shared virtual-grid algorithm; CUDA/Ascend execution,
memory ordering and numerical behavior still require their actual devices.

The CUDA profile is `compute_80` PTX for Ampere or newer. The provisional Ascend
profile is `dav-3510` / Ascend 950 SIMT from the pinned TileLang backend. This does
not establish compatibility with 910B/910C or any unspecified CANN release. Have
Huawei confirm the device, driver, CANN and Bisheng combination before building.
Articulated multibodies, MPM and 2D remain deferred. Existing numerical discrepancy
0127 and stale full-parity evidence remain visible; no tolerances were changed.

## Prepare without a device

Build the compiler once, in a separate environment:

```sh
python3 ports/tilelang/compiler.py /tmp/nexus-device-compiler --cuda-codegen-only
/tmp/nexus-device-compiler/build-env/bin/python -m pip install --force-reinstall --no-deps /tmp/nexus-device-compiler/wheels/tilelang-*.whl
```

The recorded CMake patch builds CUDA source-generation passes without linking
CUDA runtime/stub libraries. Ascend source generation uses compiler stubs. Neither
provides execution emulation. Preserve `compiler-receipt.json`, its wheel hash and
the local patch hash. The same source-only compiler can run the separate ctypes
execution adapters on a target host with real drivers and toolkits installed.
The helper needs Git, a C/C++ compiler and Python with venv/development headers;
it installs pinned CMake, Ninja and Linux patchelf in its own build environment.
Use that environment's interpreter for the commands below. A clean Linux ARM64
build passed on Ubuntu 22.04 / Python 3.12.16. Its installed wheel regenerated the
same 416-stage CUDA and Ascend scenes with byte-for-byte identical source and
identical frame manifests, then passed 13 focused host tests. This checks Linux
packaging and lowering, not x86 builds or device execution. The receipt, wheel hash
and reproduction script are in evidence/0153-linux-compiler.

Prepare one complete mesh/contact/joint scene (no reference physics execution):

```sh
python ports/tilelang/device_prepare.py --target cuda --target ascend \
  --output ports/tilelang/results-device-ready \
  --cache-directory /tmp/nexus-device-kernel-cache
```

Use the pinned interpreter for `python`. The output directory must be new. This
exports native initial buffers, lowers the full resident schedule, validates
launches, dynamic shared memory, device-controlled grids and Ascend host wrappers,
and records source hashes and preparation times. `--upload PATH` accepts another
native cold-start scene; `--pair-capacity N` chooses its initial allocation.
This deliberately does not invoke the full parity matrix or reconcile a baseline.
First-time source preparation is separate from the 290-second routine smoke.

## Vendor compilation on this Mac

With Docker Desktop running, the pinned interpreter can invoke real NVCC/PTXAS
and Bisheng in vendor Linux ARM64 containers, without an accelerator:

```sh
docker pull nvidia/cuda:13.0.2-devel-ubuntu24.04
python ports/tilelang/device_container_compile.py \
  ports/tilelang/results-device-ready/cuda --target cuda \
  --output /tmp/nexus-cuda-vendor-check

docker pull quay.io/ascend/cann:9.2.1-950-ubuntu22.04-py3.12
python ports/tilelang/device_container_compile.py \
  ports/tilelang/results-device-ready/ascend --target ascend \
  --output /tmp/nexus-ascend-vendor-check
```

The helper pins each image digest and uses the production DeviceCompiler flags
and ABI. CUDA PTX is assembled for `sm_80`; Ascend libraries must export
`nexus_launch` and embed a nonempty `.aicore_binary` device-code section. It mounts only
compiler headers read-only and its isolated output directory read-write; its
owned container has no network or devices and is removed afterward. `--jobs`
controls concurrent compilers, default four. Receipts include image identity,
compiler version, source/binary hashes, commands, and PTXAS resource reports or
Ascend ELF sections. `cuda_container_compile.py` remains a compatibility entrypoint.

CUDA 13.0.88 compiled and assembled all 92 unique kernels in each of the two
416-stage mesh/contact/joint scenes (Coulomb and simplified twist friction), in
26.753 and 26.820 seconds. This validates that compiler/profile combination, not
GPU execution or numerical parity. The general collision kernel uses 255 registers
and a 6,192-byte stack frame with reported spills; profile this on actual hardware
before drawing performance conclusions. See evidence/0150-cuda-vendor-compilation.

CANN 9.2.1 Bisheng compiled all 92 unique kernels in each of the two 416-stage
Ascend scenes for `dav-3510`, in 62.480 and 61.305 seconds. All libraries expose
the host launch entrypoint and contain device code. A sample library also passed
link-symbol resolution inside that container, whose driver dependencies use CANN
development libraries. No ACL initialization or NPU execution was attempted.
These results qualify compilation for this toolkit/profile only; they do not
establish correctness or compatibility on a different Ascend model or CANN version.
See evidence/0152-ascend-vendor-compilation.

## On the CUDA/NPU host

Use Linux (native Windows is unqualified), Rust 1.98.1, the pinned Python packages,
the source checkout at the same revision, and the appropriate vendor toolkit.
For CUDA, install the real NVIDIA driver and NVCC. For Ascend, source CANN's
provided environment script and make `libascendcl.so` and Bisheng available.
`NEXUS_TILELANG_NVCC` / `NEXUS_TILELANG_BISHENG` can name compiler executables;
`NEXUS_TILELANG_DEVICE` selects the device ordinal, default zero.

Start with preflight, then compile the prepared scene independently of execution:

```sh
export NEXUS_TILELANG_DEVICE_CACHE=/tmp/nexus-device-binaries
python ports/tilelang/device_replay.py --target cuda --probe
python ports/tilelang/device_replay.py ports/tilelang/results-device-ready/cuda \
  --target cuda --compile-only
python ports/tilelang/device_replay.py ports/tilelang/results-device-ready/cuda \
  --target cuda --steps 1
```

Replace `cuda` with `ascend` throughout for the NPU. Here `--compile-only` means
**vendor binary compilation** (PTX / a Bisheng launch library). Existing parity
runner `--compile-only` options continue to mean **source lowering**. Preflight
checks driver, device and compiler availability; it does not execute a kernel.
Scene replay writes `.actual.bin`, `stats.json`, device identity and `device-run.json`.
It checks frame gates but does not compare trajectories. Compilation failures retain
the generated source, exact compiler command and output under `*.failed` in the
binary cache; failures never fall back to CPU physics.

Then run a small fresh Rust comparison before the routine suite:

```sh
python ports/tilelang/body_replay.py --operation free_step --target cuda \
  --case n65-b1-c70-s4-i32-g1 --output ports/tilelang/results-device-first
python ports/tilelang/smoke.py --target cuda
```

The smoke preserves all 11 selected physics cases and the 290-second total budget.
Its host tooling tests run on C and are explicitly labelled `unit_backend: c`;
physics cases execute on the requested device. A timeout is incomplete, not a
pass. First-time vendor builds may need diagnosis/preparation before this fits.
The full matrix still runs only on an explicit user request. Device execution,
synchronization and performance remain unqualified until these tests actually run.

For visual testing:

```sh
python3 scripts/run_tilelang_testbed.py --python /absolute/path/to/python \
  --backend cuda --release
```

Use `--backend ascend` for the NPU. The `tilelang` feature exposes both selections;
no Rust `cuda` feature is needed for TileLang's driver adapter. `--compare` adds
the existing Rust **CPU** reference; leave it off for performance measurements.
The launcher always disables CPU mutable-reference noalias for scene conversion.
The GUI needs a working display/renderer independently of the compute accelerator.

## Scheduling and maintenance

Physics stays in the mirrored Rust shader paths. `virtual_dispatch.py` transforms
original IR before lowering: at most 256 physical groups traverse device-written
virtual groups, with a uniform barrier between waves. No dispatch-count readback
is introduced. Cooperative radix, scan and refit stages retain their original
shape. Capacity gates still read their small counters; six cache swaps and live
state copies are shared with the optimized Metal scene schedule.

`device_frame.py` shares C's transaction controller. `device_cuda.py` uses the
CUDA Driver API and one ordered stream. `device_ascend.py` uses AscendCL and a
Bisheng-generated launch wrapper. Growth migrates state by device copies. Dynamic
shared-memory demand is compiler-owned launch metadata. Normal body readback over
the persistent Python process is binary; JSON remains the small control protocol.

Compiled artifacts cache source, ABI, compiler/header identity and strict float
flags; corrupted binaries rebuild. Native/testbed sessions reuse a `device`
subdirectory of the configured kernel cache unless the device cache environment
variable overrides it. Scene state and reference answers are never cached there.
CUDA/NPU launch performance has not been measured; Python submission overhead,
256-group sizing and Ascend growth-copy synchronization may need device tuning.

Changes 0143 onward classify infrastructure, ABI and scheduling changes separately
from physics. `parity.json` conservatively watches these modules and tests;
`preparation.json` binds source-only results to source hashes. The immutable full
handoff builder keeps its stronger reconciliation requirements. This small
bring-up preparation is not a substitute for a qualified release bundle.
