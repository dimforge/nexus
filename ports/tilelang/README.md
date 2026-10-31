# TileLang 3D rigid-body port

This experimental port connects the 3D rigid-body solver to a Rust-owned runtime
that executes generated TileLang kernels on C or Metal. It includes Coulomb and
simplified twist friction, ordinary joints, analytic and general collision,
convex/mesh/polyline geometry, LBVH traversal, contact recycling, hub splitting,
coloring, scene reconstruction, live controls and continuation checkpoints.
[COMPLETION.md](COMPLETION.md) defines the agreed scope and final acceptance gates;
current qualification comes from `parity.py report`, not historical milestone notes.

The offline Ascend path generates kernels, binding manifests and source-bound
fixtures for a future CANN execution adapter. Ascend vendor compilation, device
execution, synchronization and performance remain unvalidated. Articulated
multibodies, MPM and 2D are explicitly deferred. CUDA execution and native Windows
replay are also unvalidated. Available C and Metal execution is not NPU emulation.

## Use the runtime

See [NATIVE_RUNTIME.md](NATIVE_RUNTIME.md) for `TileLangSimulation`, compiler setup,
backend selection and integration tests. Enable the `runtime` feature of
`ports/tilelang/native`; choose `TileLangBackend::C` or `TileLangBackend::Metal`.
The API accepts a typed `RbdState::read_scene_upload` archive and owns subsequent
frame execution. It is a separate public simulation API, rather than an enum
variant inside the existing `RbdPipeline`.

Metal uses the native Rust/khal executor after specialization. C uses a persistent
Python control process and compiled native C++ kernels. Both execute translated
physics kernels, retain device/runtime storage between frames and support ongoing
stepping without a trajectory allocation. The compiler is needed on first upload,
capacity growth and configuration changes. Explicit reads return typed body state.
[FRAME_RUNTIME.md](FRAME_RUNTIME.md) defines ordered dispatch, capacity gates,
persistent-region migration and retry behavior.

Cold-start uploads rebuild contact caches. Checkpoints preserve continuation state.
Insertion/removal uses scene reconstruction with explicit surviving-body mappings;
it does not promise the cache behavior of native incremental append/remove calls.
Live poses, velocities, gravity, solver parameters and friction-model changes are
supported with validation before publication. Unsupported articulated inputs fail
explicitly. Performance has not been qualified; ordered color launches preserve
dependencies but do not reproduce the native fused-launch optimization.

## Shader organization and behavioral reference

The Python shader tree mirrors `src_rbd_shaders`, retaining corresponding Rust
module paths and function names wherever practical. `inventory.json` lists every
Rust shader module, its mirrored path, mapped symbols and explicit disposition.
A counterpart file does not imply the entire Rust module or convenience API was
translated. Typed native constructors supply the ABI; accessors may be represented
by checked field indexing in callers. Namespace modules contain no separate kernel.

Compiler utilities, launch adapters, replay controllers and evidence tooling live
outside `shaders/`. `parity.json` maps stable feature IDs to reference symbols,
implementations, contracts, replay runners and target evidence. Each operation has
its own required cases. Contract `outside_scope` entries describe that operation
in isolation; other registered operations and runtime acceptance tests can cover
those capabilities. Watching a Rust file does not imply coverage of all its
features. Keep paired Rust bug fixes, port changes, new features, numerical policies
and intentional differences distinguishable in classified `changes/` records.

## Routine checks: under five minutes

Use Python 3.13, a C++17 compiler and the pinned requirements. The portable compiler
is TileLang 0.1.15. Rust fixture export uses 1.98.1 through `cpu_oracle.py`; the
standalone Metal replay adapter has its own pinned toolchain. Run from the root:

```sh
python3 -m venv ports/tilelang/.venv
ports/tilelang/.venv/bin/python -m pip install -r ports/tilelang/requirements.txt
# One-time dependency/oracle build (not part of the timed test pass):
python3 ports/tilelang/cpu_oracle.py build --locked \
  --bin bodies --bin colored --bin joints --bin collision --bin twist
# Routine test pass; use --target metal instead for Metal-specific changes:
ports/tilelang/.venv/bin/python ports/tilelang/smoke.py
```

The default smoke executes **11 representative Rust/TileLang comparisons** on one
backend, plus small cache, frame-control, runtime-service and runner regressions:

| Area | Selected cases |
| --- | --- |
| `free_step` | Empty dispatch, 65 bodies with a workgroup tail, three batches |
| `colored_program` | Multiple batches/colors and shared static bodies |
| `joint_program` | 12 joints across three batches |
| `analytic_collision` | Mixed shapes, rotated boxes, collision filters |
| `twist_solve` | Bias/friction response and a 65-row workgroup tail |

`smoke.py --list` prints exact case IDs. The selection uses the existing fixture
exporters and comparisons, preserving contract float tolerances, exact
IDs/counters and guard checks. Rust expected outputs are exported fresh with
mutable-reference noalias disabled. Shader lowering and target compilation run
fresh; it does not reuse old physics outputs. Selected-case reports are marked
`representative_smoke` and are rejected by full parity reconciliation, even if a
selected operation happens to contain every case.

A single 290-second wall-clock budget covers all steps, including incremental
builds, fixture export, compilation and tests. Timeout fails the run and kills its
owned subprocess groups. Setup/downloads and the initial Rust dependency build
are separate; unusually slow machines can still time out. `--budget-seconds` can
only shorten the limit. Each invocation gets a fresh directory under
`results-smoke/`, with step logs, operation reports and `smoke.json` timings.
`--list` shows the fixed selection without running anything.

The smoke covers integration, both contact friction models, colored dispatch,
ordinary joints and analytic collision. It does **not** qualify long scene
trajectories, exhaustive geometry/LBVH/hubs, the complete public runtime, or Ascend.
For a change in an omitted area, run a focused regression for that area as well.
A passing smoke never updates the full baseline or makes stale evidence current.

## Full qualification: explicit request only

Run the complete matrix, exhaustive Python regressions and full reconciliation
**only when the user explicitly requests them**. Do not run them automatically
while porting, fixing bugs, or finishing a task. The entry point requires `--full`:

```sh
# Only after an explicit request for the full run:
ports/tilelang/.venv/bin/python ports/tilelang/matrix.py --full --target c
ports/tilelang/.venv/bin/python ports/tilelang/matrix.py --full --target metal
ports/tilelang/.venv/bin/python -m unittest discover -s ports/tilelang/tests -v
```

Metal requires Apple silicon and the Xcode command-line tools. On Linux install
`torch==2.9.1` from `https://download.pytorch.org/whl/cpu` before the requirements
to avoid pulling CUDA libraries. The POSIX compiler commands do not implement a
native Windows runner. The base `replay.py --operation all` covers only its four
base operations; use `matrix.py` for the complete registry. `--list` prints the
chosen operations without execution. `--shard-count N --shard-index I` partitions
every registered operation exactly once for parallel local or CI runs.

`--output DIR` selects an ignored result root. Set `NEXUS_TILELANG_RESULTS=DIR`
for regression tests against that root. Reports retain generated source, bindings,
initial buffers, fresh Rust expected outputs, actual outputs and toolchain hashes.
Each run rebuilds its live native reference; cached golden outputs are insufficient.

The khal CPU backend requires mutable-reference `noalias` disabled. Always use
`cpu_oracle.py` for reference Cargo commands and native `runtime` feature builds:

```sh
python3 ports/tilelang/cpu_oracle.py clippy --locked --all-targets -- -D warnings
python3 ports/tilelang/cpu_oracle.py test --locked --bin cpu-dispatch-check --bin schedule-variants
python3 ports/tilelang/cpu_oracle.py test --locked --release --bin cpu-dispatch-check --bin schedule-variants
python3 ports/tilelang/cpu_oracle.py --native test --locked --features runtime
```

The wrapper enforces `RUSTC_BOOTSTRAP=1`, `-Zmutable-noalias=no` and the pinned
compiler, rejects custom flags/wrappers, and retains an LLVM attribute probe.
Native fused/tail CPU entry points additionally request cooperative coroutines:
they exchange storage writes at barriers even without a workgroup-memory argument.
The schedule regression compares separate, fused and tail dispatches bitwise for
both friction models, warmstart and cached output. This is a required CPU backend
configuration, not a proof of Rust aliasing soundness or Ascend synchronization.

The expensive public runtime tests are explicitly ignored by ordinary Cargo tests.
Run them with the compiler, existing workspace directory and fresh scene fixtures
as documented in [NATIVE_RUNTIME.md](NATIVE_RUNTIME.md). They cover capacity growth,
live controls, hubs, checkpoints, scene reconstruction, model switching and dense
120-frame production comparisons with sleeping disabled. Ordinary unit tests alone
do not satisfy the runtime acceptance gate.

## Numerical policy

Computed floats normally use `atol=2e-6, rtol=2e-5`; each hashed contract defines its
exact fields and exceptions. IDs, counts, copied words, inactive storage and guards
retain exact checks. Independent geometry, contact-response and physical checks
must reject matching corrupt target/reference results.

[BOUNDARY_NUMERICS.md](BOUNDARY_NUMERICS.md) defines the retained zero-distance CFM
stress policy, with native source extraction, independently recomputed outward
intervals and complete branch outcomes. [SCENE_NUMERICS.md](SCENE_NUMERICS.md)
records narrowly scoped scene/frame/field budgets and mandatory identical-state
restart tests. Neither policy authorizes global tolerance widening or case deletion.
C compilation disables floating contraction; Metal also disables contraction in
source. Genuine dispatch and reference bugs remain separate classified fixes.

## Maintain parity and detect drift

Read [AGENTS.md](AGENTS.md), then inspect:

```sh
python3 ports/tilelang/parity.py report --json
python3 ports/tilelang/inventory.py
```

The conservative impact envelope covers Rust helpers/layouts, new and removed
files, Python shaders, runtime/controller code, contracts, tests, compiler pins and
maintenance documentation. Changing any watched input invalidates evidence. Add a
new classified change record and run the smoke plus relevant focused regressions.
Keep the full evidence stale until the user requests a full run. After that
requested matrix and exhaustive regressions pass, reconcile:

```sh
python3 ports/tilelang/parity.py reconcile \
  --change ports/tilelang/changes/NEW-CHANGE.json \
  --evidence-dir ports/tilelang/results
python3 ports/tilelang/parity.py check --require c --require metal --release
```

Reconciliation requires all registered C execution reports and archives accepted
reports under `evidence/`. Additional evidence roots may supply Metal and Ascend
source reports. Old evidence remains stale; editing baseline hashes cannot refresh
it. A result certifies its recorded source, fixtures, toolchain and device only.
`--release` rejects unresolved blocking gaps. `--require ascend` requires actual
Ascend execution and cannot be satisfied by source lowering.

The Linux workflow `.github/workflows/tilelang.yml` runs the bounded C smoke on
push/PR, after dependency setup. Source/oracle checks, exhaustive replay shards,
corruption/return-protocol regressions and the public CPU runtime run only through
manual `workflow_dispatch` with `full_matrix` enabled. Remote CI has not run while this branch remains local. No workflow
or local completion step authorizes a Git push.

## Prepare the Ascend handoff

The provisional high-end target is the pinned Ascend 950 SIMT backend; the actual
device and CANN combination still need confirmation. `compiler.lock.json` pins
the source build and submodules. A separate compiler built with Ascend code
generation and runtime stubs can lower source without CANN or an NPU. The portable
macOS wheel alone lacks that backend. The stubs do not emulate hardware.

```sh
python3 ports/tilelang/compiler.py /tmp/nexus-ascend-compiler
# Only when explicitly requested:
/path/to/ascend-compiler/python ports/tilelang/matrix.py --full --target ascend --compile-only
```

The source-build configuration has been exercised locally. The complete helper's
clean Linux build remains unverified. Keep its environment separate from portable
C/Metal, retain exact compiler-library fingerprints, and never treat distinct
compiler builds as interchangeable evidence. Here `--compile-only` means source
lowering, not vendor device-binary compilation.

[HANDOFF.md](HANDOFF.md) documents immutable packaging, ordered and resident-frame
ABIs, target setup, returned outputs and offline comparison. Prepare and reconcile
all current reports, commit locally, build the bundle and verify it before delivery.
The eventual Huawei session must establish the actual toolkit/driver/device,
implement and review the CANN launcher, compile all sources, replay kernels and
resident scenes, and qualify synchronization and performance. Until then those
evidence levels remain unvalidated, without blocking the agreed offline port work.
