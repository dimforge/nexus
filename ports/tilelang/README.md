# TileLang rigid-body parity pilot

This is an experimental port of **one prepared, conflict-free 3D Coulomb contact
color**. It executes the actual Nexus normal/friction solve against Rust reference
results, using the existing typed tile layout. It is not yet selectable as a Nexus
simulation backend. Contact generation, preparation, warmstart, integration,
joints, twist friction, 2D, and Ascend execution remain outside this milestone.

The feature ID is `rbd.contact.coulomb.solve3d`. Rust remains the behavioral
reference. `parity.json` maps its symbols and conservatively watches the whole
rigid-body source envelope. `contract.json` records scope, preconditions,
tolerances, required cases, and unresolved gaps. Watching a Rust file does not
mean all of its features have been ported.

## Run locally

Use Python 3.13, a C++17 compiler, and Rust 1.99.1. The reference toolchain and Cargo
lockfile are pinned separately from Nexus's general workspace. The Rust exporter
calls the current `TwoBodyConstraint::solve_constraint_gauss_seidel`; it contains
no copied solver arithmetic and does not require cargo-gpu or a GPU runtime.

From the repository root:

```sh
python3 -m venv ports/tilelang/.venv
ports/tilelang/.venv/bin/python -m pip install -r ports/tilelang/requirements.txt
ports/tilelang/.venv/bin/python ports/tilelang/replay.py --target c
ports/tilelang/.venv/bin/python -m unittest discover -s ports/tilelang/tests -v
python3 ports/tilelang/parity.py check --require c --evidence ports/tilelang/results/c/report.json
```

On Linux, install the pinned CPU-only PyTorch wheel before the requirements to
avoid downloading CUDA libraries:

```sh
ports/tilelang/.venv/bin/python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
```

On Apple silicon with the Xcode command-line tools, run the same suite through
native Metal:

```sh
ports/tilelang/.venv/bin/python ports/tilelang/replay.py --target metal
```

Python and TileLang compile the kernel. The Metal execution process uses generated
MSL directly through `native/`, a pinned Rust/khal adapter with no Python, PyTorch,
or TVM runtime. It accepts existing native buffers and flushes prior khal writes
before each synchronous execution sequence. The runner validates binding names,
dispatch dimensions and device identity; callers must uphold the shader memory
safety contract. It is not yet selectable in the full Nexus simulation pipeline.
The original Swift harness remains available as a historical independent runner.
The CPU runner similarly calls compiled generated C++ through a native function.
Native Windows replay is not implemented by these POSIX compiler commands.

`--output DIR` changes the result root. For the replay guard tests, set
`NEXUS_TILELANG_RESULTS=DIR` when using that option. Full binaries, generated source,
ABI bindings, actual outputs, and Rust expected outputs stay under that root.
The default `results/` directory is ignored by Git.

## What is tested

The 17 required cases cover 0/1/31/32/33/63/64/65/127/129 manifolds, 0–4 points per
manifold, static bodies on either side, locked mass axes, zero friction, singular
friction mass, restitution terms, penetrating/speculative contacts, all four
bias/friction flag combinations, and up to three sequential passes. Two seeds
exercise the multi-pass cases. Pose indices deliberately differ from velocity
slots. Sleeping is not involved in this isolated operation.

The Rust exporter derives offsets and strides with `size_of`/`offset_of` and emits
raw typed buffers. IDs remain integer bits. The kernel preserves point order:
all normal rows first, then all friction rows. Poses and parameters are inputs;
only active impulses and velocity components may change. Unchanged words,
including padding, inactive points, and sentinel bodies, are compared bitwise.
Floating outputs use the tolerances in the contract and must remain finite;
normal impulses must be nonnegative and enabled friction must remain inside its
Coulomb disk. Input validation rejects overlapping velocity slots before dispatch.

Arguments are bound in the order found in the lowered device IR. TileLang may
reorder them or remove all arguments from an empty-input kernel. Never infer the
native order from the Python function signature.

## Known numerical gap: cancellation at zero contact distance

Rust host SIMD and scalar/GPU quaternion transforms can differ by a few ULPs. If
`point.dist + dot(world_a - world_b, normal)` was constructed by cancellation to
be exactly zero, the result can acquire the opposite sign on another backend.
Rust's `distance <= 0` then selects a different soft-contact CFM branch. This can
produce a much larger impulse difference than the original rounding error.

The portable suite still checks exact zero, with zero local anchors so it does
not depend on cancellation of rotated points. The original adversarial case is
retained as a diagnostic:

```sh
ports/tilelang/.venv/bin/python ports/tilelang/replay.py --target c --boundary-stress --output ports/tilelang/results-boundary
```

A numerical failure in this diagnostic is expected on the initially tested ARM
host; it may differ by compiler/device. It is never accepted as parity evidence,
even if a particular configuration passes. The contract exposes this unresolved
gap, and `parity.py check --release` fails while it is present. Do not remove the
case, widen a global tolerance, or modify Rust silently to get a green report.
The `numerics.py` diagnostic now compares the Rust CPU oracle, the unchanged
production Rust Metal kernel, and TileLang Metal on identical input buffers.
It preserves all comparison failures in `results-numerics/report.json` and exits
nonzero when they occur. Two unbiased/no-friction cases are explicitly unavailable
in the production GPU kernel, which has only three modes; they remain required
in the portable suite. Same-GPU adversarial comparisons still fail, so a numerical
policy and scene-level validation remain necessary. Run:

```sh
ports/tilelang/.venv/bin/python ports/tilelang/numerics.py
```

CPU parity compilation uses `-ffp-contract=off`. Metal additionally needs
`#pragma clang fp contract(off)`; `fastMathEnabled=false` alone did not prevent the
observed contraction differences. A source-level explicit branch also avoids
invalid MSL emitted by TileLang 0.1.15 for a mutable `if_then_else` assignment.
These choices are included in the implementation fingerprint.

## Maintain parity

Start with:

```sh
python3 ports/tilelang/parity.py report --json
```

The report identifies modified, added, and deleted reference/implementation files,
affected feature IDs, explicit gaps, and separate backend states. File additions
inside the watched scope cannot silently become untracked capabilities. The impact
envelope is deliberately broad for this first pilot; narrow it only with evidence
that transitive helpers and newly added modules remain covered.

After a change, add a new structured record under `changes/`: classify its intent,
explain the disposition, identify regressions, and list differences. Classifications
are explicit because textual diffs cannot prove whether a change is a feature,
bug fix, or equivalent refactor. Reusing an old record for changed inputs is rejected.
Then regenerate results and reconcile:

```sh
python3 ports/tilelang/parity.py reconcile \
  --change ports/tilelang/changes/NEW-CHANGE.json \
  --evidence ports/tilelang/results/c/report.json \
  --evidence ports/tilelang/results/metal/report.json
python3 ports/tilelang/parity.py check
```

The Metal argument is optional when unavailable. Its previous report remains
visible as stale after relevant changes. Reconciliation requires the complete
current CPU execution suite, pinned package versions, a change record, and passing
change-tracking tests. It archives reports and updates `baseline.json`. Hashes
are change detectors, not proofs of equivalence. Updating baseline hashes alone
cannot refresh an old execution report. Changing the contract, tests, compiler
pins, reference lockfile, launch wrappers, or maintenance instructions invalidates
evidence too. Git preserves older baselines and change records.

The report records the base Git commit **and exact file hashes**, so uncommitted
implementation changes are distinguished. It also records the Rust/compiler/runtime
versions, target device, OS, generated source hashes, and binding order. Evidence
certifies the recorded environment and case set, not every hardware configuration.
The standalone reference lockfile defines its dependency versions; update it
explicitly when reconciling dependency changes in the main workspace.

The standard `check` gate applies to this experimental scope. `--require metal`
requires current Metal execution; `--require ascend` cannot pass without an actual
Ascend execution runner and evidence. `--release` additionally rejects unresolved
blocking gaps. The workflow in `.github/workflows/tilelang.yml` runs a fresh CPU
replay on Linux; it does not pretend to run an NPU. Its first remote CI run is not
part of local validation. Make the new CI job a required branch-protection check
when adopting the port; this change does not modify GitHub settings.

## Roadmap without Ascend access

This roadmap assumes no Ascend device, emulator, or paid NPU VM is available
throughout development. An early Ascend smoke test is **not a prerequisite** for
expanding the port. The objective is to complete an explicitly inventoried
rigid-body scope on available hardware, with reproducible evidence and an Ascend
handoff package. Ascend execution and performance remain unvalidated until tested
on a real device. The milestones below are planned work, not completed features.

Keep Rust as the behavioral reference and retain its public API, scene/state
ownership, and host scheduling. TileLang supplies generated native kernels. Keep
backend launch, synchronization, memory, and artifact-loading details separate
from shared physics, with explicit target capability requirements. Preserve the
current ABI initially; any later packing/layout adaptation needs its own mapping,
round-trip tests, and change record. Do not assume Metal-optimal layouts or launch
sizes will be efficient on an NPU.

### 1. Establish the checks we can run without an NPU

Resolve the existing zero-distance numerical discrepancy by comparing the Rust
CPU reference, the original Rust GPU kernel, and TileLang on the same available
GPU. Preserve the failing stress case. Investigate transforms, contraction, and
branch selection before deciding a scoped numerical policy; do not widen global
tolerances or alter the Rust reference to obtain a passing port. Independent
compiler/runtime work can continue while this gap remains release-blocking.

In parallel, establish an ordinary Linux CPU build of a pinned TileLang revision
with the Ascend code generator enabled, then lower the pilot and representative
portability probes to Ascend source. Upstream's [Ascend build configuration](https://github.com/tile-ai/tilelang/blob/main/src/ascend/CMakeLists.txt)
explicitly separates code generation from CANN and provides runtime stubs. The pinned source build now succeeds on macOS without CANN, and all 17 pilot
specializations generate Ascend source. `evidence/ascend.json` records this
source-lowering result; `evidence/compiler-build.json` records the build. A stub permits loading the compiler/runtime
library; it does not emulate an NPU or validate the generated device program.

The original published macOS wheel lacks the required Ascend backend. Use a Linux environment
or a separate source build rather than treating that wheel's missing backend as a
kernel failure. Pin the compiler commit and submodules as well as package versions;
replay C/Metal if adopting a different compiler. Do not silently compare reports
from different compiler builds as if they certified the same configuration.

The shared arithmetic has separate C, Metal/CUDA, and experimental Ascend SIMT
launch wrappers. The CUDA and Ascend wrappers remain unvalidated. On a build
containing those backends, source lowering can be attempted with:

```sh
ports/tilelang/.venv/bin/python ports/tilelang/replay.py --target cuda --compile-only
ports/tilelang/.venv/bin/python ports/tilelang/replay.py --target ascend --compile-only
```

Here `--compile-only` means **source lowering only**. Once established, repeat it
for relevant kernels and specializations and archive generated source and binding
metadata. Add vendor-toolkit binary compilation only if a compatible compiler is
accessible and usable without a device; neither toolkit availability nor such a
build is assumed. Neither check is execution evidence.

Use the upstream [Ascend 950 backend](https://github.com/tile-ai/tilelang/blob/main/tilelang/ascend/README.md)
as the provisional high-end capability profile. The actual device and CANN
version remain undecided. This does not promise support for other Ascend
architectures. Record required capabilities so a later target change exposes the
operations that need rework.

The build can be reproduced in a dedicated directory (requires Git, CMake, Ninja,
a C++ compiler, Python and network access for the pinned dependencies):

```sh
python3 ports/tilelang/compiler.py /tmp/nexus-ascend-compiler
```

Install the resulting wheel and pinned runtime dependencies in a separate venv.
Keep Z3 and TVM FFI installed inside that environment so native-library rpaths
resolve. The build helper does not change the pilot's execution environment.
`--record-only --source CHECKOUT` records existing wheels without rebuilding;
that path has been exercised locally. The full helper's clean Linux build has
not yet been run; the local build used the equivalent pinned source/build flags.

### 2. Exercise difficult operations before broad translation

Before translating large collision modules, extract small operations from the
Rust pipeline: indexed gather/scatter, bounded variable-length loops, reductions,
scan/compaction/sort, required atomics, and representative BVH traversal or convex
contact control flow. Test their semantics on C/Metal and attempt Ascend lowering.
Use real algorithm fragments rather than only dense matrix examples.

Keep dispatch ordering explicit. Do not rely on a fixed subgroup width, implicit
warp synchronization, or a global in-kernel barrier. Initially prefer separate
ordered dispatches and disjoint writes. When an operation needs atomics or shared
memory, specify ownership, bounds, ordering, and synchronization in its contract.
Host-side scheduling checks and mock runtime tests can test our assumptions; they
cannot establish the Ascend memory model or driver behavior.

If a probe fails, isolate the requirement, implement a portable alternative, or
record a target-specific gap and a small reproducer. This need not stop unrelated
features, but a required operation with no viable Ascend lowering path must stay
visible when deciding whether to expand the affected subsystem. Preserve an
escape hatch for a small native Ascend implementation with the same ABI/contract
if the TileLang backend cannot express it; avoid creating a second physics spec.

### 3. Integrate a complete stepping slice on available hardware

Add native generated-kernel loading to Rust/khal, initially using Metal. Keep
Python/TileLang in the build/export tooling. Add explicit buffer, binding,
specialization, dispatch, lifetime, and error-handling contracts at the backend
boundary, with a replaceable execution adapter for the future NPU runtime.

Port contact preparation, warmstart, Coulomb solving, and integration using the
same semantic-ID/replay pattern. Exercise the real color schedule, including
shared static bodies; the pilot's disjoint-slot restriction is not sufficient for
arbitrary production colors. Retain the existing collision pipeline for this
first hybrid stepping milestone and make that dependency visible in reports.

Validate free motion, impacts, stacks, friction, restitution, and contact
creation/removal. Compare intermediate buffers and short-run checkpoints against
Rust, then run longer stability/invariant regressions. Long-run trajectory
agreement needs a separately justified policy; the pilot's per-operation
thresholds do not automatically apply to full simulations. Benchmark unchanged
physics settings with sleeping disabled, and report timings by stage plus state
validation. Metal measurements describe Metal only.

Metal is the primary accessible GPU. CUDA replay and benchmarks are useful
additional evidence when a CUDA machine becomes available, but neither CUDA nor
Ascend access blocks this milestone.

### 4. Expand to the agreed rigid-body feature inventory

Inventory Rust capabilities before claiming a full port. Add stable IDs and
explicit dispositions for collision primitives, convex/mesh paths, broadphase,
contact persistence, scheduling, joints, other contact models, and any remaining
rigid-body features. Port the simple collision path first, then more complex
geometry and constraints. Track unsupported capabilities explicitly and fail
clearly if a scene requests them. A hybrid scene is not a complete TileLang port.

Each addition needs a live Rust oracle where possible, replay fixtures,
intermediate-state checks, integration regressions, and evidence for each target.
Reuse this infrastructure for MPM later; MPM is outside this rigid-body milestone.
Complete the chosen 3D scope on available hardware without waiting for an NPU.
Any deferred Rust capabilities, including 2D if deferred, remain inventory gaps.

### 5. Make the eventual Huawei session reproducible and short

Prepare a versioned handoff containing source/compiler pins, capability
requirements, generated sources, ABI/binding manifests, deterministic fixtures,
expected outputs and tolerances, known failures, and a machine-readable results
schema. Include a small environment probe followed by compile, kernel replay,
scene regression, and timing stages. Each stage must fail clearly and preserve
logs and a minimized failing input; environment failure must not become a passed
or silently skipped device test.

Prepare the native Ascend adapter against the selected SDK when available, and
keep it explicitly unvalidated until it runs. Runtime mocks test allocation,
binding, dispatch ordering, and failure handling, not device correctness. Keep
hardware-specific tuning and resource-limit decisions behind the adapter.

When Huawei or a paid VM becomes available, confirm the actual device/CANN
combination, build/load the generated artifacts, replay kernels first, then run
scenes and benchmarks. Return exact fingerprints, environment details, and outputs
so most diagnosis can continue offline. Cache reusable builds and support targeted
reruns to avoid consuming paid device time on unrelated setup or repeated cases.

### Evidence and AI maintenance throughout

Continue to require source-impact review and classified change records for Rust
changes, new features, bug fixes, optimizations, numerical changes, and intentional
differences. Extend the registry to support per-feature, per-target evidence as
coverage grows. This is planned metadata work: the existing checker currently
tracks the single pilot and must not be described as a complete capability matrix.

Keep these evidence levels distinct, bound to exact inputs and toolchains:

| Evidence | Meaning |
| --- | --- |
| C/Metal execution | The recorded cases passed on the recorded host/GPU |
| Ascend source lowered | The pinned TileLang compiler generated source |
| Ascend binary compiled | A recorded vendor toolkit built a device artifact |
| Ascend execution | The recorded cases ran on the recorded NPU |
| Ascend performance | Specified workloads were timed and validated on that NPU |

Missing evidence remains unvalidated; old evidence becomes stale after relevant
changes. Development acceptance uses current C/Metal evidence for the implemented
scope and, once established, source-generation checks. An unavailable NPU is not a
development blocker. Ascend release acceptance requires actual NPU execution and
workload validation. The existing numerical release block remains in force; this
roadmap does not relax the contract or advance any evidence level.

The point at which hardware is unavoidable is Ascend qualification and tuning.
Runtime correctness, synchronization, numerical behavior, resource limits, and
useful performance may still require changes at that point. The architecture,
contracts, and replay package should keep those changes identifiable and testable;
no amount of local testing guarantees a fixed amount of remaining NPU work.
