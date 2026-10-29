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
MSL directly, with no Python, PyTorch, or TVM runtime. This small Swift runner is a
replay harness; production Rust/khal integration remains a separate milestone.
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
The next numerical milestone should compare both implementations on the same GPU
and define how branch-boundary equivalence is evaluated across devices.

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

## Ascend handoff and next steps

The shared arithmetic has separate C, Metal/CUDA, and experimental Ascend SIMT
launch wrappers. The latter two are **unvalidated** in this milestone. On a Linux
TileLang build containing those backends, source lowering can be attempted with:

```sh
ports/tilelang/.venv/bin/python ports/tilelang/replay.py --target cuda --compile-only
ports/tilelang/.venv/bin/python ports/tilelang/replay.py --target ascend --compile-only
```

`--compile-only` here means *source lowering only*. It does not invoke the vendor
toolkit, check device resources, or validate execution. The macOS wheel used for
local tests lacks the required Ascend backend. Do not interpret this as a kernel
failure on an Ascend device.

For Huawei, first confirm the device/CANN/toolchain combination and build the
pinned compiler with the required backend. Compile the generated source and add a
native execution adapter consuming `bindings.json` and the exported buffers.
Return results against the exact source fingerprint with device and CANN versions.
Only then extend reconciliation requirements to include Ascend execution.

Before expanding to the next operation:

1. Resolve the documented numerical boundary policy and compare the original
   Rust GPU kernel with TileLang on the same device.
2. Add native generated-kernel loading to the Rust runtime, retaining the existing
   typed layout and host scheduling. No full-scene serialization/repacking path is
   needed for this ABI.
3. Obtain an early Ascend compile/execution result for this operation.
4. Extend the same contract/replay/change-record pattern to preparation, warmstart,
   integration and complete scene stepping, then joints and other contact models.

Keep performance work separate from parity claims. This pilot does not measure
throughput, tune launch sizes, fuse colors, or demonstrate a complete rigid-body
simulation on Ascend.
