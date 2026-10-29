# Initial validation — 2026-10-29

This records the first pilot's local validation. Current source reconciliation and
backend evidence are determined by `parity.py report`, not by this historical note.
Exact source fingerprints and environments are in `evidence/c.json` and
`evidence/metal.json`.

- Reference: current Nexus Rust source based on `c94e14bd1a8a20408a1fb159364b57a5347c0198`,
  Rust 1.99.1, standalone checked-in Cargo lockfile.
- Compiler: TileLang 0.1.15, TVM FFI 0.1.12, Python 3.13, NumPy 2.2.6.
- Host: macOS ARM64; GPU: Apple M4 Max.
- CPU generated C++: 17/17 portable replay cases passed.
- Native Metal: 17/17 portable replay cases passed.
- Maximum absolute differences on both backends: `9.5367431640625e-7` in impulse
  outputs and `4.76837158203125e-7` in velocities. All untouched words compared
  exactly; normal/friction/finite-value invariants passed.
- Change-tracking regressions: 12/12 passed.
- Replay corruption/race regressions: 7/7 passed, using actual exported buffers.
- Rust exporter: `cargo fmt --check` and Clippy with `-D warnings` passed.
- Python: Ruff F checks passed. Ruff is only a development formatter/linter,
  not a runtime or parity dependency.

## Deliberate drift experiment

Temporarily changed the actual Rust normal-impulse update from `cfm_factor * ...`
to `0.5 * cfm_factor * ...` in the worktree's `solver_utils.rs`.

1. The parity check failed, identified the Rust source change, and marked both
   accepted CPU and Metal reports stale.
2. Rebuilding the live Rust oracle and replaying TileLang failed numerically on
   `n1-b1-f1-i1-s20261030`.
3. Restoring the exact original bytes restored source reconciliation and both
   accepted reports. The reference binary was rebuilt from the restored source.

No production Rust solver changes remain in this milestone.

## Explicit failures and unavailable validation

The retained `--boundary-stress` diagnostic failed on
`n127-b1-f1-i1-s20261156`: six impulse outputs differed beyond tolerance, with a
maximum difference of about `0.0157851`. This is the documented cancellation at
zero distance selecting a different CFM branch. The portable suite tests exact
zero without rotated-point cancellation. The original diagnostic remains
reproducible, tolerances were not widened, and `check --release` correctly fails.

Source-lowering attempts with this macOS wheel could not validate the other
backends:

- CUDA: missing `tilelang.cuda._ffi_api.AnnotateDeviceBoundTmaCopies`.
- Ascend: `Target ascend is not supported`.

Neither is an execution result. CUDA and Ascend remain unvalidated in the parity
report. The Linux CPU CI workflow has been added but has not been run remotely.
Production runtime integration, same-GPU Rust-kernel comparison, full-scene tests,
Ascend toolkit compilation/execution, and performance measurements remain future
milestones.

## Native Rust execution milestone

The default Metal replay now uses the native Rust/khal adapter. All 17 portable
cases pass through its existing-device/queue/buffer API. Strict native Clippy
checks and two binding/dispatch contract tests pass; the 19 Python regressions
also pass. Execution is synchronous and not yet wired into the simulation step.

`numerics.py` runs the original production Rust Metal solver on the same input
buffers and records all three pairwise comparisons. The retained adversarial
suite has 3 cases passing all comparisons and 12 failing at least one comparison.
Two additional cases cannot be expressed by the production kernel's three-mode
interface and are reported unavailable, not passed. TileLang/Rust CPU disagree in
3 cases; TileLang/Rust Metal and Rust Metal/Rust CPU each disagree in 12.
`evidence/numerics.json` preserves these diagnostic results and exact input hashes.
The numerical gap is unresolved; no tolerance was widened and no Rust physics
was changed. This diagnostic is not accepted portable or release evidence.

## Hardware-free Ascend source milestone

Built TileLang commit `8944978dd8761eb975a28f53d2a9be4a800be241` with
`USE_ASCEND=ON`, `USE_CUDA=OFF`, `USE_ROCM=OFF`, `USE_METAL=ON` and runtime
stubs on this macOS host, without CANN or an NPU. All 17 solver specializations
lower successfully to Ascend source. The report records native compiler-library
hashes and version `0.1.15+git8944978d`. C/Metal replays still use the original
0.1.15 wheel and both pass 17/17 cases. These are distinct compiler environments.

`compiler.lock.json` pins the source and principal submodules; `compiler.py`
provides a dedicated-environment build recipe. Its record-only path was tested
against the actual build. The complete clean helper/CI build is not yet tested.
No vendor device binary was compiled. Ascend runtime, numerical correctness and
performance remain unvalidated. Source lowering cannot satisfy `--require ascend`.

## Mirrored solver-operation milestone

The shader tree now mirrors the corresponding Rust modules, including the split
between body integration arithmetic and solver-side clamping. Four separately
contracted operations pass 32/32 live-reference replay cases on generated C++ and
native Rust Metal; all 32 specializations also lower to Ascend source in the
separate pinned source-build environment. The 26 parity and actual-buffer guard
regressions pass, as do Rust exporter fmt/strict Clippy and Python Ruff F checks.
No global tolerance changed. Full scene stepping, complete contact preparation,
and production coloring remain unimplemented; the solve numerical release block
and missing Ascend execution remain explicit.

## Inventory, handoff and native contract checks

All 108 Rust shader modules have explicit dispositions and mirrored target paths;
19 capability groups keep remaining scope visible. The native binding manifest
now records a schema and exact specialization byte sizes. Native launch and fixture
loading reject mismatched sizes. Integration fixtures preserve the full parameters
actually used by their Rust oracle. Ambiguous duplicate evidence overrides fail.

C and native Metal pass all 32 cases; the pinned Ascend compiler lowers all 32.
The 41 Python regressions and three native contract tests pass; both Rust tools
pass strict Clippy. An actual immutable 32-case handoff was built and verified
from commit a828f306b (before the native size-contract fix), covering 297 files.
Returned-output round trips, corruption/missing files, path escape, and unavailable
NPU probes are tested. They are offline validation tests, not Ascend execution.

## Irregular integer kernel and numerical isolation

Added `dynamics/canonical_order.py::gpu_stabilize_body_constraint_ids`, calling the
original Rust GPU entry function on the host for its oracle. All six cases match
bit-for-bit on C and native Metal and lower to Ascend source. They cover empty and
skewed rows, high unsigned IDs, duplicates, workgroup boundaries and a 513-entry
row. This is evidence for one real indexed-read/variable-loop/scatter operation;
atomics, collective scans and BVH traversal remain unvalidated.

The five implemented operations now pass 38/38 C and native Metal cases and all
38 Ascend source lowerings. All 45 Python regressions pass. Three native contract
tests and both Rust tools' strict Clippy checks pass.

The fresh three-way diagnostic retains 12 failing boundary-stress cases, with
three passing and two unavailable modes. The portable diagnostic has nine failing
cases, six passing and two unavailable modes. A separate zero-frame diagnostic
has 15 passing and two unavailable modes: removing frame translations isolates
the zero-distance cancellation sensitivity. Every original portable Rust input and
expected-output binary across all 38 operation cases is byte-for-byte unchanged.
No solver arithmetic or tolerance changed, and neither failing suite was replaced.
The numerical policy remains unresolved and release-blocking. CPU/Metal development
checks pass; both the release gate and required Ascend execution gate fail as expected.
