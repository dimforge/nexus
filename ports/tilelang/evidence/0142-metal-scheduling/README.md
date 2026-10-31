# Metal scheduling qualification and measurements

Changes 0134–0142. Implementation revision: `7e2a03abb93120e1965d8549af761c4cfbfdb657`.
Baseline: `a776b24d4`. See `../../PERFORMANCE_AUDIT.md` for conclusions and limits.
The source hashes and routine smoke input hashes pin the tested implementation;
this evidence commit adds documentation and receipts only.

## Performance receipts

- `summary.json` records the scene, solver settings, build profile and scope.
- `audit-baseline.log` and `scheduling-readback.log` contain actual debug testbed
  output after six physics frames. The adjacent timing receipts include commands,
  process elapsed time and successful exit status. Step time excludes preparation
  and rendering. This is one last-frame observation per variant, not a median.
- `final-timing-{0,1,2,3}-*.json` contains 30-frame stage replays in baseline,
  optimized, optimized, baseline order. Medians omit the first two frames.
  Diagnostic hashing was disabled. GPU time and host wall time are separate.
  These are initial evolving scenes, not settled dense-contact stacks.
- `final-input-validation.json` checks identical initial physical buffers;
  the only case changes are native scheduling and derived shape metadata.
  `BatchIds.contact_sort_collider_shift` is intentionally initialized from the
  collider count for packed sorting. No physics setting is changed.
- `baseline-state.json`, `final-state.json` and `final-state-validation.json`
  retain six-frame state hashes and exact comparison results. Atomic tile-slot
  allocation is canonicalized; the previous-index mapping is checked before
  normalization. Unused twist arms beyond each live contact length are excluded.
  Raw inactive scratch hashes may differ. `final-shader-validation.json` verifies
  that the subsequent readback-only change uses identical shaders/manifests.
- `replay-encoder.swift` is the probe source used with saved native variants.
  It defaults to separate tracked encoders, as in the actual runtime. The unused
  `NEXUS_PROBE_ONE_ENCODER=1` experiment is not enabled in these receipts or the
  implementation. The default diagnostic mode captures six frames; the timing
  harness overrides the frame count to 30 and adjusts the human-readable scope
  field in its saved JSON. `frames_per_run` always records the actual count.

Rebuild the probe with `swiftc -O replay-encoder.swift -o /tmp/nexus-replay`, then:

```sh
NEXUS_PROBE_STEPS=30 /tmp/nexus-replay /path/to/saved/variant-0 --timing-only
```

A saved variant requires its `frame.json`, `case.json`, `layout.json`,
`specialization.json`, initial `.bin` buffers and compiled libraries. Large raw
artifacts and binaries are intentionally not committed; scratch paths in receipts
are not an artifact-retention guarantee. `measure_viewer.py` is the exact bounded
local testbed measurement script, including its original paths and environment.
Adapt these paths when reproducing on another checkout. Do not run benchmarks
concurrently with other GPU work. No current Rust timing was taken.

## Correctness receipts

- `native-runtime-tests.log`: 24 runtime tests, 0.40 seconds. Indirect grids,
  transactional swaps, capacity gates, growth/restore/reset, conditional stages
  and aligned/unaligned body publication. The final source was built with
  `python3 ports/tilelang/cpu_oracle.py --native test --locked --lib --features runtime`.
- `native-runtime-clippy.log`: clean runtime clippy (successful, no output), using
  `cpu_oracle.py --native clippy --locked --lib --features runtime -- -D warnings`.
  Native Rust formatting and the feature-enabled debug testbed build also passed;
  the latter log is retained as `testbed-build.log`.
- `scene-scheduling-metal.log`: seven focused tests, 21.031 seconds. Fresh Rust
  colored/feedback/sort fixtures, dispatch producers, capped adjacency and live
  state copying. `scene-condition-composition.log` is the subsequent metadata test.
- `scene-scheduling-c.log`: two fresh Rust feedback/sort tests, 21.046 seconds.
- `bounded-exits-metal.log`: three focused GJK/EPA and LBVH search/control-flow
  tests, 61.751 seconds, with unchanged geometric and integer checks.
- `refit-metal.log`: six selected fresh Rust cases in one focused test, 8.222
  seconds on final sources. Covers both chunk/frontier kernels and uniform exits.
- `mesh-joint-eight-frames.log`: exact prior-versus-optimized schedule comparison
  of body trajectories and live counters, 205.762 seconds. It exports only the
  selected mesh/joint scene; this is not a full scene parity run.
- `smoke-metal.json`: final routine smoke, 238.766 seconds under the unchanged
  290-second budget. Its log contains 71 regressions; the five operation reports
  contain 11 representative physics cases. Pinned Rust 1.98.1 oracle verification
  includes mutable-reference noalias disabled.

No full matrix or reconciliation. Full baseline stale; change 0127 production
contact divergence remains blocking. CUDA/Ascend execution and the proposed
virtual-grid fallback remain unqualified. This evidence does not certify full
physics parity, hardware portability or Rust-equivalent performance.
