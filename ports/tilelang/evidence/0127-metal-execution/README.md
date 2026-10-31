# Metal execution performance and qualification

The 19-body comparison scene on Apple M4 Max showed a TileLang viewer step of
60.189 ms before these changes and 3.634 ms afterward. These are individual
last-step observations after 12 physics frames, with Rust comparison disabled.
They exclude preparation but include TileLang dispatch and publication readback.
Warm preparation remained 2.598 seconds. No Rust-GPU timing was measured here.

A separate controlled replay of the saved compiled Metal kernels measured:

| Schedule | Median wall time, ms | Median GPU time, ms | Shader dispatches |
| --- | ---: | ---: | ---: |
| Original, per-stage submission and wait | 42.27–42.78 | 3.01–3.20 | 397 |
| Original shaders, batched submission | 2.40–2.48 | 1.90–1.93 | 397 |
| Shorter sorting and fused colors, batched | 1.89–1.94 | 1.42–1.49 | 241 |

Each saved schedule ran in per-stage/batched/batched/per-stage order with fresh
initial storage, 12 frames per run, and the first two timing samples discarded.
These Swift replays use the exact already-compiled libraries and capacity gates.
They exclude viewer bookkeeping, runtime allocation checks and explicit body
publication; they are not production-runtime or large-scene throughput benchmarks.
The batched replay submits four command buffers per frame. The integrated Rust
adapter additionally flushes khal uploads once per nonempty stage range.
No physics sleeping was introduced.

## Implementation

- `bad4ca841`, `508b41bcd`: batch stages between capacity gates; use ordered
  compute passes and automatic tracked-resource dependencies, with explicit
  fences for untracked resources. Validate immutable manifests and ABI mappings
  once, retaining per-execution allocation and device checks.
- `5fa0c432e`: preserve stable canonical contact order while skipping high
  zero-byte collider-ID radix passes. Full subshape keys and inert sentinels
  remain. This removes 60 dispatches in this scene.
- `6ec67f5dd`: mirror fused solve/warmstart in the Rust-matching shader file.
  Small Metal scenes use one workgroup per batch with device-memory barriers
  between colors; wider scenes and other targets keep separate colors. This
  removes 96 dispatches in this scene.

One compute encoder with unconditional resource barriers was correct but slower
in an intermediate viewer trial (10.284 ms); ordered compute encoders are shipped.
The first prepared run after the final Python changes overlapped other compilation
and reported 8.827 ms; it is retained but excluded from the final timing comparison.

## Correctness evidence and limits

- All 161 initial buffers match between saved pre-change and optimized programs.
  Every buffer matches bit for bit after every one of 12 frames: 1,932 comparisons.
  Active contact manifolds grow from zero to ten. Within each replay, per-stage
  and batched submissions also produce identical buffers. See state-equivalence.json.
- Twenty native Metal tests cover tracked/untracked dependencies across workgroups,
  nested iteration counts, rejection before writes, capacity retries, checkpoint,
  reset and growth. Strict native Clippy and the example build passed.
- The bounded Metal smoke runs 64 focused tests and 11 explicit physics cases.
  New regressions compare shortened sorts against fresh Rust order and an
  independent integer scratch specification, plus fused/separate Metal colors
  against fresh Rust Coulomb/twist outputs. They include bias modes, multiple
  batches, partial/oversized workgroups, shared static negative-zero slots and
  an inserted empty color. The final receipt is under smoke/.
- A separate selected C color case passes the existing contract.
- An eight-frame, two-batch imported scene with two joints passes the complete
  generated-frame checks against fresh noalias-disabled Rust shader outputs.
  This particular fixture has no active contacts; it does not qualify a full
  production contact trajectory. Its maximum pose/trajectory error is 2.71e-20.
  The explicit selected exporter took 154.605 seconds, including compilation.
  Its report predates only the added issue metadata in runtime_scene.json; no
  shader, comparison tolerance or expected value changed afterward.
- The first selected-scene CLI attempt was stopped after 154 seconds because
  its exporter was executing unrelated fixtures before applying --case. The
  selected_scene.py harness uses the existing single-upload Rust export mode.
  It does not run the full matrix or every scene. Its copied native upload was
  freshly produced with the checked noalias-disabled compiler profile.

The full matrix, exhaustive regressions and reconciliation were not run.
The full baseline remains stale. Ascend, CUDA and large-scene performance remain
unqualified. Raw generated fixtures stay in the temporary diagnostic workspace;
retained reports, commands and fingerprints identify their exact scope.

## Open production-reference discrepancy

A separate final viewer run with --compare reports after 12 frames:

- maximum position difference: 0.025307816 m;
- RMS position difference: 0.0117006845 m;
- maximum linear-velocity difference: 2.000419 m/s;
- maximum angular difference: 0.004065453 rad;
- maximum angular-velocity difference: 0.24336359 rad/s.

This is the viewer's independently stepping Rust CPU reference, built with
mutable-reference noalias disabled. The timing comparison above does not use
that CPU run. The screenshot and log are viewer-final-compare.*.

This discrepancy is unresolved and is not accepted as floating-point tolerance.
The performance changes preserve the previous TileLang result in the controlled
contact replay, but that is not proof of production Rust parity. The blocking
runtime_scene contract gap testbed-contact-reference-divergence prevents a later
full-suite run from silently claiming release qualification. Investigate the first
divergent frame and identical-state inputs before changing any physics or tolerance.
