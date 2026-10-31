# TileLang preparation optimization measurements

The 19-body comparison example on Apple M4 Max prepared in 54.264 seconds with an
empty application cache and 2.594 seconds after relaunching the application with
that cache. The earlier retained measurements were 149.599 and 12.083 seconds,
respectively: approximately 2.76x faster cold and 4.66x faster warm. These are single
local development observations, not repeated controlled throughput benchmarks.
The OS and Apple compiler caches were not purged. This measures preparation,
not steady-state physics performance.

The implementation commits are `bd0295264` and `689160e49`. The earlier measurement
receipt is `../0119-kernel-cache/validation.json`. `validation.json` records current
source fingerprints, exact timings and verification scope. The new cold and warm
logs and images show two physics frames and zero displayed pose/velocity error
against the independently stepping Rust reference.

## Changes and remaining cost

- Lower independent kernels in four isolated processes by default, starting the
  largest IR first. `NEXUS_TILELANG_COMPILE_JOBS=1..16` controls concurrency.
- Cache the compiled program and schedule with its transitive Python builders,
  Rust-derived layout, case parameters, buffer sizes and compiler identity.
  A hit bypasses kernel construction and lowering. Dynamic builder dependencies
  bypass this cache. Uploaded state is always fresh.
- Share identical source/ABI artifacts while preserving every ordered dispatch,
  and build native libraries concurrently.
- Key compiler artifacts by executable inputs. Tests and documentation still
  invalidate the conservative parity evidence envelope without unnecessarily
  discarding unchanged compiled code.

The final cold run spent 6.394 seconds constructing kernels, 43.550 lowering them,
1.476 producing native libraries, and about 0.238 loading the native runtime.
`program_preparation` includes construction and lowering; these nested timings
must not be added together. The largest general-collision lowering alone took
41.385 seconds. Its isolated pass profile in `largest-lowering.json` identifies
repeated simplification passes as the main remaining expense. This change does
not modify compiler passes, shader algorithms, numerical flags or tolerances.
Eight workers improved the separate compiler trial only slightly (55.980 to
54.416 seconds), so four remains the default. See `worker-trials.json`.

## Validation

- The bounded Metal smoke passed in 83.334 seconds: 60 focused unit regressions and
  11 explicit physics cases. All five smoke reports pass the subset gate but
  remain `diagnostic_only` under the full parity gate.
- The focused C and Metal cache/restart tests passed in 171.63 seconds (174.007
  seconds including command overhead). They test fresh initial states, A-to-B-to-A
  scene changes, loaded-runtime reset and bitwise equality of cached and uncached
  eight-frame contact trajectories. An unchanged Metal restart took 0.003 seconds.
- All 397 ordered stages retain byte-identical generated shader sources, bindings
  and buffer maps against the prior compiler path. The complete frame metadata
  is identical after normalizing source filenames. All 161 initial buffers,
  layout and case bytes match. Only the number of stored source files changes,
  from 397 to 76. `source-equivalence.json` records the comparisons and hashes.
- `artifact-audit.json` verifies 2,976 artifacts from ten current runtime
  specializations, their native upload archives, and 406 smoke artifacts.
  Rust 1.98.1 used `-Zmutable-noalias=no`; the generated LLVM probes were verified.
- Strict viewer/native Clippy, Rust formatting, focused Python lint, the viewer
  test build and the example build passed.

The full parity matrix, exhaustive regressions and reconciliation were not run,
as explicitly required by the user. The full qualification baseline remains stale.
No Ascend or CUDA device qualification is inferred from these measurements.

## Reproducing the measurements

Use the testbed setup in `src_viewer/TILELANG.md`, then launch the comparison
example with a new `NEXUS_TILELANG_KERNEL_CACHE` directory and the configured
`NEXUS_TILELANG_PYTHON`. Relaunch with the same cache directory for the warm case.
`testbed-*-timing.json` records the exact binary invocation. Each run used
`--example tileLangComparison --tilelang metal --compare --run --headless
--snapshot 4 <image-path>` and stopped after the snapshot. The Rust logs give
preparation time separately from whole-process startup and rendering.

The focused cache command is retained in `cache-contact-tests-timing.json`; it
selected only `kernel_cache_and_restart`, with C and Metal tests sharing a
285-second external watchdog. The smoke suite has its own 290-second total
budget. Full matrix commands are deliberately absent from this procedure.
