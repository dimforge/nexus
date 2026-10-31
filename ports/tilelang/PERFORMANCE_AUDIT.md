# Port scheduling audit — 2026-10-31

The audit found further translations that preserve outputs but lose Rust's
parallelism or early termination. Changes 0131 and 0132 fix the direct launch
and data-dependent loop errors below. The port still has significant scheduling
differences; this is not a performance-equivalence or full-parity qualification.
Machine-readable dispositions and source pointers are in PERFORMANCE_AUDIT.json.

## Fixed in this audit

| Path | Error | Correction |
| --- | --- | --- |
| Scene contact canonicalization | Key generation and permutation always used one workgroup, copied from fixture orchestration. | Scale the launch to contact capacity, retaining the live-count guard. Production Rust uses an indirect live-count grid; that remaining difference is tracked below. |
| Mesh/polyline candidate traversal | A node-count loop continued after reaching the traversal exit. | Retain the node-count safety bound and terminate at the exit. |
| Mesh point projection | Same inactive node-count tail. | Restore bounded traversal termination. |
| Contact reduction | Each run leader looped to the end of the entire PFM list after discovering its run length. With many singleton runs, this introduced quadratic loop overhead. Merge searches also continued after a successful merge. | Stop at the first different key and the first successful merge, preserving order. |
| Warmstart matching | Continued scanning after finding the first matching prior contact. | Terminate on that first match; preserve duplicate ordering and miss sentinel. |
| Coloring | Luby priority and topological conflict searches continued after a losing neighbor/conflict was found. | Restore the Rust termination conditions, preserving deterministic pending-color handling. |

The pinned TileLang emitters reject `loop_break`. These translations use bounded
`while` loops. No workgroup barriers occur inside the changed divergent loops.
Metal execution is checked; Ascend and CUDA execution are still unqualified.

## Open findings, in recommended order

1. **Capacity-sized dispatches.** `scene_kernels.py`, `colored_kernels.py`, and
   collision adapters launch for maximum storage; Rust's host consumes
   GPU-written live-count/per-color grids. In the saved Balls profile from the
   preceding experiment, 473 of 706 launches used 9,375 workgroups for a 600,000
   contact capacity. That profile predates this audit and is not a new timing
   result. Many lanes simply fail their count guard. Add native indirect grids
   with tests for zero counts, capacity gates, cache reload and backend limits.
2. **Full-buffer persistence copies.** The scene copies Links, Counts, IDs,
   Indices, Colors, Tiles and state halves every frame. Rust swaps current and
   previous buffer ownership/frame offsets. Adopt runtime binding swaps or
   equivalent ping-pong programs, preserving checkpoint/restore and resize
   semantics. Avoid treating copies as equivalent in performance reviews.
3. **Serial coloring feedback.** `gpu_coloring_requirements` scans all live
   contacts on one thread. This is a TileLang runtime extension for retry/color
   budgeting, not a faithful native shader scheduling choice. Replace it with
   a bounded parallel reduction, retaining invalid-color and pending-work
   feedback exactly.
4. **Unconditional canonical sorting passes.** The scene always requests
   `[subshape, collider_b, collider_a]`. Rust omits subshape for simple shapes and
   can pack collider IDs into one key. Use native shape metadata and the native
   reserved-sentinel bit bound. Existing digit-width optimization does not
   implement that pass-selection optimization. Preserve deterministic output
   ordering, including inert rows and composite subshapes.
5. **Fixed-size inactive loops.** GJK/EPA termination and some LBVH build/escape
   searches remain flag-predicated 32/64/100-iteration loops. They have bounded
   per-query overhead, unlike the scene-size-dependent errors fixed above.
   Profile and port their exits with geometric/degenerate-case regressions.
   Do not change Rust's own bounded split-search behavior incidentally.
6. **Body adjacency launch cap.** Rust caps at 65,535 groups and grid-strides over
   remaining bodies. `primitive_kernels.py` launches one group per body without
   that cap. It covers the workload on Metal but does not preserve the native
   backend-limit policy. Add capped grid-striding before claiming that policy
   across devices.

## Checked without identifying the same error

- World AABB and body kernels derive their launch size from body/collider count;
  the broad-phase wrapper overrides the pair-grid size for world bounds.
- Prefix scan retains its hierarchy; radix stages and hub reduction retain their
  cooperative groups and barriers.
- Per-body adjacency rank sorting is quadratic in adjacency degree in **both**
  implementations; it was not introduced by this port.
- LBVH refit uses a uniform fixed-depth cooperative schedule. A lane-local early
  exit would violate barrier participation. That is not a safe candidate for
  the scalar-loop rewrite applied above.
- `T.Kernel(1)` in the C-only launch adapters intentionally serializes the CPU
  test target. Scalar uniform/reset/publication kernels do constant-sized work.
  These are distinct from the serial contact scan called out above.

Scope: manual review of implemented 3D scene launch composition, hard-coded
workgroup sizes, data-dependent traversal/search exits, cooperative schedules,
canonical ordering and persistence. No claim of an exhaustive arithmetic,
concurrency, host-runtime or device-compiler audit. MPM, 2D and articulated
multibodies remain deferred. The open production-reference contact discrepancy
in change 0127 remains blocking; these performance fixes do not resolve it.

## Validation

- Parallel canonicalization: two Metal tests, including two fresh Rust sorting
  fixtures (65 contacts, duplicates/two-field order), passed in **15.993 s**.
- Early exits: five focused tests, with **11 selected Rust-reference cases**,
  passed on Metal in **65.224 s**. Covered composite candidates and overflow,
  normal/permissive reduction, mesh hits/misses, duplicate warmstart matches,
  Luby priority and deterministic/nondeterministic conflicts. Source-level
  termination checks detect inactive-tail regressions that output parity misses.
- Rust fixtures use the pinned noalias-disabled CPU oracle. Existing comparison
  tolerances and independent geometric/integer/guard checks are unchanged.
- Final routine Metal smoke: **69 regressions plus 11 representative physics
  cases passed in 233.828 s**, within the unchanged 290-second budget.
- Full parity matrix was not run and the baseline remains stale. No end-to-end
  speedup claim is made for this audit; new launch geometry can also trade
  under-parallelism for idle capacity-sized work.

Run the focused geometry regression explicitly when these paths change:

```sh
NEXUS_TILELANG_TEST_TARGET=metal python -m unittest discover \
  -s ports/tilelang/tests -p test_traversal_exits.py -v
```

The routine smoke retains its fixed 290-second budget. This geometry subset is
not appended to every smoke run. Accepted receipts live under
`evidence/0133-scheduling-audit/`.
