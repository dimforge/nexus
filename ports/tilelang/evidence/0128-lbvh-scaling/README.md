# Balls LBVH scaling

The full Balls example contains 63,005 bodies (63,000 dynamic balls and five
walls), one batch, a 300,000-pair allocation, and 600,000 contact slots. This
saved workload uses twist friction, four substeps, one PGS iteration and 33
solver colors. Sleeping was not introduced and no scene settings were reduced.

On Apple M4 Max, the saved pre-change program took 2.64 seconds per frame in
controlled replay. Two runs of the optimized program took 32.38 and 34.32 ms
(medians after discarding the first two of six frames). The actual testbed then
reported 28.52 and 26.10 ms for its sixth physics frame, without Rust comparison,
capacity retries or reconstruction. The replay improvement is about 77–82 times.
These are early frames with 908 broad-phase pairs, not settled-pile throughput
or a timing comparison against Rust-GPU. Startup/preparation is excluded.

## Cause and implementation

The per-stage GPU profile attributes 2,572–2,577 ms to pair traversal alone.
The scene adapter launched one 64-thread workgroup for all 63,005 colliders.
The mirrored traversal also iterated `2 * collider_count` times per leaf even
once it had reached the escape sentinel. Rust breaks at that sentinel.

- `b33ac46b7` restores the bounded early exit and follows native scene launch
  geometry: 985 workgroups for leaves/pairs, 1,969 for escape nodes and 16 for
  the bounds reduction. The fixed-size domain scratch remains capped at 64
  workgroups per batch. Standalone fixtures keep explicit launch sizes.
- The pinned TileLang C/Metal emitters reject `T.loop_break`. A bounded `while`
  expresses the same stopping rule; unlike the Rust SPIR-V path, this generates
  Metal directly. Fresh C and Metal tests exercise this exact code. Publication
  barriers remain outside the divergent traversal.
- `ce0320d` checks each bounds partition exactly against input positions. The
  composed CPU oracle uses one group; production scene launches use the scaled
  partition. Both documented layouts are allowed, while corrupted partials,
  bounds, topology, guards and pair counts remain errors. This intentional
  scratch difference is recorded in the scene contracts and change record 0129.

No Rust shader arithmetic, physics formulas, floating-point tolerances, contact
margin or solver settings changed. Shader file/function correspondence remains.

## Measurements and state checks

`profile.swift` times two frames one dispatch at a time to identify bottlenecks.
`timing_replay.swift` executes six frames with normal batching at capacity gates.
The four timing runs use before/after/after/before order and fresh initial
storage each time. Both programs have 706 dispatches and 185 initial buffers,
984,619,440 bytes in total. Their cases and all initial buffer hashes match.
The probes use the exact saved compiled Metal libraries. The final testbed
specialization has identical inputs, frame/layout/case metadata and all 77
generated shader sources as the measured optimized specialization. Hashing and diagnostic
copies happen outside the timed interval; viewer publication is not included.

`replay.swift` additionally reconstructs warmstart state in canonical contact
order and verifies each previous-slot link against the preceding frame's map.
Body poses, velocities, canonical contacts, states, counts, colors, adjacency,
LBVH tree/domain/keys/order, and the collision-pair set match bit for bit over
all six frames. Atomic color-bucket allocation legally changes raw slot order,
even between two unchanged pre-change runs. After undoing that order, all live
tile fields, impulses and AoS warmstart caches match exactly. Remaining raw tile
differences are solely xyz coordinates of inactive twist contact arms; each
point index is checked against the live count, not accepted by a tolerance.
Run `python validate_state.py` here to recheck the retained compact records.

The first diagnostic normalization treated the AoS warmstart tail as SoA and
was corrected before accepting this evidence. It then identified the inactive
arm differences explicitly. No solver implementation changed to make this pass.

## Validation and limits

Focused fresh Rust execution covers empty/singleton trees, filters, overflow,
multiple batches, partial workgroups and a complete duplicate-Morton-key LBVH
program with scaled launches. An independent 4,097-collider, two-batch test
checks both domain partitions and rejects a one-bit corruption. The focused C
run passed in 68.789 seconds. Metal coverage is included in the bounded smoke.
The final Metal smoke passed 68 focused regressions and 11 representative
physics cases in 246.2 seconds (4 minutes 6 seconds), within its 290-second
budget. Its receipt and logs are retained under `smoke/`.

The full matrix and exhaustive suite were not run. The qualification baseline
remains stale. Ascend/CUDA device execution and the previously recorded
production-reference contact discrepancy remain unresolved. These performance
checks preserve prior TileLang behavior; they do not establish release parity.

Large-scene loading remains a separate bottleneck: the cached testbed restart
hit the program cache and all 77 native libraries but still took 76.26 seconds
to prepare/load. This change addresses simulation steps. The logs retain phase
measurements so future loading work can target that cost separately.
