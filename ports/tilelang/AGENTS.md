# Maintaining this port

1. Read `parity.py report --json`, `parity.json`, and `contract.json` before editing.
   Use the stable feature IDs from the registry in change records.
2. Inspect the actual Rust symbols in the mapping, including helpers, layouts,
   parameters, Cargo changes, and upstream math dependencies. All watched Rust
   files conservatively invalidate this pilot; watching a file does not mean its
   features are implemented in TileLang.
3. Preserve normal-row order followed by friction-row order. Keep integer IDs
   integer. Derive offsets from Rust; never copy literal byte offsets into Python.
   Solving or applying direct scatter warmstart requires disjoint velocity slots
   under those contracts. Cached gather writes one ordinary body per invocation;
   hub warmstart writes one split slot per invocation. Preparation only reads body
   inputs, so shared indices are allowed there. Fused colors and barriers remain
   outside the contracts.
4. Add a regression for every bug fix. A bug discovered in Rust should be fixed
   there with the same regression, or recorded as an explicit unresolved
   difference. Do not change Rust arithmetic just to make the port tests pass.
5. Classify the change under `changes/` as port, feature, bugfix, refactor,
   optimization, numerical, abi, tooling, or test. Explain semantic effects,
   regressions, and intentional differences. Use a new record for a new change.
6. Run the stdlib parity tests, Rust fmt/clippy, and `replay.py --target c`.
   The runner rebuilds the current Rust oracle; do not substitute cached golden
   output. Run `test_replay.py` against the generated fixtures. Run Metal when
   available. Use `--compile-only` for source-lowering checks of other targets;
   those results are not execution or vendor-toolkit compilation evidence.
7. Reconcile using the new change record and current execution reports. Merely
   editing hashes, loosening tolerances, or deleting a failing case is not a fix.
   `contract.json` is hashed; adding/removing cases and changing tolerances must
   have an explicit explanation. Keep the zero-distance cancellation stress case
   and its release block until the numerical policy is resolved.
8. Finish with `parity.py check`. Missing hardware results remain unvalidated;
   old results remain stale after relevant changes. The default check accepts
   the documented experimental pilot. `--release` also rejects blocking gaps.
   Report actual scope and limitations, including the lack of production runtime
   integration and Ascend execution.

`reference/Cargo.lock` and compiler requirements are tracked deliberately. A
compiler/runtime upgrade is a change requiring revalidation. Reports identify the
exact source hashes and tested environment; they do not certify other devices or
compiler configurations. Generated artifacts and fixtures belong in `results/`
(ignored); accepted reports are copied into `evidence/` by reconciliation.

Extend the registry and contracts as new operations are ported. Do not replace the
conservative impact envelope with narrower dependencies without tests showing
that helper, layout, configuration, and newly added source changes still trigger
reconciliation. Metadata plus tests is an audit aid, not a proof of equivalence or
an adversarial security boundary against an agent editing its own checks.

Mirror `src_rbd_shaders` paths under `shaders/`, with corresponding Rust function
names. Do not combine unrelated Rust modules into a monolithic port. Shared
Python compiler/buffer helpers belong outside that tree and are not physics ports.
Every new operation needs its own contract and per-target evidence; the global
check requires evidence for all registered operations.

`inventory.json` gives every Rust shader module an explicit disposition and a
mirrored target path. Run `inventory.py` after adding or moving either language's
files. Update the exact module, implemented symbols, operation IDs and remaining
scope; do not mark an entire module or capability complete merely because one
function passed. `parity.py check` enforces inventory consistency.

The Rust CPU oracle requires mutable-reference noalias disabled. Use
`cpu_oracle.py` for Cargo build/run/test/clippy commands; it selects the pinned
1.98.1 compiler and enforces `RUSTC_BOOTSTRAP=1` plus `-Zmutable-noalias=no`.
Do not remove the build guard or LLVM attribute probe to accommodate a compiler
upgrade. The probe and effective configuration are required evidence, including
for source-only lowering because its expected outputs still come from Rust CPU.
The native Metal adapter uses a separate toolchain and does not need this flag.

Hub assignment has nondeterministic atomic allocation order. Its contract checks
legal reservations, exact requested totals, capacity, membership and ownership;
do not require identical slot/list order across devices or weaken integer checks
elsewhere. Hub averaging preserves the Rust per-lane accumulation and six-step
64-lane tree. Keep the cancellation fixture and both workgroup barriers between
rounds. Serial CPU phase execution is not proof of GPU synchronization; run Metal
and retain Ascend hardware synchronization as unvalidated until tested.
