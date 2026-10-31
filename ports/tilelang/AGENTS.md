# Maintaining this port

1. Read `parity.py report --json`, `parity.json`, and `contract.json` before editing.
   Use the stable feature IDs from the registry in change records.
2. Inspect the actual Rust symbols in the mapping, including helpers, layouts,
   parameters, Cargo changes, and upstream math dependencies. All watched Rust
   files conservatively invalidate this pilot; watching a file does not mean its
   features are implemented in TileLang.
3. Preserve normal-row order followed by friction-row order. Keep integer IDs
   integer. Derive offsets from Rust; never copy literal byte offsets into Python.
   Solving or applying direct scatter warmstart requires disjoint writable velocity
   slots under those contracts. Colored dispatch permits shared read-only static
   slots only when both inverse mass and inverse inertia are zero; any read/write
   alias remains invalid. Cached gather writes one ordinary body per invocation;
   hub warmstart writes one split slot per invocation. Preparation only reads body
   inputs, so shared indices are allowed there. Metal small scenes use fused solve/warmstart with per-batch device-memory
   barriers; other targets, large scenes, cached fused and tail variants use
   ordered color launches. See NATIVE_RUNTIME.md and schedule-variants tests.
   Ascend barriers and runtime synchronization still require device qualification.
4. Add a regression for every bug fix. A bug discovered in Rust should be fixed
   there with the same regression, or recorded as an explicit unresolved
   difference. Do not change Rust arithmetic just to make the port tests pass.
5. Classify the change under `changes/` as port, feature, bugfix, refactor,
   optimization, numerical, abi, tooling, or test. Explain semantic effects,
   regressions, and intentional differences. Use a new record for a new change.
6. Routine validation is `python ports/tilelang/smoke.py` (C by default), with
   a shared 290-second execution budget. Use `--target metal` when the change is
   Metal-specific; choose one backend per routine pass. It executes 11 explicit
   representative cases against freshly exported Rust outputs and focused
   cache/runtime/tooling regressions. The budget includes rebuilds, export and
   compilation; prepare the toolchain/dependencies once as described in README.
   Add focused regressions for changed behavior. Run Rust fmt/clippy for Rust edits.
7. The user explicitly requires the full matrix to run ONLY on their request.
   Do not run `matrix.py --full`, the exhaustive Python suite, expensive ignored
   runtime tests, or full reconciliation automatically after edits, for completion,
   or because a stale baseline is reported. Do not invoke every runner separately
   to bypass this restriction. CI full qualification is manual-only as well.
8. Report the smoke scope and actual elapsed time. A passing smoke is partial
   validation, never complete parity. Keep `parity.py report` honest: old evidence
   stays stale, and the baseline remains unchanged until a requested full run.
   A timeout is a failed/incomplete smoke, never success or permission to increase
   the budget, omit cases, or start the full matrix. Diagnose with focused tests.

When the user explicitly requests full qualification, use `matrix.py --full` for
requested targets, the exhaustive regressions, then reconcile with the new change
record and finish with `parity.py check`. Reconciliation still requires complete
current execution reports. Merely editing hashes, loosening tolerances, or deleting
a failing case is not a fix. `contract.json` is hashed; case/tolerance changes need
an explicit explanation. Keep every zero-distance stress case in `solve_boundary`
and follow BOUNDARY_NUMERICS.md. Missing hardware remains unvalidated;
`--release` rejects blocking gaps. Source lowering is not vendor compilation or
execution evidence. The quick suite never relaxes these qualification gates.

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

Contact-sensitive scene policies are documented in `SCENE_NUMERICS.md` and the
hashed per-case contracts. Keep their registered identical-state restart cases
required. Exact IDs/counters/guards, all-frame physical bounds and independently
reconstructed contact response matrices must remain mandatory. An exception for
a named scene and field does not authorize a global tolerance change.

The CFM boundary policy requires current native source extraction, a bitwise check
of its unforced output, independently recomputed outward distance intervals, and
one complete branch outcome per manifold. Do not widen intervals or mix output
components to accept a mismatch. The policy does not cover multiple ambiguous
points, changing poses or more than three passes; extend its proof and tests first.
