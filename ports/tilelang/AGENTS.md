# Maintaining this port

1. Read `parity.py report --json`, `parity.json`, and `contract.json` before editing.
   Use the stable feature ID `rbd.contact.coulomb.solve3d` in change records.
2. Inspect the actual Rust symbols in the mapping, including helpers, layouts,
   parameters, Cargo changes, and upstream math dependencies. All watched Rust
   files conservatively invalidate this pilot; watching a file does not mean its
   features are implemented in TileLang.
3. Preserve normal-row order followed by friction-row order. Keep integer IDs
   integer. Derive offsets from Rust; never copy literal byte offsets into Python.
   One dispatch must have disjoint velocity slots. Fused colors and barriers are
   outside the contract.
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
