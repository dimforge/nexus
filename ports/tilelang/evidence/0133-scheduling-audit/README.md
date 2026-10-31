# Scheduling audit evidence

Changes 0131–0133; see ../../PERFORMANCE_AUDIT.md for findings and limitations.

- Canonicalization: two focused Metal tests passed in 15.993 seconds, including
  two fresh 65-contact Rust fixtures. Result captured in the task tool output.
- `traversal-exits-metal.log`: five focused tests, 11 selected native-reference
  cases, passed on Metal in 65.224 seconds. `focused-source-hashes.json` pins the
  sources used for that run. The subsequent test-only cleanup changes a dict
  literal and an unused variable name; the shader files are unchanged.
- `smoke-metal.json`: final-source routine smoke passed in 233.828 seconds;
  69 regression tests and 11 physics cases, with the fixed 290-second budget.
  The five selected-operation reports and unit log are retained alongside it.
  The receipt records complete current input hashes and absolute scratch paths;
  raw temporary buffers are not an artifact-retention guarantee.
- Rust oracle: pinned 1.98.1 with mutable-reference noalias disabled. Per-operation
  reports retain the effective compiler verification.
- Inventory and audit reference/ID checks passed. Changed shader/test lint passed
  with existing SIM102 guarded-buffer-access patterns excluded. No shader
  guard was merged or arithmetic changed to satisfy a lint suggestion.

No full matrix, end-to-end performance claim, CUDA run or Ascend execution.
The full baseline remains stale; the production-reference contact discrepancy
from 0127 remains blocking. This evidence covers the listed scheduling fixes,
not complete port equivalence or a guarantee of bug-free code.
