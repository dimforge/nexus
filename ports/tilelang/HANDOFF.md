# Offline Ascend engineering handoff

This package contains solver-operation replays, connected scene and restart
fixtures, and resident-frame capacity-retry programs. The Rust-owned runtime runs
the connected 3D rigid-body path on C and Metal. An AscendCL execution adapter is
implemented but has no hardware execution evidence. Selected complete Coulomb
and twist scenes pass real Bisheng compilation with CANN 9.2.1 for the provisional
Ascend 950 SIMT (`dav-3510`) profile. This does not refresh the stale full-parity
baseline. Confirm the actual device, CANN, driver and compiler combination before
execution. No NPU access
is required to maintain the port, run C/Metal parity, or build this package.

For initial device testing without a new full matrix, use the bounded preparation
and commands in [DEVICE_BRINGUP.md](DEVICE_BRINGUP.md). The full package below
retains its reconciliation requirements and is only rebuilt after an explicitly
requested full qualification run.

## Create a package

Run all operation replays on available targets, lower Ascend source with the
pinned compiler build, reconcile the classified change, and commit locally.
The builder rejects dirty or unreconciled source and stale/incomplete evidence.
It does not push, contact Huawei, or start a paid VM.

```sh
python ports/tilelang/handoff.py build --output /tmp/nexus-ascend-handoff \
  --results ports/tilelang/results
python ports/tilelang/handoff.py verify /tmp/nexus-ascend-handoff
```

The bundle includes `nexus-source.tar` from the exact Git commit, compiler pins,
contracts, module/capability inventory, reports, and one directory per operation
and case. `cpu_oracles` binds each feature's report to its LLVM attribute probe;
the builder and verifier check the probe hash and require mutable-reference
`noalias` disabled. Expected outputs come from the pinned Rust 1.98.1 CPU oracle
through `cpu_oracle.py`. Historical bundles without this provenance need to be
rebuilt. This requirement does not apply to the generated Ascend code.
Every file has a SHA-256 receipt in `manifest.json`. Input buffers and
Rust expected outputs are little-endian raw words; `layouts/` gives the actual
Rust strides/offsets. `bindings.json` records generated argument order, launch
sizes, and sequential iteration count. Unbound storage buffers must still be
preserved; empty kernels may have no arguments. Each contract lists its own buffers
and comparison mode. Integer sorting uses exact comparisons. Contact/integration
fixtures use four buffers; the body lifecycle fixtures use eleven, including
separate body/solver state, mass properties and cleanup counters. Copied fields,
integer counters, padding and inactive words compare exactly; computed float
fields use the contract tolerances.

Bindings schema 2 denotes an **ordered program**. Allocate its physical
`storage_buffers` once. For every program iteration, execute every `stages` entry
in order: compile the named `source`, use that stage's `bindings` for argument order
and launch dimensions, and map local buffer names through `buffer_map`. Complete
writes before the next dependent dispatch. Preserve shared allocations between
stages and iterations; do not reinitialize, read back or replace intermediate
results with host calculations. Stage-local iteration counts apply inside each
program iteration. Every referenced source is required in the hash inventory;
`kernel.ascend` alone does not implement a multi-stage case. Prefix scans exercise
this with partial blocks, independent batches and multiple hierarchy levels.

Bindings schema 3 denotes **resident frames**. Its `variants` list names every
capacity specialization, including a schema-1 `frame.json`, initial buffers and
ordered program sources. Start from variant zero and keep storage resident across
frames. Execute stages through each named gate, synchronize and read only its
listed counters. If any demand exceeds that gate's capacity, stop before solver
commit, select a later fitting variant, initialize it from its own native upload,
copy the named persistent regions by identity, and retry the frame. Do not advance
the frame counter on a rejected attempt. Check every gate even when multiple
gates follow the same stage; pair, deferred-work, color and round limits have
different meanings. Missing capacity is an explicit error, never permission to
truncate work or continue with unresolved contacts.

Preserve native initialized defaults in new storage, not arbitrary zero fills.
Only copy the manifest's live persistent regions; preserve guards. Complete every
successful frame before beginning the next. Return all final storage buffers with
the sizes declared in the outer bindings and a `stats.json` array containing one
record per committed frame: `step` (one-based), `variant` (zero-based), `retries`
and `counts` by gate ID. The validator requires the fixture's exact growth frames,
no committed overflow and the final specialization. See FRAME_RUNTIME.md and
`frame_scene_validation.py` for the full contract. These resident fixtures test
retries and continuation; the public application API can additionally compile new
variants on demand, without a pre-exported ladder.

The six-float `Params` replay ABI is distinct from `RbdSimParams`. Body lifecycle
contracts name its fields: dt, linear/angular speed caps, and gravity xyz. The
corresponding full Rust uniforms are included for reproduction. Batch counts and
capacities are compile-time specializations recorded in each case. `free_step`
fuses independent per-body operations and substeps into one kernel; dispatch it
for the listed frame count. This composition is valid only without contacts,
joints, hubs or other cross-body dependencies. It is not the production schedule.

## Device session stages

1. Verify the bundle before using it. Extract `nexus-source.tar` into a fresh
   source directory. The source archive contains the matching validator.
2. Run `python ports/tilelang/handoff.py probe --output environment.json` on the
   target host. It records `npu-smi info`, the platform, and `ASCEND_HOME_PATH` when
   available. A missing or failing device tool exits nonzero and is recorded as
   unavailable/failed. Probe success is not kernel execution evidence. Record
   the actual device model/ID, driver, CANN and vendor compiler versions in the
   returned `session.json`; do not infer compatibility from an environment path.
3. Build every generated `.ascend` source referenced by bindings/variants with a compatible vendor toolchain
   and the pinned TileLang headers. The selected SDK's host launcher is not yet
   implemented in this port. Preserve compiler command lines, versions, logs,
   resource reports and device binaries. A binary build is not an execution pass.
4. Implement/review the native adapter's allocation, byte sizes, binding order,
   device ownership and ordered dispatch contracts. Initialize every storage
   buffer from its `.bin`, follow the applicable single-kernel, ordered-program or
   resident-frame schema above, wait for completion, and save **all storage**
   buffers as `<Name>.actual.bin` (plus resident `stats.json`). Never use an
   expected output as an input. Do not substitute the CPU interpreter on errors.
5. Return the environment, compiler and runtime logs, exact binary/source hashes,
   and outputs in the same `cases/<operation>/<case>/` hierarchy. Keep failed
   inputs intact. Case directories are independent, so targeted reruns can reuse
   compiled artifacts; merge them into a complete result directory before the
   final comparison. Cache compiled code keyed by source, compiler, target and
   binding hashes, not by case name alone.
6. Compare outputs offline using the matching extracted source and pinned NumPy:

   ```sh
   python ports/tilelang/handoff.py verify-results /path/to/bundle /path/to/returned \
     --output /path/to/comparison.json
   ```

   The command checks source identity, every case, exact untouched words,
   operation-specific masks, fixed float tolerances and invariants. It preserves
   individual failures and exits nonzero for missing or failing cases. These raw
   returned-buffer comparisons do not automatically advance parity evidence;
   execution provenance and the runtime adapter must also be reviewed. Keep
   source lowering, binary compilation, execution, and performance distinct.

`session.schema.json` describes the return header. Example (replace every value
with actual target information):

```json
{
  "schema": 1,
  "bundle_sha256": "the contents of manifest.sha256",
  "environment": {
    "device_model": "actual model",
    "device_id": "actual device ID",
    "cann_version": "actual version",
    "driver_version": "actual version",
    "compiler_version": "actual compiler and version"
  }
}
```

Run operation replays before connected native-upload scenes, identical-state
restarts and resident retry scenes. The package includes these scene regressions;
the target adapter must execute their complete schedules. Keep sleeping disabled.
The scoped zero-distance CFM and contact-sensitive scene policies are documented
in BOUNDARY_NUMERICS.md and SCENE_NUMERICS.md and enforced by hashed contracts;
do not widen tolerances or drop a failing stress case to obtain a device pass.

C/Metal return-protocol tests validate packaging and comparisons only. Even a
complete passing numerical comparison does not establish that outputs came from
an NPU. Preserve compiler/runtime provenance separately. Ascend synchronization,
resource limits, stability on broader workloads and performance require actual
device qualification. No performance claim or fixed amount of remaining NPU work
is implied by this offline package.
