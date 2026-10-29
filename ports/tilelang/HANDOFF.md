# Offline Ascend engineering handoff

This package contains individually tested solver operations, not a complete
simulation backend. Its Ascend qualification is **source lowering only**. The
provisional profile is Ascend 950 SIMT. Confirm the actual device, CANN, driver,
and compiler combination before choosing an SDK/runtime adapter. No NPU access
is required to maintain the port, run C/Metal parity, or build this package.

## Create a package

Run all operation replays on available targets, lower Ascend source with the
pinned compiler build, reconcile the classified change, and commit locally.
The builder rejects dirty or unreconciled source and stale/incomplete evidence.
It does not push, contact Huawei, or start a paid VM.

```sh
python ports/tilelang/handoff.py build --output /tmp/nexus-ascend-handoff
python ports/tilelang/handoff.py verify /tmp/nexus-ascend-handoff
```

The bundle includes `nexus-source.tar` from the exact Git commit, compiler pins,
contracts, module/capability inventory, reports, and one directory per operation
and case. Every file has a SHA-256 receipt in `manifest.json`. Input buffers and
Rust expected outputs are little-endian raw words; `layouts/` gives the actual
Rust strides/offsets. `bindings.json` records generated argument order, launch
sizes, and sequential iteration count. Unbound storage buffers must still be
preserved; empty kernels may have no arguments. Integer sorting has three `uint32` buffers and exact comparison; solver operations
have four buffers and operation-specific floating-point masks. The six-float `Params` replay ABI
is distinct from the full production `RbdSimParams` struct.

## Device session stages

1. Verify the bundle before using it. Extract `nexus-source.tar` into a fresh
   source directory. The source archive contains the matching validator.
2. Run `python ports/tilelang/handoff.py probe --output environment.json` on the
   target host. It records `npu-smi info`, the platform, and `ASCEND_HOME_PATH` when
   available. A missing or failing device tool exits nonzero and is recorded as
   unavailable/failed. Probe success is not kernel execution evidence. Record
   the actual device model/ID, driver, CANN and vendor compiler versions in the
   returned `session.json`; do not infer compatibility from an environment path.
3. Build the generated `kernel.ascend` sources with a compatible vendor toolchain
   and the pinned TileLang headers. The selected SDK's host launcher is not yet
   implemented in this port. Preserve compiler command lines, versions, logs,
   resource reports and device binaries. A binary build is not an execution pass.
4. Implement/review the native adapter's allocation, byte sizes, binding order,
   device ownership and ordered dispatch contracts. Initialize every storage
   buffer from its `.bin`, run exactly the listed iterations in order, wait for
   completion, and save **all storage** buffers as `<Name>.actual.bin`. Never use an
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

Full-scene regression and timing stages remain unavailable because the production
scene backend has not been ported. Do not turn their absence into passing or
skipped qualification. Once implemented, run kernel replays first, then scene
checkpoints, long-run stability and timings with sleeping disabled. The retained
zero-distance numerical discrepancy is still release-blocking. Performance and
memory-model correctness require actual device work.
