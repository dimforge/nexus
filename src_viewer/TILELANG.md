# TileLang in the Nexus testbed

The optional `tilelang` feature adds generated C and (with `metal`) Metal solvers
in the existing 3D testbed. Ordinary rigid bodies, collision shapes and impulse
joints use the native Nexus scene conversion and the TileLang runtime. MPM and
articulated multibodies report an error and pause. There is no Ascend testbed
backend yet.

This is a source-checkout feature. The existing release-packaging script omits
`tilelang` from the published viewer manifest until the standalone runtime is
published; normal registry viewer releases keep their existing backends.

## Run

Use Python 3.13, the [pinned TileLang requirements](../ports/tilelang/requirements.txt),
a C++17 compiler, and Rust 1.98.1. Metal additionally needs Apple silicon and the
Xcode command-line tools. The port's POSIX compiler runner does not currently
support native Windows; C execution is available on the supported Linux/macOS
setup. See the [port setup](../ports/tilelang/README.md).

From the repository root:

```sh
python3.13 -m venv ports/tilelang/.venv
ports/tilelang/.venv/bin/python -m pip install -r ports/tilelang/requirements.txt
rustup toolchain install 1.98.1
python3 scripts/run_tilelang_testbed.py --python ports/tilelang/.venv/bin/python --backend c --compare
# On Apple silicon:
python3 scripts/run_tilelang_testbed.py --python ports/tilelang/.venv/bin/python --backend metal --compare
```

The launcher builds with the `tilelang` feature, disables mutable-reference
`noalias` for khal's CPU backend, and verifies the LLVM attributes before running.
Its verified profile is retained under `target/tilelang-testbed`. Use `--release`
for an optimized viewer/reference build. Do not override the compiler flags.

The default scene is **TileLang comparison**: a small box stack, balls, a swinging
body connected by an ordinary spherical joint, and a floor. Sleeping is disabled.
Additional testbed arguments follow `--`, for example:

```sh
python3 scripts/run_tilelang_testbed.py --python ports/tilelang/.venv/bin/python --backend metal --compare -- --headless --snapshot 62 /tmp/tilelang.png
```

For a direct Cargo invocation, set `NEXUS_TILELANG_PYTHON` to the absolute pinned
interpreter path and build/run with `RUSTUP_TOOLCHAIN=1.98.1`, `RUSTC_BOOTSTRAP=1`,
and `CARGO_ENCODED_RUSTFLAGS=-Zmutable-noalias=no`. Enable `tilelang` or
`tilelang,metal` on `nexus_examples_3d`. The executable options are
`--example tileLangComparison --tilelang c|metal --compare --run`.
`--reference-view` starts with the Rust poses displayed and also enables comparison.
`--list` prints the case-sensitive scene names accepted by `--example`.

## Compare visually

In **Settings → Physics Backend**, select TileLang C, TileLang Metal, Rust CPU,
or another compiled Rust backend. Changing backend restarts the current scene
with the current settings. Enable **Compare with Rust CPU** to run both solvers
from the same initial native input; this also restarts the scene.

Pause, single-step, and toggle **Display Rust reference** to inspect both results
at the same frame. The banner identifies which solver's body poses are displayed.
The panel reports RMS/max position error, max orientation error (radians), and
max linear/angular velocity error. These are observed differences, not pass/fail
tolerances; contact trajectories can diverge through floating-point branch choices.

In comparison mode, the Rust CPU simulation and TileLang advance independently.
Only explicit caller pose/velocity edits and solver settings are forwarded.
Normal stepping never overwrites TileLang with Rust results. Without comparison,
Rust physics is never advanced: CPU Nexus buffers carry scene inputs and the
actual TileLang output for the renderer. Body-attached visuals and sensors use
the displayed solver's poses. Contact/LBVH debug rendering is unavailable for
TileLang, since those buffers belong to the carrier/reference state.

Topology, material, shape or joint changes rebuild the TileLang scene from the
new native upload, reset its contact cache and start a new comparison segment.
The UI counts and labels these resets; numerical measurements across a reset
must not be interpreted as an uninterrupted independent trajectory. Native
Nexus full rebuilds can themselves reset bodies from the host Rapier world.

## Kernel reuse and errors

An unchanged restart reuses the loaded solver and resets every body, contact cache,
scratch buffer and frame counter. Switching examples or relaunching the application
creates fresh scene state and reuses compatible compiled kernels from
`target/tilelang-kernel-cache`. Only the most recent loaded runtime is retained.
A changed topology or solver configuration uses a fresh runtime; compatible kernels
can still hit the disk cache. Changed initial poses or velocities are never taken
from a prior example. Switching C/Metal uses distinct target entries.

The panel and terminal report preparation time, loaded-runtime reuse, and numbers
of lowerings/libraries reused or built. A warm launch still imports Python, checks
source/compiler hashes, constructs native inputs and allocates buffers. Metal also
creates device pipeline objects, so reuse does not make startup instantaneous.
Installing Apple's optional offline Metal compiler enables cached `.metallib`
libraries; without it the MSL lowering is cached and the device compiler still runs
for each new runtime. Loaded-runtime restarts need neither compiler path.

Set `NEXUS_TILELANG_KERNEL_CACHE` to an alternate directory, or to `off` to disable
persistent caching. Delete the cache directory while the testbed is closed to clear
it; there is currently no automatic disk eviction. Cache entries contain code,
never scene state or reference results. Source, compiler, target, ABI or specialization
changes invalidate the relevant entries. The source fingerprint is conservative:
port tooling, tests and contract changes also invalidate it. An already loaded
runtime keeps its code until replaced or the application exits.

Initial specialization can take several minutes. The window presents a compilation
banner before blocking; the artifact path and any errors are printed in the terminal. Capacity growth,
solver specialization changes and scene rebuilds can also compile and pause the
window. Large existing stress scenes remain large: start with the comparison
scene. The adapter reads native inputs and body outputs each frame and adds host
bookkeeping, so its timings are useful diagnostics, not kernel throughput results.

Runtime errors pause the interactive viewer with the error visible; there is no
fallback to Rust physics. Correct the configuration and restart or choose another
backend. A snapshot run fails on a TileLang error instead of saving a misleading
success image. Snapshots count rendered frames: frame 62 follows 60 simulation
steps in the small running scene (initial synchronization takes one frame).

`NEXUS_TILELANG_WORKDIR` chooses the parent directory for retained runtime
artifacts (default: the OS temporary directory's `nexus-tilelang-testbed`). Every
new runtime uses a fresh subdirectory; a reused restart keeps its existing one. `NEXUS_TILELANG_MAX_PAIRS` sets an application pair
capacity limit (default: at least 1,048,576); it is not a solver/device limit.
`NEXUS_TILELANG_PORT` overrides the compiler source directory. These interpreter,
source and artifact paths identify trusted executable code. Retained artifacts
can be removed after the testbed exits.

## Maintenance and validation

The adapter lives in `src_viewer/tilelang.rs`, with behavioral regressions in
`src_viewer/tilelang_tests.rs`. Shader implementation and the 1:1 Rust/TileLang
mapping remain in `ports/tilelang`. New demos must call `viewer.simulate`, rather
than calling `NexusPipeline::simulate` directly, to honor the backend selection.

Run the ordinary adapter unit tests with `cargo test -p nexus_viewer3d --features
tilelang,metal --lib tilelang::tests` under the pinned compiler environment above.
The C/Metal integration tests are opt-in because they require the compiler/device;
add `-- --include-ignored --test-threads=1` and set `NEXUS_TILELANG_PYTHON` to run them.
They exercise actual execution, independent divergence, live edits, solver setting
changes, explicit scene rebuilds, and the absence of Rust execution in TileLang-only
mode, cache hits across fresh compiler processes, A–B–A scene changes, exact restart
reset, and cached/uncached numerical equality. Root Cargo resolution and feature unification are distinct from the standalone
runtime build; retain the root lockfile and testbed evidence alongside port parity.

The opt-in offscreen UI transition regression is separate: use the same Cargo
command with the filter `tilelang_viewer_tests` and `-- --include-ignored
--test-threads=1`. It verifies that a backend/demo switch finishes the current
UI render pass. It uses an isolated process with process-scoped graphics lifetime,
as the real kiss3d application does.
