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

## Compilation and errors

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
run uses a fresh subdirectory. `NEXUS_TILELANG_MAX_PAIRS` sets an application pair
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
mode. Root Cargo resolution and feature unification are distinct from the standalone
runtime build; retain the root lockfile and testbed evidence alongside port parity.

The opt-in offscreen UI transition regression is separate: use the same Cargo
command with the filter `tilelang_viewer_tests` and `-- --include-ignored
--test-threads=1`. It verifies that a backend/demo switch finishes the current
UI render pass. It uses an isolated process with process-scoped graphics lifetime,
as the real kiss3d application does.
