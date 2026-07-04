# nexus3d — Python bindings

Python bindings for the 3D [`nexus`](../../README.md) GPU physics engine and its
viewer, built with [PyO3](https://pyo3.rs) + [maturin](https://www.maturin.rs).

Published on PyPI as [`dimforge-nexus3d`](https://pypi.org/project/dimforge-nexus3d/)
(the plain `nexus3d` name is taken by an unrelated project); the import name is
still `nexus3d`:

```bash
pip install dimforge-nexus3d
python -c "import nexus3d"
```

The API mirrors the Rust one closely, so the Rust examples translate almost
line-for-line into Python. Compare [`examples3d/boxes3.rs`](../examples3d/boxes3.rs)
with [`examples/boxes3.py`](examples/boxes3.py).

## Building

Requires the Rust toolchain used by the rest of the workspace (rust-gpu shader
compilation runs as part of the build) and `maturin`.

```bash
# From the repo root. Pick a GPU backend feature: metal (macOS), cuda, or cpu.
# webgpu is the default. `extension-module` is a default feature and must stay
# enabled (it is, unless you pass --no-default-features).
#
# Always build with --release: without it the Rust per-frame command encoding
# runs unoptimized and is ~10-40x slower (tens of ms/frame instead of a couple).
maturin develop --release -m crates/nexus_python3d/Cargo.toml --features metal

# or build a wheel:
maturin build --release -m crates/nexus_python3d/Cargo.toml --features metal -i python3
pip install target/wheels/dimforge_nexus3d-*.whl
```

> **Note:** the module is built against the stable ABI (`abi3`, CPython ≥ 3.9),
> so a single wheel works across all supported Python versions — you don't need
> to match the build interpreter to the run interpreter.

## Examples

[`examples/`](examples) contains Python ports of the 3D Rust demos in
[`crates/examples3d/`](../examples3d) (each `<name>.py` mirrors `<name>.rs`).


### Running them

First build the module into your active environment (see [Building](#building)),
then run any example directly — each has an `if __name__ == "__main__"` entry
point. Paths below assume you are at the repo root:

```bash
# After `maturin develop --release -m crates/nexus_python3d/Cargo.toml`
python crates/nexus_python3d/examples/boxes3.py
```

Each opens a viewer window. Because the viewer must own the main thread, run the
scripts directly (not from a REPL).

To play through every example back-to-back (the next launches when you close the
current window), use the runner script:

```bash
python crates/nexus_python3d/run_examples.py            # every example, in order
python crates/nexus_python3d/run_examples.py boxes3 balls3   # only these
python crates/nexus_python3d/run_examples.py --list      # list what would run
```

(`urdf3` / `mujoco_menagerie3` are included but need the asset env vars below,
or they'll exit with an error the runner reports before continuing.)

Two examples need an external asset, supplied via environment variables:

```bash
# URDF: NEXUS_URDF defaults to the Rust example's local path, so set your own.
NEXUS_URDF=/path/to/robot.urdf \
  python crates/nexus_python3d/examples/urdf3.py

# MJCF: clone google-deepmind/mujoco_menagerie and point at it.
# MUJOCO_MENAGERIE_SCENE picks the model by substring match (default: unitree_a1).
MUJOCO_MENAGERIE_DIR=/path/to/mujoco_menagerie \
  MUJOCO_MENAGERIE_SCENE=unitree_a1 \
  python crates/nexus_python3d/examples/mujoco_menagerie3.py
```

## Usage

The Rust and Python call sequences are intentionally identical: build bodies and
colliders with the rapier-style builders, insert them into a `NexusState`,
register their shapes with the `NexusViewer`, then drive the same render loop.

```python
from nexus3d import (
    NexusViewer, NexusPipeline, NexusState,
    RigidBodyBuilder, ColliderBuilder, GpuTimestamps, Vec3, Vec4, Pose,
)

viewer = NexusViewer()
pipeline = NexusPipeline()
pipeline.preload_pipelines(viewer)

state = NexusState()
body = RigidBodyBuilder.dynamic().translation(Vec3(0, 5, 0)).build()
collider = ColliderBuilder.cuboid(0.5, 0.5, 0.5).build()
handle = state.insert_rigid_body(body, collider)
viewer.insert_shape(handle, collider.shared_shape(), Pose.IDENTITY)

timestamps = GpuTimestamps(viewer, 2048)
viewer.add_directional_light(Vec3(1.0, -2.0, 3.0))
state.finalize(viewer)

while viewer.render_frame():
    if viewer.simulating():
        pipeline.simulate(viewer, state, timestamps)
    viewer.sync(state, timestamps)
```

### Differences from the Rust API

These are the only deliberate deviations; everything else matches the Rust names:

- **Async is hidden.** The Rust GPU calls (`NexusViewer::new`,
  `render_frame`, `sync`, `NexusState::finalize`, `NexusPipeline::simulate`)
  are `async`; the bindings block on them so Python code stays synchronous. The
  `while viewer.render_frame(): ...` loop is written exactly as in Rust.
- **The backend is passed via the viewer.** Where Rust calls
  `pipeline.simulate(viewer.backend(), &mut state, ...)`, Python passes the
  viewer itself: `pipeline.simulate(viewer, state, timestamps)`. Same for
  `state.finalize(viewer)`, etc. The binding pulls `backend()` out of the
  viewer internally.
- **`viewer.add_directional_light(dir)`** is a shortcut for
  `viewer.scene3d_mut().add_directional_light(dir)`.
- **Backend selection** uses `viewer.with_cpu()` / `viewer.with_metal()` /
  `viewer.with_cuda()` (the `with_metal`/`with_cuda` methods exist only when the
  crate is built with that feature).

### Threading

`NexusViewer()` opens a window and must be created on the main thread (an OS
windowing requirement), so run viewer scripts on the interpreter's main thread.

### Batched environments

Many independent simulations run in parallel as GPU batches. Allocate one with
`state.add_environment()` and use the `*_in(env, ...)` inserts:

```python
env = state.add_environment()
handle = state.insert_rigid_body_in(env, body, collider)
viewer.insert_shape_in(env, handle, collider.shared_shape(), Pose.IDENTITY)
```

### Loading robots (URDF / MJCF)

Loading a robot manipulates rapier's `PhysicsWorld` directly, so the bindings do
the load + insert in Rust and hand back what you need to render and actuate:

```python
# URDF: returns render shapes + link count.
opts = UrdfLoaderOptions(scale=40.0, make_roots_fixed=True,
                         create_colliders_from_visual_shapes=True)
robot = state.insert_urdf(path, opts, actuate_angx_motors=True)
for body, shape, pose in robot.render_shapes:
    viewer.insert_visual_shape(0, body, shape, pose)
# per-frame: state.set_multibody_motor_velocity(viewer, batch, link, JointAxis.AngX, v)

# MJCF: registers shapes/floor/camera/light with the viewer itself.
viewer.set_up_axis(Vec3.Z)                 # MJCF is Z-up
info = state.insert_mjcf(viewer, scene_path)
state.finalize(viewer)
state.set_rbd_gravity(viewer, Vec3(0, 0, -9.81))
```

The MJCF port doesn't reproduce the Rust example's runtime egui model picker
(closures over the UI context aren't bound); it loads a single scene chosen up
front.
