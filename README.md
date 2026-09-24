<p align="center">
  <img src="assets/nexus-logo.jpg" height="200px">
</p>
<p align="center" style="font-size: xx-large">
        Cross-platform GPU multiphysics simulation
</p>
<p align="center">
    <a href="https://discord.gg/vt9DJSW">
        <img src="https://img.shields.io/discord/507548572338880513.svg?logo=discord&colorB=7289DA">
    </a>
</p>

**/!\ This library is still under heavy development and is still missing many features.**

The goal of **nexus** is to essentially be "**rapier** on the GPU". It aims to be a cross-platform GPU-accelerated
multiphysics engine, running compute shaders via WebGPU. Shaders are written in Rust using
[Rust-GPU](https://github.com/Rust-GPU/rust-gpu) and compiled to SPIR-V.

## Physics modules

Nexus is organized into independent physics modules, each available in 2D and 3D:

- **nexus_rbd** - Rigid-body dynamics: colliders (boxes, balls, capsules, cones, cylinders, convex shapes, polylines,
  trimeshes), joints (ball, fixed, prismatic, revolute), contact resolution.
- **nexus_mpm** - Material Point Method: a hybrid particle/grid-based method for simulating deformable objects, granular
  materials, fluids, etc. Supports one-way coupling with rigid-bodies (= rigid-bodies can push particles but particles
  cannot push rigid-bodies).

## Rigid-body friction (3D)

Choose the contact model globally with `RbdSimParams::friction_model`:

```rust
use nexus_rbd3d::dynamics::{FrictionModel, RbdSimParams};

let params = RbdSimParams {
    friction_model: FrictionModel::Simplified,
    ..RbdSimParams::default()
};
```

`Coulomb` (the default) solves sliding friction at each contact point.
`Simplified` follows Rapier's twist-friction model: per-point normal constraints,
one sliding-friction constraint at the manifold center, and one angular constraint
resisting twist about its normal. A single-point contact has no twist constraint.
The models approximate friction differently and can produce different motion.
Multibody contacts always use Coulomb, as in Rapier; 2D is unchanged.

Pass these parameters to `RbdState::from_rapier` (identically for every environment),
or call `NexusState::set_rbd_friction_model` before `finalize`. A live `RbdState`
can switch all environments with `set_friction_model(&backend, model)`.
Switching clears incompatible friction warmstarts and preserves normal warmstarts.

## Prerequisites

### Install `cargo gpu`

Nexus uses [`cargo gpu`](https://github.com/Rust-GPU/cargo-gpu) to compile its Rust-GPU shaders to SPIR-V during
the build. **You must install it before building**, otherwise the shader compilation step will fail:

```sh
cargo install cargo-gpu --version 0.10.0
cargo gpu install # Install the toolchain needed by cargo-gpu
```

For building running on the browser:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-server-runner
```

**WebGpu might not be enabled on your browser:**
- It is often already enabled by default on Windows and Macos major browsers.
- On Firefox, go to `about:config` then set `dom.webgpu.enabled` to `true`.
- On chromium, go to `chrome://flags` then set `Unsafe WebGPU Support` to `Enabled`. Keep in mind that, on Ubuntu, we observed WebGPU performances to be significantly worse on chromium compared to Firefox.
- Safari is currently not supported by Nexus.

## Running the examples

The example binaries launch a viewer window with all available demos. Use the `--release` flag for good performance,
as debug builds of GPU physics code will be very slow.

```sh
# Run natively
cargo run --release --bin all_examples3
cargo run --release --bin all_examples2
# Run on the browser
cargo run --release --bin all_examples3 --target wasm32-unknown-unknown
cargo run --release --bin all_examples2 --target wasm32-unknown-unknown
```

## Python bindings

The 3D engine and viewer are also available from Python as the `nexus3d` module
(PyO3 + maturin), mirroring the Rust API closely. See
[`crates/nexus_python3d/README.md`](crates/nexus_python3d/README.md) for building
instructions and a full set of example scripts (rigid bodies, joints, and
URDF/MJCF robots).

## License

MIT OR Apache-2.0
