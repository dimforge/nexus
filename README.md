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

- **nexus_rbd** - Rigid-body dynamics: colliders (boxes, balls, convex shapes, trimeshes, heightfields), joints (ball, fixed,
  prismatic, revolute), contact resolution.

## Prerequisites

### Install `cargo gpu`

Nexus uses [`cargo gpu`](https://github.com/Rust-GPU/cargo-gpu) to compile its Rust-GPU shaders to SPIR-V during
the build. **You must install it before building**, otherwise the shader compilation step will fail:

```sh
cargo install cargo-gpu --version 0.10.0-alpha.1
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
