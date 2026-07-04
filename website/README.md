# nexus Website

This website is built using [Docusaurus](https://docusaurus.io/) and includes the interactive
WebAssembly builds of the nexus `all_examples2` and `all_examples3` demos.

## Prerequisites

- [Node.js](https://nodejs.org/) (v20 or later)
- [Rust](https://rustup.rs/) with the `wasm32-unknown-unknown` target
- [cargo-gpu](https://github.com/Rust-GPU/cargo-gpu) (needed to compile the Rust-GPU shaders, see the main README)
- [wasm-bindgen-cli](https://rustwasm.github.io/wasm-bindgen/) (optional: the build script
  auto-installs the version matching `Cargo.lock` into `target/` if the global one doesn't match)

Install the required Rust tooling:

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli
```

## Installation

```bash
npm install
```

## Building the Demos

The website hosts the two nexus demo binaries compiled to WebAssembly. Each is a full
application with its own built-in demo picker UI.

Build both demos:

```bash
npm run build:demos
```

Build a single demo:

```bash
npm run build:demo <demo_name>
# e.g., npm run build:demo all_examples3
```

The demos are built to `static/demos/` and will be included in the website. Note that the
demos require a WebGPU-enabled browser to run.

## Local Development

```bash
npm start
```

This starts a local development server at http://localhost:3000. Most changes are reflected live without restarting the server.

## Build for Production

Build everything (demos + website):

```bash
npm run build:all
```

Or build just the website (assumes demos are already built):

```bash
npm run build
```

The static site is generated in the `build` directory.

## Deployment

The website is deployed with `./publish.sh` (builds everything, then rsyncs `build/` to the
hosting server). It can also be deployed to any static hosting service.

## Project Structure

```
website/
├── src/
│   ├── pages/          # React pages (index, demos)
│   └── css/            # Custom styles
├── static/
│   ├── demos/          # Compiled WASM demos
│   └── img/            # Images and logos
├── scripts/
│   └── build-demos.sh  # Demo build script
└── docusaurus.config.ts
```
