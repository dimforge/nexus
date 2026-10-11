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
npm ci
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

Or build the guide and website (interactive demos must be built separately to run them):

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

## User guides

The guide follows Rapier's documentation workflow: one shared MDX source, language
conditions, and examples extracted from code checked against the local engine.
See [WRITING_STYLE.md](WRITING_STYLE.md) for the analysis of Rapier's writing style.

- Edit `docs/user_guides/templates/*.mdx`, never the generated `rust/`, `python/`, or `all/` pages.
- Use `<rust>Rust-only text</rust>` and `<python>Python-only text</python>` inside a
  sentence or around a whole section. Shared explanations are written once.
- Rust 2D and 3D examples use synchronized Docusaurus tabs (`groupId="dim"`). Python
  only has a 3D API, so it has no dimension selector.
- Use `<load path='rust3d/src/lib.rs' marker='body' />` in a fenced code block. Paths
  are relative to `docs-examples/`. Python uses the same syntax with a `.py` source.
- Delimit a region in Rust with `// DOCUSAURUS: body start` and
  `// DOCUSAURUS: body stop`; use `#` comments in Python. Names are unique per file.
  Regions can nest. Extraction strips the marker lines and common indentation.
- Add chapters to the appropriate group in `guide-pages.json` to put them in all
  guide sidebars. Keep rigid-body dynamics and MPM in separate groups.
- Keep every chapter substantial and include an extracted code example for each
  language. Merge short reference-only topics into their related workflow.
- Link Rapier concepts and APIs to the corresponding Rapier user-guide section.
  Name methods, fields, and types explicitly in prose so readers know what to use.

`npm start` generates all three variants and watches templates and example sources.
`npm run build` regenerates the guide before every production build. Generation
fails on missing files/markers, empty or duplicate regions, or mismatched language
conditions. The edit links on generated pages point back to the shared templates.
Generated pages are ignored by Git.

Each language has its own sidebar and pagination. The sticky Rust/Python/All selector
above guide articles opens the corresponding chapter and preserves the reading
position within a shared section, following Rapier's user-guide. Its language
catalog is `src/userGuideVariants.ts`; All mode keeps shared text once and shows
both languages with labeled examples and color-coded conditional passages. Nested
conditions are resolved within their outer language. As with the single-language
pages, all snippets are extracted from checked sources.
The layout override uses the installed
Docusaurus 3.9.2 `DocItem/Layout` structure.

### Checking examples

```bash
npm run docs:generate      # extraction only; does not claim compilation succeeded
npm run docs:test          # generator regressions
npm run docs:check:rust    # compile all Rust examples against the workspace
npm run docs:test:cpu      # run small 2D/3D scenes with the CPU physics backend
npm run typecheck
npm run build             # MDX, route, link, and production build checks
```

The Rust example packages are workspace members so normal Cargo checks also see
API changes. The complete getting-started programs can be run from the repo root:

```bash
cargo run --release -p nexus-guide-2d --example getting_started2
cargo run --release -p nexus-guide-3d --example getting_started3
```

Python snippets are checked against the **real locally built extension**. From the
repository root, with an activated virtual environment and maturin installed:

```bash
maturin develop --release -m crates/nexus_python3d/Cargo.toml
cd website
npm run docs:check:python
python3 -m pip install numpy                # needed by array readbacks
python3 docs-examples/python/check.py --gpu  # also run headless backend operations
python3 docs-examples/python/getting_started.py
```

The normal Python check parses every example, imports the extension, and executes
scene-building snippets without opening a window. It fails if the extension is
missing; syntax-only validation is not a replacement. `--gpu` additionally executes
upload, stepping, readback, runtime insertion, motors, and particle chunk operations
using a headless viewer, which still needs a graphics backend.

CI checks Rust examples, generator behavior, TypeScript, and the production site in
the documentation job. The Python job checks snippets after building/installing its
wheel. The test job also runs the small documentation scenes with the CPU backend.
