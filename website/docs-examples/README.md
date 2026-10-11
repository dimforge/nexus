# User-guide examples

These files are the source of the code blocks in the user guide, following Rapier's
`DOCUSAURUS` marker / `<load>` convention. Do not paste separately maintained Rust
or Python examples into MDX.

- `rust2d` and `rust3d` are real Cargo workspace packages using the in-tree Nexus
  crates. `examples/getting_started2.rs` and `examples/getting_started3.rs` are complete viewer applications;
  `src/lib.rs` contains the smaller chapter examples with their surrounding setup.
- `tests/simulation.rs` runs representative scenes with the `cpu` feature.
- `python/getting_started.py` is a complete viewer application.
- `python/guide.py` contains the chapter examples. `python/check.py` executes them
  against the built extension, with optional headless GPU checks.

See [the website README](../README.md#checking-examples) for commands. The documentation
job requires compilation; extraction alone does not establish that examples compile.
