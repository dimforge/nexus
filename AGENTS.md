# TileLang parity pilot

The Rust implementation remains the behavioral reference. The experimental port
and its maintenance procedure live in `ports/tilelang/README.md` and
`ports/tilelang/AGENTS.md`.

For changes under `src_rbd`, `src_rbd_shaders`, or their crate manifests, start
with `python3 ports/tilelang/parity.py report`. The pilot intentionally watches
the entire rigid-body source envelope, so even a reference-only change requires
an explicit impact disposition before its parity baseline is advanced.

Preserve each feature contract when expanding coverage. CPU or Metal evidence
does not certify Ascend or a complete simulation. Keep unsupported
features and known numerical gaps visible in the contract and generated report.

Keep TileLang shader paths under `ports/tilelang/shaders` as close to 1-to-1
with `src_rbd_shaders` as practical, preserving corresponding Rust symbol names.
Keep replay/build/evidence tooling outside the mirrored shader tree.
