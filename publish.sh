#! /bin/bash
#
# Publishes every nexus crate to crates.io with a single
# `cargo publish --workspace` invocation:
#
#   - nexus2d / nexus3d
#   - nexus_rbd2d / nexus_rbd3d
#   - nexus_rbd_shaders2d / nexus_rbd_shaders3d
#   - nexus_viewer2d / nexus_viewer3d
#
# Cargo computes the dependency order itself and waits for each crate to become
# available on the registry before publishing the ones that depend on it. The
# example crates and the python-binding crate are marked `publish = false`, so
# `--workspace` skips them.
#
# Why this script exists
# ----------------------
# Each 2d/3d crate pair shares a single source tree at the repo root,
# referenced from each manifest as `path = "../../<shared>/lib.rs"`:
#
#   - nexus2d / nexus3d                       -> src
#   - nexus_rbd2d / nexus_rbd3d               -> src_rbd
#   - nexus_rbd_shaders2d / nexus_rbd_shaders3d -> src_rbd_shaders
#   - nexus_viewer2d / nexus_viewer3d         -> src_viewer
#
# Those paths point outside the crate directory, which `cargo publish` refuses
# to package.
#
# To work around it *only during publishing*, this script temporarily, for each
# affected crate:
#   1. rewrites the `[lib] path` to a crate-local one (e.g. `src/lib.rs`), and
#   2. creates a symlink inside the crate pointing at the shared source tree.
#
# Cargo follows the symlink and bundles the real source into each `.crate`. A
# trap restores the manifests and removes the symlinks on exit (including on
# error or Ctrl-C), leaving the tree exactly as it was.
#
# Extra arguments are forwarded to `cargo publish`, e.g.:
#   ./publish.sh --dry-run
#   ./publish.sh --token "$CARGO_TOKEN"
#
# Requires cargo >= 1.90 (for `cargo publish --workspace`). The shader crates
# compile their SPIR-V in build.rs during the verification build, so the
# rust-gpu toolchain must be installed (`cargo gpu install`).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# `crate:shared_dir` pairs: each crate's `[lib] path` points at
# `../../<shared_dir>/lib.rs`.
CRATES=(
    nexus2d:src
    nexus3d:src
    nexus_rbd2d:src_rbd
    nexus_rbd3d:src_rbd
    nexus_rbd_shaders2d:src_rbd_shaders
    nexus_rbd_shaders3d:src_rbd_shaders
    nexus_viewer2d:src_viewer
    nexus_viewer3d:src_viewer
)

# Refuse to run on a dirty tree: the only diff during publishing must be our own
# temporary edits, so the restore at the end is guaranteed to be correct.
if [ -n "$(git status --porcelain)" ]; then
    echo "error: working tree is not clean. Commit or stash changes before publishing." >&2
    exit 1
fi

backup_dir="$(mktemp -d)"

cleanup() {
    for entry in "${CRATES[@]}"; do
        local_crate="${entry%%:*}"
        shared_dir="${entry#*:}"
        # Remove the symlink we created (only if it is in fact a symlink).
        link="crates/$local_crate/$shared_dir"
        [ -L "$link" ] && rm -f "$link"
        # Restore the original manifest.
        if [ -f "$backup_dir/$local_crate.Cargo.toml" ]; then
            cp "$backup_dir/$local_crate.Cargo.toml" "crates/$local_crate/Cargo.toml"
        fi
    done
    rm -rf "$backup_dir"
}
trap cleanup EXIT INT TERM

# Apply the temporary symlink layout: `shared_dir` lives at the repo root and is
# linked into each crate, while the manifest's `[lib] path` is made crate-local.
link_shared() {
    local crate="$1" shared_dir="$2"
    local manifest="crates/$crate/Cargo.toml"

    cp "$manifest" "$backup_dir/$crate.Cargo.toml"

    local tmp
    tmp="$(mktemp)"
    sed "s#path = \"\.\./\.\./$shared_dir/lib.rs\"#path = \"$shared_dir/lib.rs\"#" "$manifest" > "$tmp"
    mv "$tmp" "$manifest"

    ln -s "../../$shared_dir" "crates/$crate/$shared_dir"
}

for entry in "${CRATES[@]}"; do
    link_shared "${entry%%:*}" "${entry#*:}"
done

# Publish the whole workspace. `--allow-dirty` is required because our temporary
# edits make the tree dirty; the clean-tree check above keeps that safe.
cargo publish --workspace --allow-dirty "$@"
