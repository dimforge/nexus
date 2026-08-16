#! /bin/bash
#
# Publishes every nexus crate to crates.io:
#
#   - nexus2d / nexus3d
#   - nexus_rbd2d / nexus_rbd3d
#   - nexus_rbd_shaders2d / nexus_rbd_shaders3d
#   - nexus_mpm2d / nexus_mpm3d
#   - nexus_mpm_shaders2d / nexus_mpm_shaders3d
#   - nexus_viewer2d / nexus_viewer3d
#
# Publishing happens in waves (see `WAVES` below); within a wave cargo computes
# the dependency order itself and waits for each crate to become available on
# the registry before publishing the ones that depend on it. The example crates
# and the python-binding crate are marked `publish = false` and are simply not
# listed.
#
# Why this script exists
# ----------------------
# Each 2d/3d crate pair shares a single source tree at the repo root,
# referenced from each manifest as `path = "../../<shared>/lib.rs"`:
#
#   - nexus2d / nexus3d                         -> src
#   - nexus_rbd2d / nexus_rbd3d                 -> src_rbd
#   - nexus_rbd_shaders2d / nexus_rbd_shaders3d -> src_rbd_shaders
#   - nexus_mpm2d / nexus_mpm3d                 -> src_mpm
#   - nexus_mpm_shaders2d / nexus_mpm_shaders3d -> src_mpm_shaders
#   - nexus_viewer2d / nexus_viewer3d           -> src_viewer
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
# Why the waves
# -------------
# When several crates are published at once, cargo verifies them against a
# temporary local registry (`target/package/tmp-registry`) holding the siblings
# it is about to upload. That overlay is only configured for the verification
# build itself: the *nested* cargo that a build script spawns does not see it.
#
# The host crates compile their shaders through `khal-builder`, which runs
# `cargo gpu build` on the packaged shader crate; that nested cargo resolves the
# shader crate's own dependencies against the real crates.io. So any nexus crate
# a shader crate depends on must already be live on crates.io before the crates
# that compile that shader can be verified, which the waves below guarantee.
#
# "Live" means served by the index, not merely uploaded: between waves the
# script polls index.crates.io itself, because `cargo publish` only warns (it
# does not fail) when a crate has not propagated within its own timeout.
#
# Re-running after a failure is safe: crates already published at this version
# are skipped, and the leftovers of the failed run are purged first (see
# `purge_publish_cache`).
#
# Extra arguments are forwarded to every `cargo publish` invocation, e.g.:
#   ./publish.sh --dry-run --no-verify
#   ./publish.sh --token "$CARGO_TOKEN"
#
# Note that a plain `--dry-run` only rehearses the first wave: nothing is
# actually uploaded, so the later waves fail to verify for the reason described
# above. Add `--no-verify` to rehearse the packaging of the whole set.
#
# Requires cargo >= 1.90 (for publishing several crates at once). The shader
# crates compile their SPIR-V in build.rs during the verification build, so the
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
    nexus_mpm2d:src_mpm
    nexus_mpm3d:src_mpm
    nexus_mpm_shaders2d:src_mpm_shaders
    nexus_mpm_shaders3d:src_mpm_shaders
    nexus_viewer2d:src_viewer
    nexus_viewer3d:src_viewer
)

# Publication waves, published one after the other. A crate may only share a
# wave with a shader crate it compiles; every *other* nexus crate that shader
# crate depends on must be live on crates.io by then (see "Why the waves").
# Today the only such edge is nexus_mpm_shaders* -> nexus_rbd_shaders*, so two
# waves are enough. Adding a shader crate that depends on another nexus shader
# crate means adding a wave.
WAVES=(
    "nexus_rbd_shaders2d nexus_rbd_shaders3d"
    "nexus_rbd2d nexus_rbd3d nexus_mpm_shaders2d nexus_mpm_shaders3d nexus_mpm2d nexus_mpm3d nexus2d nexus3d nexus_viewer2d nexus_viewer3d"
)

VERSION="$(sed -n '/^\[workspace.package\]/,/^\[/ s/^version = "\(.*\)"/\1/p' Cargo.toml)"

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

# True if `$1` is already on crates.io at `$VERSION`. Since the waves upload
# incrementally, a run that fails halfway leaves the earlier waves published;
# skipping them makes the script re-runnable. An unreachable index means "not
# published", so a network hiccup can never silently skip a crate.
already_published() {
    local crate="$1" prefix body
    case "${#crate}" in
        1 | 2) prefix="${#crate}" ;;
        3) prefix="3/${crate:0:1}" ;;
        *) prefix="${crate:0:2}/${crate:2:2}" ;;
    esac
    body="$(curl -sf "https://index.crates.io/$prefix/$crate" || true)"
    case "$body" in
        *"\"vers\":\"$VERSION\""*) return 0 ;;
        *) return 1 ;;
    esac
}

# Block until every crate named in `$@` is actually served by the index.
# `cargo publish` does wait for propagation on its own, but it gives up after a
# timeout with a warning rather than an error; the next wave then starts while
# crates.io still serves the previous version, and the nested cargo-gpu
# resolution fails exactly as if the wave order were wrong.
wait_for_index() {
    local deadline=$((SECONDS + 900)) crate pending

    while :; do
        pending=""
        for crate in "$@"; do
            already_published "$crate" || pending="$pending $crate"
        done

        if [ -z "$pending" ]; then
            # Our probe and cargo's own fetch may land on different CDN edges,
            # so let the slower ones catch up before building against them.
            sleep 20
            return
        fi

        if [ "$SECONDS" -ge "$deadline" ]; then
            echo "error: timed out waiting for crates.io to serve:$pending" >&2
            exit 1
        fi

        echo "waiting for crates.io to serve:$pending"
        sleep 10
    done
}

# Cargo unpacks the temporary publish registry into `$CARGO_HOME/registry` and
# keys those copies by name and version alone, so a previous run of the same
# version leaves sources behind that the next run silently reuses, lockfile
# checksums included. Once a wave has been uploaded those checksums no longer
# match what crates.io serves, and the nested cargo-gpu resolution dies with
# "checksum for <crate> changed between lock files". Dropping the stale copies
# makes every run start from freshly packaged sources. Only our own crates are
# removed, and only from the `file://` overlay registries (their directory name
# starts with `-`, since such a URL has no host); crates.io is left alone.
purge_publish_cache() {
    local registry_dir entry crate

    rm -rf target/package

    for registry_dir in "${CARGO_HOME:-$HOME/.cargo}"/registry/src/-* \
        "${CARGO_HOME:-$HOME/.cargo}"/registry/cache/-*; do
        [ -d "$registry_dir" ] || continue
        for entry in "${CRATES[@]}"; do
            crate="${entry%%:*}"
            rm -rf "$registry_dir/$crate-$VERSION" "$registry_dir/$crate-$VERSION.crate"
        done
    done
}

purge_publish_cache

# `--dry-run` uploads nothing, so there is never anything to wait for.
dry_run=0
for arg in "$@"; do
    if [ "$arg" = "--dry-run" ]; then
        dry_run=1
    fi
done

# `--allow-dirty` is required because our temporary edits make the tree dirty;
# the clean-tree check above keeps that safe.
for wave in "${WAVES[@]}"; do
    pkgs=()
    names=""
    for crate in $wave; do
        if already_published "$crate"; then
            echo "skipping $crate $VERSION: already on crates.io"
        else
            pkgs+=(-p "$crate")
            names="$names $crate"
        fi
    done

    if [ "${#pkgs[@]}" -eq 0 ]; then
        continue
    fi

    echo "publishing wave:$names"
    cargo publish "${pkgs[@]}" --allow-dirty "$@"

    if [ "$dry_run" -eq 0 ]; then
        wait_for_index $names
    fi
done
