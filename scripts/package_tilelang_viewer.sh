#!/bin/bash
# Strip the unpublished development backend from a backed-up, staged manifest.
# The source checkout retains TileLang; ordinary registry releases do not yet
# depend on the standalone experimental runtime.
set -euo pipefail
manifest="$1"
grep -q '^name = "nexus_viewer3d"$' "$manifest"
[ "$(grep -c '^tilelang = ' "$manifest")" -eq 1 ]
[ "$(grep -c '^nexus-tilelang-native = ' "$manifest")" -eq 1 ]
[ "$(grep -F -c '"nexus-tilelang-native?/metal"' "$manifest")" -eq 1 ]
staged_manifest="$(mktemp)"
trap 'rm -f "$staged_manifest"' EXIT
sed -e '/^tilelang = /d' \
    -e '/^nexus-tilelang-native = /d' \
    -e 's#, "nexus-tilelang-native?/metal"##' "$manifest" > "$staged_manifest"
if grep -q 'nexus-tilelang-native' "$staged_manifest"; then
    echo 'error: TileLang packaging transform needs review for this manifest' >&2
    exit 1
fi
mv "$staged_manifest" "$manifest"
