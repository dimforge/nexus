#!/bin/bash

# Default values
OS=""
OUTPUT_DIR="$HOME/.cache/slang"
SLANG_VERSION=""
SLANG_TAG=""
ASSET_SUFFIX=""
SLANG_URL_BASE="https://github.com/shader-slang/slang/releases/download"

# Help message
usage() {
    echo "Usage: $0 [--os <linux|macos|macos-arm64|windows>] [--output-dir <path>] [--version <version>]"
    echo "  --os: Target OS (default: auto-detect from current platform)"
    echo "  --output-dir: Directory to extract Slang (default: ~/.cache/slang)"
    echo "  --version: Slang version (e.g., 2025.18.2, default: latest)"
    echo "Example: $0 --os linux --output-dir /tmp/slang"
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --os) OS="$2"; shift ;;
        --output-dir) export OUTPUT_DIR="$2"; shift ;;
        --version) export SLANG_VERSION="$2"; shift ;;
        *) usage ; exit 1 ;;
    esac
    shift
done

# Detect OS if not specified
if [[ -z "$OS" ]]; then
    case "$(uname -s)" in
        Linux*) OS="linux" ;;
        Darwin*)
            if [[ "$(uname -m)" == "arm64" ]]; then
                OS="macos-aarch64"
            else
                OS="macos"
            fi
            ;;
        CYGWIN*|MINGW*|MSYS*) OS="windows" ;;
        *) echo "Error: Unable to detect OS. Specify --os (linux, macos, macos-arm64, windows)"; exit 1 ;;
    esac
fi

# Determine asset suffix based on OS
case "$OS" in
    linux) ASSET_SUFFIX="linux-x86_64.zip" ;;
    macos) ASSET_SUFFIX="macos-x86_64.zip" ;;
    macos-aarch64) ASSET_SUFFIX="macos-aarch64.zip" ;;
    windows) ASSET_SUFFIX="windows-x86_64.zip" ;;
    *) echo "Error: Unsupported OS: $OS"; exit 1 ;;
esac

# Get Slang version if not specified
if [[ -z "$SLANG_VERSION" ]]; then
    export SLANG_TAG=$(curl -s https://api.github.com/repos/shader-slang/slang/releases/latest | grep '"tag_name"' | sed -E 's/.*"([^"]+)".*/\1/')
    export SLANG_VERSION=$(echo "$SLANG_TAG" | sed 's/v//')  # e.g., v2025.18.2 -> 2025.18.2
else
    export SLANG_TAG="v$SLANG_VERSION"
fi

if [[ -z "$SLANG_VERSION" ]]; then
    echo "Error: Could not determine Slang version"
    exit 1
fi

# Set up paths
SLANG_DIR="$OUTPUT_DIR/slang-v$SLANG_VERSION-$OS"
ZIP_URL="$SLANG_URL_BASE/$SLANG_TAG/slang-$SLANG_VERSION-$ASSET_SUFFIX"
TEMP_ZIP="/tmp/slang-$SLANG_VERSION.zip"

# Check if Slang is already extracted
if [[ -d "$SLANG_DIR" ]] && [[ -f "$SLANG_DIR/bin/slangc" || -f "$SLANG_DIR/bin/slangc.exe" ]]; then
    echo "Using existing Slang at $SLANG_DIR"
    echo "SLANG_DIR=$SLANG_DIR"
    exit 0
fi

# Download Slang release
echo "Downloading Slang v$SLANG_VERSION for $OS from $ZIP_URL..."
mkdir -p "$OUTPUT_DIR"
curl -L -o "$TEMP_ZIP" "$ZIP_URL" || { echo "Error: Download failed for $ZIP_URL"; exit 1; }

# Extract based on OS
echo "Extracting to $SLANG_DIR..."
if [[ "$OS" == "windows" ]]; then
    # Windows: Assume 7z is available (or adjust for PowerShell/Expand-Archive)
    7z x "$TEMP_ZIP" -o"$SLANG_DIR" -y > /dev/null || { echo "Error: Extraction failed"; rm -f "$TEMP_ZIP"; exit 1; }
else
    # Linux/macOS: Use unzip
    unzip -q "$TEMP_ZIP" -d "$SLANG_DIR" || { echo "Error: Extraction failed"; rm -f "$TEMP_ZIP"; exit 1; }
fi

# Clean up
rm -f "$TEMP_ZIP"

# Verify extraction
if [[ ! -f "$SLANG_DIR/bin/slangc" && ! -f "$SLANG_DIR/bin/slangc.exe" ]]; then
    echo "Error: Extraction incomplete, slangc not found in $SLANG_DIR/bin"
    exit 1
fi

echo "Slang v$SLANG_VERSION extracted to $SLANG_DIR"
echo "SLANG_DIR=$SLANG_DIR"

# For use in calling script
export SLANG_DIR