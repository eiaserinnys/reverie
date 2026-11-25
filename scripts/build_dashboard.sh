#!/bin/bash
# Build the Reverie dashboard and copy to the server package.
#
# This script:
# 1. Installs npm dependencies (if needed)
# 2. Builds the dashboard (npm run build)
# 3. Copies dist/ to src/reverie/dashboard_dist/
#
# Usage:
#   ./scripts/build_dashboard.sh
#   ./scripts/build_dashboard.sh --skip-install  # Skip npm install
#   ./scripts/build_dashboard.sh --clean         # Clean build

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DASHBOARD_DIR="$PROJECT_ROOT/dashboard"
DIST_DIR="$DASHBOARD_DIR/dist"
TARGET_DIR="$PROJECT_ROOT/src/reverie/dashboard_dist"

# Parse arguments
SKIP_INSTALL=false
CLEAN_BUILD=false

for arg in "$@"; do
    case $arg in
        --skip-install)
            SKIP_INSTALL=true
            ;;
        --clean)
            CLEAN_BUILD=true
            ;;
        --help|-h)
            echo "Usage: $0 [--skip-install] [--clean]"
            echo ""
            echo "Options:"
            echo "  --skip-install  Skip npm install step"
            echo "  --clean         Remove existing build before building"
            exit 0
            ;;
    esac
done

echo "=== Reverie Dashboard Build ==="
echo "Dashboard dir: $DASHBOARD_DIR"
echo "Target dir: $TARGET_DIR"
echo ""

# Check if dashboard directory exists
if [ ! -d "$DASHBOARD_DIR" ]; then
    echo "Error: Dashboard directory not found at $DASHBOARD_DIR"
    exit 1
fi

cd "$DASHBOARD_DIR"

# Clean build if requested
if [ "$CLEAN_BUILD" = true ]; then
    echo "Cleaning existing build..."
    rm -rf "$DIST_DIR"
    rm -rf "$TARGET_DIR"
    echo "Done."
    echo ""
fi

# Install dependencies
if [ "$SKIP_INSTALL" = false ]; then
    echo "Installing dependencies..."
    if [ -f "package-lock.json" ]; then
        npm ci
    else
        npm install
    fi
    echo "Done."
    echo ""
fi

# Build dashboard
echo "Building dashboard..."
npm run build
echo "Done."
echo ""

# Check if build succeeded
if [ ! -d "$DIST_DIR" ]; then
    echo "Error: Build failed - dist directory not found"
    exit 1
fi

# Copy to target directory
echo "Copying to $TARGET_DIR..."
rm -rf "$TARGET_DIR"
cp -r "$DIST_DIR" "$TARGET_DIR"
echo "Done."
echo ""

# Verify
echo "=== Build Complete ==="
echo "Files in $TARGET_DIR:"
ls -la "$TARGET_DIR"
echo ""
echo "Dashboard is ready to serve at /reverie"
