#!/usr/bin/env bash
set -euo pipefail

VERSION="${VERSION:-3.14.6}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_DIR="$SCRIPT_DIR/python"
URL="https://www.python.org/ftp/python/${VERSION}/python-${VERSION}-embed-amd64.zip"

echo "==> Preparing embedded Python ${VERSION}"

if [[ -d "$PYTHON_DIR" ]]; then
    echo "==> Removing existing $PYTHON_DIR"
    rm -rf "$PYTHON_DIR"
fi

echo "==> Downloading $URL"
TMP_ZIP="$(mktemp -t embed-python.XXXXXX.zip)"
trap 'rm -f "$TMP_ZIP"' EXIT
curl -fL --progress-bar -o "$TMP_ZIP" "$URL"

echo "==> Unzipping to $PYTHON_DIR"
mkdir -p "$PYTHON_DIR"
unzip -q "$TMP_ZIP" -d "$PYTHON_DIR"

echo "==> Done. Embedded Python ready at $PYTHON_DIR"
