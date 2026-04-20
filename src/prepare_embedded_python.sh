#!/usr/bin/env bash
set -euo pipefail

VERSION="${VERSION:-3.14.4}"
PACKAGES=(pandas notebook matplotlib mplfinance pyarrow tqdm)

VERSION_BASE="${VERSION%.*}"
VERSION_NODOT="${VERSION_BASE//./}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_DIR="$SCRIPT_DIR/python"
SITE_PACKAGES="$PYTHON_DIR/Lib/site-packages"
PTH_FILE="$PYTHON_DIR/python${VERSION_NODOT}._pth"
URL="https://www.python.org/ftp/python/${VERSION}/python-${VERSION}-embed-amd64.zip"

echo "==> Preparing embedded Python ${VERSION} (base ${VERSION_BASE})"

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

echo "==> Installing packages into $SITE_PACKAGES"
mkdir -p "$SITE_PACKAGES"
pip install \
    --target="$SITE_PACKAGES" \
    --platform=win_amd64 \
    --python-version="$VERSION_BASE" \
    --implementation=cp \
    --only-binary=:all: \
    "${PACKAGES[@]}"

echo "==> Enabling 'import site' in $PTH_FILE"
if [[ ! -f "$PTH_FILE" ]]; then
    echo "ERROR: $PTH_FILE not found" >&2
    exit 1
fi
# Add Lib/site-packages path and uncomment `import site`
python3 - "$PTH_FILE" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
lines = p.read_text().splitlines()
out = []
has_site_packages = any(l.strip() == "Lib/site-packages" for l in lines)
for line in lines:
    if line.strip() == "#import site":
        out.append("import site")
    else:
        out.append(line)
if not has_site_packages:
    # Insert after the first "." line if present, else at top
    try:
        idx = out.index(".") + 1
    except ValueError:
        idx = 0
    out.insert(idx, "Lib/site-packages")
p.write_text("\n".join(out) + "\n")
PY

SHARE_SRC="$SITE_PACKAGES/share"
SHARE_DST="$PYTHON_DIR/share"
if [[ -d "$SHARE_SRC" ]]; then
    echo "==> Moving $SHARE_SRC -> $SHARE_DST"
    mv "$SHARE_SRC" "$SHARE_DST"
fi

echo "==> Done. Embedded Python ready at $PYTHON_DIR"
