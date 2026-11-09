#!/usr/bin/env bash
# Precache Clean-FID Inception torchscript (~91MB) for offline FID.
# cleanfid only downloads if /tmp/inception-2015-12-05.pt is missing; an incomplete
# file blocks re-download and can hang eval. This script verifies exact size.
set -euo pipefail

FROST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPECTED=95607719
URL='https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
CACHE="$FROST_ROOT/.cache/inception-2015-12-05.pt"
TMP='/tmp/inception-2015-12-05.pt'
mkdir -p "$FROST_ROOT/.cache"

ok() {
  local p="$1"
  [[ -f "$p" ]] || return 1
  local sz
  sz="$(stat -c '%s' "$p" 2>/dev/null || echo 0)"
  [[ "$sz" -eq "$EXPECTED" ]]
}

if ! ok "$CACHE"; then
  echo "Downloading Inception -> $CACHE.partial"
  rm -f "$CACHE.partial"
  curl -L --retry 30 --retry-delay 5 -C - -o "$CACHE.partial" "$URL"
  sz="$(stat -c '%s' "$CACHE.partial")"
  if [[ "$sz" -ne "$EXPECTED" ]]; then
    echo "ERROR: size $sz != expected $EXPECTED" >&2
    exit 1
  fi
  mv "$CACHE.partial" "$CACHE"
fi

cp -f "$CACHE" "$TMP"
echo "OK $TMP ($(stat -c '%s' "$TMP") bytes)"
