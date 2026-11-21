#!/usr/bin/env bash
# Fetch PennLINC's AtlasPack (the 4S atlases: Schaefer + subcortex, thalamus, hippocampus,
# amygdala and cerebellum) as a BIDS atlas dataset, for pipelines other than XCP-D, e.g.
# tractography. XCP-D (and QSIRecon) ship their own copy; don't also give them this one with
# --datasets, or they stop with "Multiple datasets contain the same atlas".
#
# usage: code/fetch_atlaspack.sh [DEST]      default DEST: data/AtlasPack (next to data/atlases)
#
# This is the exact archive XCP-D 26.3.0 builds into its container (Dockerfile.base), checked
# against its sha256. Contents: tpl-MNI152NLin6Asym (1 mm), tpl-MNI152NLin2009cAsym and
# tpl-fsLR (91k) versions of 4S156Parcels ... 4S1056Parcels. License: CC BY-ND.
set -euo pipefail

URL=https://upenn.box.com/shared/static/0gxdoe77tuih5p9ainn4utdqj3kcq7p1.tar.gz
SHA256=543db13adbef32a9d913f5aac9234e9a77a0a5963ae9ac57d3c8bd4ede65cb92

DEST=${1:-$(cd "$(dirname "$0")/../.." && pwd)/AtlasPack}
if [[ -e $DEST ]]; then echo "already exists: $DEST" >&2; exit 0; fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fsSL --retry 3 -o "$tmp/AtlasPack.tar.gz" "$URL"
if command -v sha256sum > /dev/null; then sum=$(sha256sum "$tmp/AtlasPack.tar.gz"); else sum=$(shasum -a 256 "$tmp/AtlasPack.tar.gz"); fi
[[ ${sum%% *} == "$SHA256" ]] || { echo "error: sha256 mismatch for $URL" >&2; exit 1; }
tar -xzf "$tmp/AtlasPack.tar.gz" -C "$tmp"
mv "$tmp/AtlasPack" "$DEST"
echo "AtlasPack in $DEST"
