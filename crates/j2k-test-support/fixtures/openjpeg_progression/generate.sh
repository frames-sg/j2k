#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Regenerate the OpenJPEG position-major progression fixtures. Run manually and
# commit the output; CI only reads the committed files. README.md records the
# OpenJPEG version used and each file's SHA-256.
#
# Requires: OpenJPEG `opj_compress` on PATH.
#   Debian/Ubuntu: apt-get install libopenjp2-tools
#   macOS:         brew install openjpeg
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v opj_compress >/dev/null; then
    echo "error: opj_compress not on PATH; install OpenJPEG" >&2
    exit 1
fi

WIDTH=97
HEIGHT=61

# Interleaved RGB8 source, also the lossless oracle: source_97x61.rgb.
python3 - "$WIDTH" "$HEIGHT" <<'PY'
import sys

width, height = int(sys.argv[1]), int(sys.argv[2])
body = bytearray()
for y in range(height):
    for x in range(width):
        body.extend([
            (x * 7 + y * 3) & 0xFF,
            (x * 5 + y * 11 + (x * y) // 7) & 0xFF,
            ((x ^ y) * 9 + 31) & 0xFF,
        ])
open("source_97x61.rgb", "wb").write(body)
open("source_97x61.ppm", "wb").write(f"P6\n{width} {height}\n255\n".encode() + body)
PY

# Four resolutions (`-n` counts resolutions, so three decomposition levels),
# 8x8 code-blocks, and 32x32 precincts at the two highest resolutions with
# smaller ones below, so every resolution has several precincts and their
# sizes differ between resolutions. Three layers; the last is lossless.
common=(-n 4 -b 8,8 -c "[32,32],[32,32],[16,16]" -r 40,10,1)

for order in PCRL CPRL; do
    lower=$(echo "$order" | tr '[:upper:]' '[:lower:]')
    opj_compress -i source_97x61.ppm -o "${lower}_97x61.j2k" -p "$order" "${common[@]}"
    # 64x64 tiles, so both tiles are partial, a new tile-part at every
    # resolution change, and PLT packet lengths. With `-TP R` and 48x32
    # tiles, OpenJPEG 2.5.3 writes codestreams that both Kakadu and OpenJPEG
    # reject, so these use 64x64 tiles.
    opj_compress -i source_97x61.ppm -o "${lower}_97x61_tiles_plt.j2k" -p "$order" \
        "${common[@]}" -t 64,64 -TP R -PLT
done

rm source_97x61.ppm
opj_compress -h 2>&1 | grep -m1 -i 'openjp2\|version' || true
sha256sum ./*.j2k ./*.rgb 2>/dev/null || shasum -a 256 ./*.j2k ./*.rgb
