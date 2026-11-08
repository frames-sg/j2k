#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Regenerate the odd-origin fixtures and their reduced-resolution references.
# Run manually and commit the output; CI only reads the committed files.
# README.md records the tool versions used and each file's SHA-256.
#
# Requires: OpenJPEG `opj_compress` and `opj_decompress`, and OpenJPH
# `ojph_compress`, and `trash` for scratch cleanup, on PATH.
#   Debian/Ubuntu: apt-get install libopenjp2-tools libopenjph-tools
#   macOS:         brew install openjpeg openjph
set -euo pipefail

cd "$(dirname "$0")"

for tool in opj_compress opj_decompress ojph_compress trash; do
    if ! command -v "$tool" >/dev/null; then
        echo "error: required tool $tool is not on PATH" >&2
        exit 1
    fi
done

WIDTH=23
HEIGHT=17

# Interleaved RGB8 source, also the lossless oracle: source_23x17.rgb.
python3 - "$WIDTH" "$HEIGHT" <<'PY'
import sys

width, height = int(sys.argv[1]), int(sys.argv[2])
body = bytearray()
for y in range(height):
    for x in range(width):
        body.extend([
            (x * 29 + y * 53 + (x * y) // 3) & 0xFF,
            (x * 11 + y * 7) & 0xFF,
            ((x ^ y) * 13 + 41) & 0xFF,
        ])
open("source_23x17.rgb", "wb").write(body)
open("source_23x17.ppm", "wb").write(f"P6\n{width} {height}\n255\n".encode() + body)
PY

# Image origin (3, 5) on the reference grid: odd at every resolution, so each
# reduced image spans [ceil(origin / s), ceil(extent / s)) and is narrower
# than ceil(size / s). Three decomposition levels, one tile, reversible 5/3
# with the reversible colour transform, lossless.
opj_compress -i source_23x17.ppm -o rgb_23x17_origin_3_5.j2k -d 3,5 -n 4
ojph_compress -i source_23x17.ppm -o rgb_23x17_origin_3_5_ht.j2c -num_decomps 3 \
    -reversible true -colour_trans true -image_offset "{3,5}"

# OpenJPEG's decodes with 1, 2 and 3 resolutions discarded, as raw RGB8. Both
# codestreams must decode to the same samples, at full resolution as well.
for codestream in rgb_23x17_origin_3_5.j2k rgb_23x17_origin_3_5_ht.j2c; do
    for levels in 0 1 2 3; do
        opj_decompress -i "$codestream" -o "decoded_r${levels}.ppm" -r "$levels" >/dev/null
        python3 - "decoded_r${levels}.ppm" "decoded_r${levels}.rgb" <<'PY'
import sys

data = open(sys.argv[1], "rb").read()
fields = []
position = 0
while len(fields) < 4:
    while data[position:position + 1].isspace():
        position += 1
    if data[position:position + 1] == b"#":
        position = data.index(b"\n", position) + 1
        continue
    end = position
    while not data[end:end + 1].isspace():
        end += 1
    fields.append(data[position:end])
    position = end
width, height = int(fields[1]), int(fields[2])
samples = data[position + 1:]
assert fields[0] == b"P6" and int(fields[3]) == 255 and len(samples) == width * height * 3
open(sys.argv[2], "wb").write(samples)
print(sys.argv[2], width, height)
PY
        rm "decoded_r${levels}.ppm"
        if [ "$levels" = 0 ]; then
            cmp decoded_r0.rgb source_23x17.rgb
            rm decoded_r0.rgb
        elif [ "$codestream" = rgb_23x17_origin_3_5.j2k ]; then
            mv "decoded_r${levels}.rgb" "reduced_r${levels}.rgb"
        else
            cmp "decoded_r${levels}.rgb" "reduced_r${levels}.rgb"
            rm "decoded_r${levels}.rgb"
        fi
    done
done

rm source_23x17.ppm
opj_compress -h 2>&1 | grep -m1 -i 'openjp2\|version' || true
sha256sum ./*.j2k ./*.j2c ./*.rgb 2>/dev/null || shasum -a 256 ./*.j2k ./*.j2c ./*.rgb

# 29-bit grayscale exercises the exact integer inverse transform at an odd
# image origin. Keep the source formula in the test and the reduced OpenJPEG
# samples here; these fixtures do not need a second stored source image.
python3 - <<'PY'
from pathlib import Path
import subprocess
import tempfile

width, height = 23, 17
samples = [
    (1 << 28) + x * 17_003 + y * 9_001 + x * y * 997
    for y in range(height) for x in range(width)
]
work = Path(tempfile.mkdtemp(prefix="j2k-gray29-origin-"))
try:
    source = work / "source.pgx"
    source.write_bytes(
        f"PG ML + 29 {width} {height}\n".encode()
        + b"".join(value.to_bytes(4, "big") for value in samples)
    )
    codestream = "gray29_23x17_origin_3_3.j2k"
    subprocess.run([
        "opj_compress", "-i", str(source), "-o", codestream,
        "-d", "3,3", "-n", "3",
    ], check=True)
    for reduction, dimensions in enumerate([(23, 17), (11, 8), (6, 4)]):
        output = work / f"decoded_r{reduction}.pgx"
        subprocess.run([
            "opj_decompress", "-i", codestream, "-o", str(output),
            "-r", str(reduction), "-quiet",
        ], check=True)
        header, body = output.with_name(f"decoded_r{reduction}_0.pgx").read_bytes().split(b"\n", 1)
        fields = header.split()
        assert fields[:4] == [b"PG", b"ML", b"+", b"29"]
        assert tuple(map(int, fields[4:])) == dimensions
        decoded = [int.from_bytes(body[i:i + 4], "big") for i in range(0, len(body), 4)]
        assert len(decoded) == dimensions[0] * dimensions[1]
        if reduction == 0:
            assert decoded == samples
        else:
            Path(f"gray29_reduced_r{reduction}.raw").write_bytes(
                b"".join(value.to_bytes(4, "little") for value in decoded)
            )
finally:
    subprocess.run(["trash", str(work)], check=True)
PY
