#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Regenerate libjpeg-turbo reference fixtures. Run manually and commit the
# output. CI does not run this script; it checks the committed manifest hashes.
#
# Requires: libjpeg-turbo installed with `cjpeg` and `djpeg` on PATH.
#   macOS:  brew install jpeg-turbo
#   Linux:  apt-get install libjpeg-turbo-progs
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v cjpeg >/dev/null || ! command -v djpeg >/dev/null; then
    echo "error: cjpeg/djpeg not on PATH; install libjpeg-turbo" >&2
    exit 1
fi

LJT_VERSION=$(cjpeg -version 2>&1 | head -1 || true)
echo "Using: $LJT_VERSION"

write_rgb_ppm() {
    local width="$1"
    local height="$2"
    python3 - "$width" "$height" <<'PY'
import sys

width = int(sys.argv[1])
height = int(sys.argv[2])
header = f"P6\n{width} {height}\n255\n".encode()
body = bytearray()
for y in range(height):
    for x in range(width):
        body.extend([
            (x * 255 // max(width - 1, 1)) & 0xFF,
            (y * 255 // max(height - 1, 1)) & 0xFF,
            ((x + y) * 255 // max(width + height - 2, 1)) & 0xFF,
        ])
sys.stdout.buffer.write(header + bytes(body))
PY
}

write_gray_pgm() {
    local width="$1"
    local height="$2"
    python3 - "$width" "$height" <<'PY'
import sys

width = int(sys.argv[1])
height = int(sys.argv[2])
header = f"P5\n{width} {height}\n255\n".encode()
body = bytes((x * 255 // max(width - 1, 1)) & 0xFF for _ in range(height) for x in range(width))
sys.stdout.buffer.write(header + body)
PY
}

# Integer-only texture (no floating point or library RNG) so every host
# produces the same pixels: triangle waves, two flat patches that code as
# DC-only blocks, LCG noise, and a saturated corner.
write_texture_ppm() {
    local width="$1"
    local height="$2"
    python3 - "$width" "$height" <<'PY'
import sys

width = int(sys.argv[1])
height = int(sys.argv[2])
state = 20260927


def noise():
    global state
    state = (state * 1103515245 + 12345) & 0x7FFFFFFF
    return state % 121 - 60


header = f"P6\n{width} {height}\n255\n".encode()
body = bytearray()
for y in range(height):
    for x in range(width):
        rgb = [
            28 + abs((x * 9 + y * 4) % 144 - 72) * 200 // 72,
            40 + x * 180 // max(width - 1, 1),
            200 - y * 150 // max(height - 1, 1) + abs((x + y) * 7 % 64 - 32) - 16,
        ]
        if y < 16 and 40 <= x < 56:
            rgb = [201, 77, 143]
        if 24 <= y < 40 and 8 <= x < 24:
            rgb = [19, 233, 98]
        if y >= 30 and x >= 40:
            rgb = [value + noise() for value in rgb]
        if y < 10 and x < 12:
            rgb = [255, 255, 0]
        body.extend(min(max(value, 0), 255) for value in rgb)
sys.stdout.buffer.write(header + bytes(body))
PY
}

strip_pnm_header() {
    python3 -c 'import sys
data = sys.stdin.buffer.read()
i = 0
for _ in range(3):
    i = data.index(b"\n", i) + 1
sys.stdout.buffer.write(data[i:])'
}

sha256_file() {
    if command -v sha256sum >/dev/null; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

assert_size() {
    local path="$1"
    local expected="$2"
    local actual
    actual=$(wc -c < "$path" | tr -d ' ')
    if [ "$actual" != "$expected" ]; then
        echo "error: $path has $actual bytes, expected $expected" >&2
        exit 1
    fi
}

write_rgb_ppm 16 16 \
    | cjpeg -quality 90 -sample 2x2,1x1,1x1 -baseline -optimize \
        -outfile baseline_420_16x16.jpg
djpeg -rgb baseline_420_16x16.jpg | strip_pnm_header > baseline_420_16x16.rgb
assert_size baseline_420_16x16.rgb 768

write_rgb_ppm 32 16 \
    | cjpeg -quality 90 -sample 2x2,1x1,1x1 -baseline -optimize -restart 1 \
        -outfile baseline_420_restart_32x16.jpg
djpeg -rgb baseline_420_restart_32x16.jpg \
    | strip_pnm_header > baseline_420_restart_32x16.rgb
assert_size baseline_420_restart_32x16.rgb 1536

write_rgb_ppm 16 8 \
    | cjpeg -quality 90 -sample 2x1,1x1,1x1 -baseline -optimize \
        -outfile baseline_422_16x8.jpg
djpeg -rgb baseline_422_16x8.jpg | strip_pnm_header > baseline_422_16x8.rgb
assert_size baseline_422_16x8.rgb 384

write_rgb_ppm 8 8 \
    | cjpeg -quality 90 -sample 1x1,1x1,1x1 -baseline -optimize \
        -outfile baseline_444_8x8.jpg
djpeg -rgb baseline_444_8x8.jpg | strip_pnm_header > baseline_444_8x8.rgb
assert_size baseline_444_8x8.rgb 192

write_gray_pgm 8 8 \
    | cjpeg -quality 90 -grayscale -baseline -optimize \
        -outfile grayscale_8x8.jpg
djpeg -grayscale grayscale_8x8.jpg | strip_pnm_header > grayscale_8x8.gray
assert_size grayscale_8x8.gray 64

# DCT-scaled references: the reduced IDCTs behind OpenSlide's NDPI/VMS levels.
write_texture_ppm 67 45 \
    | cjpeg -quality 90 -sample 1x1,1x1,1x1 -baseline -optimize -restart 3B \
        -outfile baseline_444_restart_67x45.jpg
djpeg -rgb -scale 1/2 baseline_444_restart_67x45.jpg \
    | strip_pnm_header > baseline_444_restart_67x45_half.rgb
assert_size baseline_444_restart_67x45_half.rgb 2346
djpeg -rgb -scale 1/4 baseline_444_restart_67x45.jpg \
    | strip_pnm_header > baseline_444_restart_67x45_quarter.rgb
assert_size baseline_444_restart_67x45_quarter.rgb 612
djpeg -rgb -scale 1/8 baseline_444_restart_67x45.jpg \
    | strip_pnm_header > baseline_444_restart_67x45_eighth.rgb
assert_size baseline_444_restart_67x45_eighth.rgb 162

cat > manifest.json <<EOF
{
  "libjpeg_turbo_version": "$LJT_VERSION",
  "generated_on": "$(date -u +%FT%TZ)",
  "fixtures": [
    {
      "input": "baseline_420_16x16.jpg",
      "input_sha256": "$(sha256_file baseline_420_16x16.jpg)",
      "reference": "baseline_420_16x16.rgb",
      "reference_sha256": "$(sha256_file baseline_420_16x16.rgb)",
      "format": "Rgb8",
      "width": 16,
      "height": 16,
      "tolerance": "bit_exact",
      "sampling": "4:2:0"
    },
    {
      "input": "baseline_420_restart_32x16.jpg",
      "input_sha256": "$(sha256_file baseline_420_restart_32x16.jpg)",
      "reference": "baseline_420_restart_32x16.rgb",
      "reference_sha256": "$(sha256_file baseline_420_restart_32x16.rgb)",
      "format": "Rgb8",
      "width": 32,
      "height": 16,
      "tolerance": "bit_exact",
      "sampling": "4:2:0 restart-coded"
    },
    {
      "input": "baseline_422_16x8.jpg",
      "input_sha256": "$(sha256_file baseline_422_16x8.jpg)",
      "reference": "baseline_422_16x8.rgb",
      "reference_sha256": "$(sha256_file baseline_422_16x8.rgb)",
      "format": "Rgb8",
      "width": 16,
      "height": 8,
      "tolerance": "bit_exact",
      "sampling": "4:2:2"
    },
    {
      "input": "baseline_444_8x8.jpg",
      "input_sha256": "$(sha256_file baseline_444_8x8.jpg)",
      "reference": "baseline_444_8x8.rgb",
      "reference_sha256": "$(sha256_file baseline_444_8x8.rgb)",
      "format": "Rgb8",
      "width": 8,
      "height": 8,
      "tolerance": "bit_exact",
      "sampling": "4:4:4"
    },
    {
      "input": "grayscale_8x8.jpg",
      "input_sha256": "$(sha256_file grayscale_8x8.jpg)",
      "reference": "grayscale_8x8.gray",
      "reference_sha256": "$(sha256_file grayscale_8x8.gray)",
      "format": "Gray8",
      "width": 8,
      "height": 8,
      "tolerance": "bit_exact",
      "sampling": "grayscale"
    },
    {
      "input": "baseline_444_restart_67x45.jpg",
      "input_sha256": "$(sha256_file baseline_444_restart_67x45.jpg)",
      "reference": "baseline_444_restart_67x45_half.rgb",
      "reference_sha256": "$(sha256_file baseline_444_restart_67x45_half.rgb)",
      "format": "Rgb8",
      "width": 34,
      "height": 23,
      "scale": "1/2",
      "tolerance": "bit_exact",
      "sampling": "4:4:4 restart-coded",
      "generator": "$LJT_VERSION"
    },
    {
      "input": "baseline_444_restart_67x45.jpg",
      "input_sha256": "$(sha256_file baseline_444_restart_67x45.jpg)",
      "reference": "baseline_444_restart_67x45_quarter.rgb",
      "reference_sha256": "$(sha256_file baseline_444_restart_67x45_quarter.rgb)",
      "format": "Rgb8",
      "width": 17,
      "height": 12,
      "scale": "1/4",
      "tolerance": "bit_exact",
      "sampling": "4:4:4 restart-coded",
      "generator": "$LJT_VERSION"
    },
    {
      "input": "baseline_444_restart_67x45.jpg",
      "input_sha256": "$(sha256_file baseline_444_restart_67x45.jpg)",
      "reference": "baseline_444_restart_67x45_eighth.rgb",
      "reference_sha256": "$(sha256_file baseline_444_restart_67x45_eighth.rgb)",
      "format": "Rgb8",
      "width": 9,
      "height": 6,
      "scale": "1/8",
      "tolerance": "bit_exact",
      "sampling": "4:4:4 restart-coded",
      "generator": "$LJT_VERSION"
    }
  ]
}
EOF

# DCT-scaling reference matrix (every chroma layout, progressive and 12-bit)
# for the scaled-decode tests; written to j2k-test-support/fixtures.
python3 scaled_matrix.py

echo "Regenerated fixtures:"
ls -la *.jpg *.rgb *.gray manifest.json
