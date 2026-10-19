#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Regenerate the libjpeg-turbo DCT-scaling reference matrix.

Run manually (it is also invoked by generate.sh) and commit the output. CI does
not run it; tests read the committed files.

Every case is encoded with `cjpeg` and decoded with `djpeg -scale 1/N` for
N = 1, 2, 4, 8 using default (fancy) upsampling and the accurate integer IDCT,
which is how OpenSlide scale-decodes NDPI/VMS levels. The matrix covers the
libjpeg-turbo behaviours j2k must reproduce:

- chroma that libjpeg-turbo decodes with a larger reduced IDCT so it needs less
  or no upsampling (4:2:0, 4:1:0 at 1/2, 1/4, 1/8);
- box (replicating) upsampling at 1/8, where libjpeg-turbo disables smoothing;
- box upsampling when a smoothed 2:1 component is at most 2 samples wide;
- progressive and 12-bit reduced decodes.

Output (in crates/j2k-test-support/fixtures/scaled_matrix/):

- cases.tsv  one line per case: name, layout, coding, precision, width,
             height, then offset:length of the JPEG and of each reference in
             data.bin (full, 1/2, 1/4, 1/8).
- data.bin   the JPEG inputs and headerless references, concatenated. 8-bit
             references are interleaved RGB or gray bytes; 12-bit references
             are big-endian 16-bit samples as djpeg writes them.

Requires libjpeg-turbo 3.x `cjpeg` and `djpeg` on PATH.
"""

import math
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(
    HERE, "..", "..", "crates", "j2k-test-support", "fixtures", "scaled_matrix"
)

# name -> cjpeg -sample argument (luma factors; chroma are 1x1), or None for
# grayscale.
LAYOUTS = [
    ("gray", None),
    ("444", "1x1"),
    ("422", "2x1"),
    ("440", "1x2"),
    ("420", "2x2"),
    ("411", "4x1"),
    ("410", "4x2"),
    ("1x4", "1x4"),
]

# 3 and 4 pixel wide images leave 2:1 chroma at most 2 samples wide at full
# size; 9 does at 1/4; the larger sizes span several MCUs and partial edges.
SIZES_8BIT = [(3, 3), (4, 4), (9, 9), (18, 14), (45, 31), (67, 45)]
# Tall narrow images make the component planes dominate the scratch budget,
# while full-size 2:1 chroma still takes the scaled (replicating) writer.
SIZES_12BIT = [(4, 4), (9, 9), (45, 31), (4, 64)]

CODINGS_8BIT = [
    ("baseline", ["-baseline", "-optimize"]),
    ("restart", ["-baseline", "-optimize", "-restart", "1B"]),
    ("progressive", ["-progressive"]),
]
CODINGS_12BIT = [
    ("extended", ["-optimize"]),
    ("progressive", ["-progressive"]),
]

# Multi-scan sequential coding: the components split across scans, one per
# scan or partly interleaved, alternating with a restart interval so both
# the component-grid and the MCU-grid restart counts are exercised.
MULTISCAN_SIZES_8BIT = [(4, 4), (18, 14), (45, 31), (67, 45)]
MULTISCAN_SIZES_12BIT = [(9, 9), (45, 31)]
SPLIT_SCRIPTS = {
    3: ["0;\n1;\n2;\n", "0 1;\n2;\n"],
    4: ["0;\n1;\n2;\n3;\n", "0 1 2;\n3;\n"],
}

# Four-component JPEG (Adobe APP14): name, transform (0 CMYK, 2 YCCK), and
# per-component sampling. cjpeg cannot read CMYK input, so
# `encode_four_component` writes a sequential source and jpegtran derives the
# other codings from its coefficients losslessly.
FOUR_COMPONENT_LAYOUTS = [
    ("cmyk444", 0, [(1, 1), (1, 1), (1, 1), (1, 1)]),
    ("cmyk420", 0, [(2, 2), (1, 1), (1, 1), (1, 1)]),
    ("ycck444", 2, [(1, 1), (1, 1), (1, 1), (1, 1)]),
    ("ycck420", 2, [(2, 2), (1, 1), (1, 1), (2, 2)]),
]
FOUR_COMPONENT_SIZES_8BIT = [(4, 4), (18, 14), (45, 31)]
FOUR_COMPONENT_SIZES_12BIT = [(9, 9), (45, 31)]
FOUR_COMPONENT_CODINGS_8BIT = [
    ("baseline", ["-optimize"]),
    ("restart", ["-optimize", "-restart", "1B"]),
    ("progressive", ["-progressive"]),
]
FOUR_COMPONENT_CODINGS_12BIT = [
    ("extended", ["-optimize"]),
    ("progressive", ["-progressive"]),
]

SCALES = [1, 2, 4, 8]


def texture(width, height, precision):
    """Deterministic integer-only texture with strong chroma detail."""
    state = 20260927
    maxval = (1 << precision) - 1
    shift = precision - 8
    pixels = []
    for y in range(height):
        for x in range(width):
            rgb = [
                28 + abs((x * 9 + y * 4) % 144 - 72) * 200 // 72,
                40 + x * 180 // max(width - 1, 1),
                200 - y * 150 // max(height - 1, 1) + abs((x + y) * 7 % 64 - 32) - 16,
            ]
            if (x + 2 * y) % 5 == 0:
                rgb = [rgb[2], rgb[0], rgb[1]]
            if y >= height // 2 and x >= width // 2:
                state = (state * 1103515245 + 12345) & 0x7FFFFFFF
                rgb = [value + state % 121 - 60 for value in rgb]
            # Saturated corner, kept off tiny images so they stay textured.
            if width >= 9 and y < 3 and x < 3:
                rgb = [255, 255, 0]
            rgb = [min(max(value, 0), 255) for value in rgb]
            if shift > 0:
                rgb = [(value << shift) | ((x * 5 + y * 3 + i) % (1 << shift)) for i, value in enumerate(rgb)]
                rgb = [min(value, maxval) for value in rgb]
            elif shift < 0:
                rgb = [value >> -shift for value in rgb]
            pixels.append(rgb)
    return pixels


def pnm(width, height, precision, gray):
    maxval = (1 << precision) - 1
    magic = "P5" if gray else "P6"
    body = bytearray()
    for rgb in texture(width, height, precision):
        samples = [(rgb[0] * 77 + rgb[1] * 150 + rgb[2] * 29) >> 8] if gray else rgb
        for value in samples:
            if maxval > 255:
                body += value.to_bytes(2, "big")
            else:
                body.append(value)
    return f"{magic}\n{width} {height}\n{maxval}\n".encode() + bytes(body)


class BitWriter:
    def __init__(self):
        self.out = bytearray()
        self.acc = 0
        self.bits = 0

    def put(self, value, count):
        for bit in range(count - 1, -1, -1):
            self.acc = (self.acc << 1) | ((value >> bit) & 1)
            self.bits += 1
            if self.bits == 8:
                self.out.append(self.acc)
                if self.acc == 0xFF:
                    self.out.append(0)
                self.acc = 0
                self.bits = 0

    def flush(self):
        if self.bits:
            self.put((1 << (8 - self.bits)) - 1, 8 - self.bits)
        data = bytes(self.out)
        self.out = bytearray()
        return data


def segment(marker, payload):
    return bytes([0xFF, marker]) + (len(payload) + 2).to_bytes(2, "big") + payload


def strip_pnm_header(data):
    index = 0
    for _ in range(3):
        index = data.index(b"\n", index) + 1
    return data[index:]


def run(args, stdin):
    return subprocess.run(args, input=stdin, capture_output=True, check=True).stdout


# Annex K.1 quantization tables in natural order.
LUMA_QUANT = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
]
CHROMA_QUANT = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99,
] + [99] * 32
ZIGZAG = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
]
# Orthonormal 8-point DCT basis scaled by 2^13 (JPEG's C(u)/2 factors).
DCT_SCALE = 8192
DCT_BASIS = [
    [
        round(DCT_SCALE * (math.sqrt(0.125) if u == 0 else 0.5) * math.cos((2 * x + 1) * u * math.pi / 16))
        for x in range(8)
    ]
    for u in range(8)
]


def scaled_quant(base, quality=90):
    scale = 200 - 2 * quality
    return [min(max((value * scale + 50) // 100, 1), 255) for value in base]


def div_round(numerator, denominator):
    magnitude = (abs(numerator) + denominator // 2) // denominator
    return magnitude if numerator >= 0 else -magnitude


def forward_dct_quantized(block, quant):
    """Integer FDCT of 64 level-shifted samples, quantized, in zigzag order."""
    rows = [[sum(DCT_BASIS[u][x] * block[y * 8 + x] for x in range(8)) for u in range(8)] for y in range(8)]
    natural = [
        div_round(sum(DCT_BASIS[v][y] * rows[y][u] for y in range(8)), DCT_SCALE * DCT_SCALE * quant[v * 8 + u])
        for v in range(8)
        for u in range(8)
    ]
    return [natural[index] for index in ZIGZAG]


def four_component_planes(width, height, precision, transform):
    """Full-size stored planes: complemented CMYK from the RGB texture, or the
    YCCK libjpeg's cmyk_ycck_convert writes for it."""
    maxval = (1 << precision) - 1
    center = 1 << (precision - 1)
    planes = [[], [], [], []]
    for index, (c, m, y_ink) in enumerate(texture(width, height, precision)):
        x, y = index % width, index // width
        k = maxval - ((x * 3 + y * 5) % 97) * maxval // 255
        if transform == 2:
            r, g, b = maxval - c, maxval - m, maxval - y_ink
            luma = (19595 * r + 38470 * g + 7471 * b + 32768) >> 16
            cb = ((-11059 * r - 21709 * g + 32768 * b + 32768) >> 16) + center
            cr = ((32768 * r - 27439 * g - 5329 * b + 32768) >> 16) + center
            c, m, y_ink = (min(max(value, 0), maxval) for value in (luma, cb, cr))
        for plane, value in zip(planes, (c, m, y_ink, k)):
            plane.append(value)
    return planes


def fixed_length_table(table_class, symbols, length):
    """DHT payload giving every symbol a `length`-bit code, and the codes."""
    counts = [0] * 16
    counts[length - 1] = len(symbols)
    payload = bytes([table_class << 4]) + bytes(counts) + bytes(symbols)
    return payload, {symbol: code for code, symbol in enumerate(symbols)}


def encode_four_component(width, height, precision, transform, sampling):
    """Minimal interleaved sequential encoder for Adobe CMYK/YCCK: box
    downsampling, integer FDCT, fixed-length Huffman codes (jpegtran
    re-optimizes them)."""
    max_size = 11 if precision == 8 else 15
    dc_symbols = list(range(max_size + 1))
    dc_length = 4 if len(dc_symbols) < 16 else 5
    ac_symbols = [0x00, 0xF0] + [(run << 4) | size for run in range(16) for size in range(1, max_size)]
    dc_payload, dc_codes = fixed_length_table(0, dc_symbols, dc_length)
    ac_payload, ac_codes = fixed_length_table(1, ac_symbols, 8)
    quant = [scaled_quant(LUMA_QUANT), scaled_quant(CHROMA_QUANT)]
    table_of = [0, 1, 1, 0] if transform == 2 else [0, 0, 0, 0]

    hmax = max(h for h, _ in sampling)
    vmax = max(v for _, v in sampling)
    mcus_x, mcus_y = -(-width // (8 * hmax)), -(-height // (8 * vmax))
    full = four_component_planes(width, height, precision, transform)
    planes = []
    for plane, (h, v) in zip(full, sampling):
        fx, fy = hmax // h, vmax // v
        comp_w, comp_h = -(-width * h // hmax), -(-height * v // vmax)

        def sample(cx, cy, plane=plane, fx=fx, fy=fy):
            box = [
                plane[min(cy * fy + j, height - 1) * width + min(cx * fx + i, width - 1)]
                for j in range(fy)
                for i in range(fx)
            ]
            return (sum(box) + len(box) // 2) // len(box)

        sampled = [[sample(cx, cy) for cx in range(comp_w)] for cy in range(comp_h)]
        planes.append((sampled, comp_w, comp_h))

    writer = BitWriter()
    prev_dc = [0, 0, 0, 0]
    shift = 1 << (precision - 1)
    for mcu_y in range(mcus_y):
        for mcu_x in range(mcus_x):
            for component, ((sampled, comp_w, comp_h), (h, v)) in enumerate(zip(planes, sampling)):
                for by in range(v):
                    for bx in range(h):
                        x0, y0 = (mcu_x * h + bx) * 8, (mcu_y * v + by) * 8
                        block = [
                            sampled[min(y0 + j, comp_h - 1)][min(x0 + i, comp_w - 1)] - shift
                            for j in range(8)
                            for i in range(8)
                        ]
                        coeffs = forward_dct_quantized(block, quant[table_of[component]])
                        diff = coeffs[0] - prev_dc[component]
                        prev_dc[component] = coeffs[0]
                        size = abs(diff).bit_length()
                        assert size <= max_size, "DC difference exceeds the table"
                        writer.put(dc_codes[size], dc_length)
                        writer.put(diff if diff >= 0 else diff + (1 << size) - 1, size)
                        zero_run = 0
                        for coeff in coeffs[1:]:
                            if coeff == 0:
                                zero_run += 1
                                continue
                            while zero_run > 15:
                                writer.put(ac_codes[0xF0], 8)
                                zero_run -= 16
                            size = abs(coeff).bit_length()
                            assert size < max_size, "AC coefficient exceeds the table"
                            writer.put(ac_codes[(zero_run << 4) | size], 8)
                            writer.put(coeff if coeff >= 0 else coeff + (1 << size) - 1, size)
                            zero_run = 0
                        if zero_run:
                            writer.put(ac_codes[0x00], 8)

    adobe = b"Adobe" + (100).to_bytes(2, "big") + bytes(4) + bytes([transform])
    dqt = b"".join(bytes([index]) + bytes(table[i] for i in ZIGZAG) for index, table in enumerate(quant))
    sof = bytes([precision]) + height.to_bytes(2, "big") + width.to_bytes(2, "big") + bytes([4])
    sof += b"".join(bytes([i + 1, (h << 4) | v, table_of[i]]) for i, (h, v) in enumerate(sampling))
    sos = bytes([4]) + b"".join(bytes([i + 1, 0x00]) for i in range(4)) + bytes([0, 63, 0])
    return (
        b"\xFF\xD8"
        + segment(0xEE, adobe)
        + segment(0xDB, dqt)
        + segment(0xC0 if precision == 8 else 0xC1, sof)
        + segment(0xC4, dc_payload + ac_payload)
        + segment(0xDA, sos)
        + writer.flush()
        + b"\xFF\xD9"
    )


def multiscan_args(script_dir, components, index):
    """jpegtran/cjpeg arguments splitting `components` across scans."""
    script = os.path.join(script_dir, f"split{components}_{index % 2}.txt")
    with open(script, "w", encoding="ascii") as handle:
        handle.write(SPLIT_SCRIPTS[components][index % 2])
    args = ["-optimize", "-scans", script]
    return args + (["-restart", "3B"] if index % 2 else [])


def main():
    probe = subprocess.run(["cjpeg", "-version"], capture_output=True, check=False)
    version = (probe.stderr or probe.stdout).decode().strip().splitlines()[0]
    if "libjpeg-turbo version 3." not in version:
        sys.exit(f"error: need libjpeg-turbo 3.x cjpeg/djpeg, found {version!r}")

    os.makedirs(OUT_DIR, exist_ok=True)
    data = bytearray()
    lines = [
        f"# Generated by corpus/conformance/scaled_matrix.py with {version}",
        "# name\tlayout\tcoding\tprecision\twidth\theight\tjpeg\tscale1\tscale2\tscale4\tscale8",
    ]

    def append(blob):
        start = len(data)
        data.extend(blob)
        return f"{start}:{len(blob)}"

    def record(layout, coding, precision, width, height, jpeg):
        gray = layout == "gray"
        refs = []
        for denom in SCALES:
            out = ["djpeg", "-dct", "int", "-scale", f"1/{denom}"]
            # djpeg writes CMYK/YCCK as RGB (c * k / max per channel).
            out += ["-grayscale"] if gray else [] if layout[:4] in ("cmyk", "ycck") else ["-rgb"]
            ref = strip_pnm_header(run(out, jpeg))
            channels = 1 if gray else 3
            bytes_per_sample = 2 if precision > 8 else 1
            expected = -(-width // denom) * -(-height // denom) * channels * bytes_per_sample
            if len(ref) != expected:
                sys.exit(f"error: {layout} {width}x{height} {coding} 1/{denom}: {len(ref)} != {expected}")
            refs.append(append(ref))
        name = f"{layout}_{width}x{height}_{coding}_{precision}bit"
        lines.append(
            "\t".join([name, layout, coding, str(precision), str(width), str(height), append(jpeg)] + refs)
        )

    with tempfile.TemporaryDirectory() as script_dir:
        matrix = [
            (8, SIZES_8BIT, CODINGS_8BIT, MULTISCAN_SIZES_8BIT),
            (12, SIZES_12BIT, CODINGS_12BIT, MULTISCAN_SIZES_12BIT),
        ]
        for precision, sizes, codings, multiscan_sizes in matrix:
            for layout, sample in LAYOUTS:
                gray = sample is None
                args = ["cjpeg", "-quality", "90", "-precision", str(precision)]
                args += ["-grayscale"] if gray else ["-sample", f"{sample},1x1,1x1"]
                for width, height in sizes:
                    for coding, coding_args in codings:
                        jpeg = run(args + coding_args, pnm(width, height, precision, gray))
                        record(layout, coding, precision, width, height, jpeg)
                if gray:
                    continue
                for index, (width, height) in enumerate(multiscan_sizes):
                    split = multiscan_args(script_dir, 3, index)
                    jpeg = run(args + split, pnm(width, height, precision, gray))
                    record(layout, "multiscan", precision, width, height, jpeg)

        four_component = [
            (8, FOUR_COMPONENT_SIZES_8BIT, FOUR_COMPONENT_CODINGS_8BIT),
            (12, FOUR_COMPONENT_SIZES_12BIT, FOUR_COMPONENT_CODINGS_12BIT),
        ]
        for precision, sizes, codings in four_component:
            for layout, transform, sampling in FOUR_COMPONENT_LAYOUTS:
                for index, (width, height) in enumerate(sizes):
                    source = encode_four_component(width, height, precision, transform, sampling)
                    for coding, coding_args in codings:
                        record(layout, coding, precision, width, height, run(["jpegtran"] + coding_args, source))
                    split = multiscan_args(script_dir, 4, index)
                    record(layout, "multiscan", precision, width, height, run(["jpegtran"] + split, source))

    with open(os.path.join(OUT_DIR, "data.bin"), "wb") as handle:
        handle.write(data)
    with open(os.path.join(OUT_DIR, "cases.tsv"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines) - 2} cases, {len(data)} bytes")


if __name__ == "__main__":
    main()
