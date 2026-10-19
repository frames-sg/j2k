#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Regenerate the lossless (SOF3) JPEG matrix.

Run manually (it is also invoked by generate.sh) and commit the output. CI does
not run it; tests read the committed files.

Grayscale and RGB 4:4:4 cases are encoded with libjpeg-turbo `cjpeg
-lossless`, split (multi-scan) RGB cases with `cjpeg -scans`. libjpeg-turbo's
lossless mode performs no color conversion or subsampling, so YCbCr and
subsampled cases come from the small reference SOF3 encoder below. The
matrix covers what lossless decoders must reconstruct exactly:

- every predictor (1..7) at every precision libjpeg-turbo encodes (2..16),
  on odd and narrow images;
- point transforms, which shift reconstructed samples on output;
- restart intervals, which reseed the 1-D predictor;
- RGB and YCbCr color at 4:4:4, 4:2:2, 4:2:0, 4:4:0, and 4:1:1, at odd sizes;
- split frames whose components are coded in several scans, each with its
  own predictor, point transform, and restart interval.

Every case libjpeg-turbo decodes (grayscale and RGB; it rejects YCbCr color
conversion in lossless mode) records `djpeg` output as its reference:
libjpeg-turbo replicates subsampled components in lossless mode.

Output (in crates/j2k-test-support/fixtures/lossless_matrix/):

- cases.tsv  one line per case: name, layout, precision, predictor, point
             transform, restart rows, width, height, offset:length of the
             JPEG in data.bin, the scan split (`-` for one interleaved scan,
             else `0/12` style component groups), and offset:length of the
             reference (`-` when libjpeg-turbo cannot decode the case).
- data.bin   the JPEG inputs and headerless references, concatenated.
             References above 8 bits are big-endian 16-bit samples.

Requires libjpeg-turbo 3.x `cjpeg` and `djpeg` on PATH.
"""

import os
import subprocess
import sys
import tempfile

from scaled_matrix import BitWriter, pnm, segment, strip_pnm_header, texture

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(
    HERE, "..", "..", "crates", "j2k-test-support", "fixtures", "lossless_matrix"
)

# layout -> (cjpeg color arguments, gray input)
# layout -> cjpeg color arguments, or (luma sampling, APP14 RGB) for the
# reference encoder.
CJPEG_LAYOUTS = {
    "gray": ["-grayscale"],
    "rgb444": ["-rgb"],
}
ENCODER_LAYOUTS = {
    "ycbcr444": ((1, 1), False),
    "ycbcr422": ((2, 1), False),
    "ycbcr420": ((2, 2), False),
    "ycbcr440": ((1, 2), False),
    "rgb422": ((2, 1), True),
    "rgb420": ((2, 2), True),
    "rgb440": ((1, 2), True),
    "rgb411": ((4, 1), True),
}

# (layout, precision, predictor, point transform, restart rows, width, height)
CASES = (
    [("gray", 8, p, 0, 0, 67, 45) for p in range(1, 8)]
    + [("gray", 16, p, 0, 0, 67, 45) for p in range(1, 8)]
    + [
        # Point transforms, restarts, and narrow images.
        ("gray", 8, 1, 3, 0, 67, 45),
        ("gray", 16, 7, 5, 0, 67, 45),
        ("gray", 8, 4, 0, 7, 67, 45),
        ("gray", 16, 6, 0, 4, 67, 45),
        ("gray", 8, 5, 0, 0, 1, 37),
        ("gray", 16, 2, 0, 0, 37, 1),
        # 12-bit precision.
        ("gray", 12, 1, 0, 0, 67, 45),
    ]
    + [("ycbcr444", 8, p, 0, 0, 33, 17) for p in (1, 4, 5, 6, 7)]
    + [("ycbcr444", 8, 3, 0, 5, 33, 17), ("ycbcr422", 8, 2, 0, 0, 33, 17)]
    + [("ycbcr444", 16, p, 0, 0, 33, 17) for p in (2, 6)]
    + [("rgb444", 8, p, 0, 0, 33, 17) for p in (1, 3)]
    + [("rgb444", 16, p, 0, 0, 33, 17) for p in (4, 7)]
    + [("ycbcr422", 8, p, 0, 0, 34, 18) for p in (1, 6)]
    + [("ycbcr422", 16, p, 0, 0, 34, 18) for p in (1, 7)]
    + [("ycbcr420", 8, p, 0, 0, 34, 18) for p in (1, 5)]
    + [("ycbcr420", 16, p, 0, 0, 34, 18) for p in (2, 6)]
    + [
        ("ycbcr420", 8, 1, 0, 3, 34, 18),
        ("ycbcr422", 16, 4, 0, 2, 34, 18),
        ("ycbcr420", 8, 7, 0, 0, 66, 46),
        ("rgb422", 8, 4, 0, 0, 34, 18),
        ("rgb420", 8, 4, 0, 0, 34, 18),
        # Odd-width 4:2:2 and odd-height 4:2:0.
        ("ycbcr422", 8, 1, 0, 0, 33, 18),
        ("ycbcr420", 16, 1, 0, 0, 34, 17),
    ]
)


# Appended after the original cases so their names and bytes stay stable.
EXTRA_CASES = (
    # Every precision libjpeg-turbo encodes, with point transforms and restarts.
    [
        ("gray", precision, predictor, point_transform, restart, 67, 45)
        for precision, predictor, point_transform, restart in [
            (2, 1, 0, 0),
            (4, 7, 1, 0),
            (6, 4, 0, 3),
            (7, 2, 2, 0),
            (10, 1, 0, 0),
            (10, 5, 3, 2),
            (12, 7, 0, 0),
            (12, 1, 4, 5),
            (13, 6, 0, 0),
            (14, 3, 2, 0),
            (15, 1, 0, 4),
            (16, 1, 8, 0),
        ]
    ]
    + [
        ("rgb444", 12, 1, 0, 0, 33, 17),
        ("rgb444", 12, 6, 2, 1, 33, 17),
        ("rgb444", 4, 5, 1, 0, 33, 17),
        ("rgb444", 10, 7, 0, 0, 33, 17),
        ("rgb444", 16, 3, 3, 0, 33, 17),
    ]
    # Replicated chroma at odd sizes and every whole ratio.
    + [
        ("rgb422", 8, 1, 0, 0, 33, 17),
        ("rgb420", 8, 6, 0, 0, 33, 17),
        ("rgb420", 16, 2, 0, 2, 35, 19),
        ("rgb420", 12, 1, 0, 0, 34, 18),
        ("rgb440", 8, 3, 0, 0, 34, 17),
        ("rgb411", 8, 7, 0, 0, 35, 18),
        ("ycbcr420", 12, 5, 0, 0, 34, 18),
        ("ycbcr440", 10, 2, 0, 0, 33, 17),
    ]
)

# Split frames: (layout, precision, scans, restart rows, width, height), each
# scan a (frame component group, predictor, point transform).
SPLIT_CASES = [
    ("rgb444", 8, [((0,), 1, 0), ((1,), 1, 0), ((2,), 1, 0)], 0, 33, 17),
    ("rgb444", 12, [((0,), 1, 0), ((1, 2), 7, 0)], 2, 33, 17),
    ("rgb444", 16, [((0, 1), 4, 1), ((2,), 6, 3)], 0, 33, 17),
    ("ycbcr420", 8, [((0,), 1, 0), ((1, 2), 5, 0)], 2, 34, 17),
    ("rgb420", 8, [((0,), 3, 0), ((1,), 1, 0), ((2,), 7, 0)], 0, 33, 18),
    ("ycbcr422", 16, [((0, 1), 2, 0), ((2,), 6, 0)], 1, 33, 18),
    ("rgb440", 12, [((2,), 4, 0), ((0, 1), 1, 0)], 0, 34, 17),
]


def texture_planes(width, height, precision, sampling):
    """Component planes for the reference encoder: luma at full size, two
    chroma planes subsampled by `sampling` (taken from the RGB texture)."""
    maxval = (1 << precision) - 1
    rgb = texture(width, height, precision)
    h, v = sampling
    chroma_w, chroma_h = -(-width // h), -(-height // v)
    c0 = [min(rgb[i][0], maxval) for i in range(width * height)]
    c1 = [rgb[min(y * v, height - 1) * width + min(x * h, width - 1)][1]
          for y in range(chroma_h) for x in range(chroma_w)]
    c2 = [rgb[min(y * v, height - 1) * width + min(x * h, width - 1)][2]
          for y in range(chroma_h) for x in range(chroma_w)]
    return [(c0, width, height, h, v), (c1, chroma_w, chroma_h, 1, 1), (c2, chroma_w, chroma_h, 1, 1)]


def lossless_scan_entropy(width, height, precision, planes, predictor, restart_rows, frame_max):
    """Entropy data of one scan over `planes`. One plane is a non-interleaved
    scan whose MCU is one sample; several share the frame's MCU grid. MCU
    padding codes zero differences."""
    interleaved = len(planes) > 1
    if interleaved:
        mcus_x, mcus_y = -(-width // frame_max[0]), -(-height // frame_max[1])
    else:
        mcus_x, mcus_y = planes[0][1], planes[0][2]
    seed = 1 << (precision - 1)

    def prediction(plane, w, x, y, first_row):
        at = lambda xx, yy: plane[yy * w + xx]
        if first_row:
            return seed if x == 0 else at(x - 1, y)
        if x == 0:
            return at(0, y - 1)
        ra, rb, rc = at(x - 1, y), at(x, y - 1), at(x - 1, y - 1)
        return [None, ra, rb, rc, ra + rb - rc, ra + ((rb - rc) >> 1), rb + ((ra - rc) >> 1),
                (ra + rb) >> 1][predictor] & 0xFFFF

    writer = BitWriter()
    scan = bytearray()
    for my in range(mcus_y):
        if restart_rows and my and my % restart_rows == 0:
            scan += writer.flush() + bytes([0xFF, 0xD0 + (my // restart_rows - 1) % 8])
        for mx in range(mcus_x):
            for plane, w, h, ph, pv in planes:
                (ph, pv) = (ph, pv) if interleaved else (1, 1)
                for vy in range(pv):
                    for vx in range(ph):
                        x, y = mx * ph + vx, my * pv + vy
                        diff = 0
                        if x < w and y < h:
                            first_row = y == 0 or (restart_rows and y % (restart_rows * pv) == 0)
                            diff = (plane[y * w + x] - prediction(plane, w, x, y, first_row)) & 0xFFFF
                            if diff >= 0x8000:
                                diff -= 0x10000
                        category = 0 if diff == 0 else min(abs(diff).bit_length(), 16)
                        writer.put(category, 5)
                        if 0 < category < 16:
                            writer.put(diff if diff > 0 else diff - 1 + (1 << category), category)
    return bytes(scan + writer.flush()), mcus_x


def encode_lossless(width, height, precision, scans, restart_rows, planes, rgb):
    """Minimal SOF3 encoder: one Huffman table whose 17 categories all have
    5-bit codes. `scans` lists (frame component group, predictor) in coding
    order; each scan restarts every `restart_rows` of its own MCU rows, with
    its own DRI as `cjpeg -restart` writes."""
    frame_max = (max(p[3] for p in planes), max(p[4] for p in planes))
    ids = [ord("R"), ord("G"), ord("B")] if rgb else [1, 2, 3]
    out = bytes([0xFF, 0xD8])
    if rgb:
        out += segment(0xEE, b"Adobe" + bytes([0, 100, 0, 0, 0, 0, 0]))
    else:
        out += segment(0xE0, b"JFIF\x00" + bytes([1, 1, 0, 0, 1, 0, 1, 0, 0]))
    counts = [0] * 16
    counts[4] = 17
    out += segment(0xC4, bytes([0x00]) + bytes(counts) + bytes(range(17)))
    sof = bytes([precision]) + height.to_bytes(2, "big") + width.to_bytes(2, "big") + bytes([len(planes)])
    for component_id, (_, _, _, ph, pv) in zip(ids, planes):
        sof += bytes([component_id, (ph << 4) | pv, 0])
    out += segment(0xC3, sof)
    for group, predictor in scans:
        entropy, mcus_x = lossless_scan_entropy(
            width, height, precision, [planes[i] for i in group], predictor, restart_rows, frame_max
        )
        if restart_rows:
            out += segment(0xDD, (restart_rows * mcus_x).to_bytes(2, "big"))
        sos = bytes([len(group)]) + b"".join(bytes([ids[i], 0x00]) for i in group)
        out += segment(0xDA, sos + bytes([predictor, 0, 0])) + entropy
    return out + bytes([0xFF, 0xD9])


def run(args, stdin):
    return subprocess.run(args, input=stdin, capture_output=True, check=True).stdout


def djpeg_reference(jpeg):
    """`djpeg -pnm` samples, or `None` when libjpeg-turbo rejects the case."""
    result = subprocess.run(["djpeg", "-pnm"], input=jpeg, capture_output=True, check=False)
    return strip_pnm_header(result.stdout) if result.returncode == 0 else None


def scan_script(scans):
    """`cjpeg -scans` lines: components, then predictor (Ss), Se, Ah, Pt (Al)."""
    return "".join(
        " ".join(str(c) for c in group) + f": {predictor} 0 0 {point_transform};\n"
        for group, predictor, point_transform in scans
    )


def main():
    probe = subprocess.run(["cjpeg", "-version"], capture_output=True, check=False)
    version = (probe.stderr or probe.stdout).decode().strip().splitlines()[0]
    if "libjpeg-turbo version 3." not in version:
        sys.exit(f"error: need libjpeg-turbo 3.x cjpeg, found {version!r}")

    os.makedirs(OUT_DIR, exist_ok=True)
    data = bytearray()
    lines = [
        f"# Generated by corpus/conformance/lossless_matrix.py with {version}",
        "# name\tlayout\tprecision\tpredictor\tpoint_transform\trestart_rows\twidth\theight\tjpeg"
        "\tscans\treference",
    ]

    def append(blob):
        start = len(data)
        data.extend(blob)
        return f"{start}:{len(blob)}"

    def record(name, fields, jpeg, split):
        reference = djpeg_reference(jpeg)
        lines.append(
            "\t".join(
                [name]
                + [str(field) for field in fields]
                + [append(jpeg), split, append(reference) if reference is not None else "-"]
            )
        )

    for layout, precision, predictor, point_transform, restart, width, height in CASES + EXTRA_CASES:
        if layout in CJPEG_LAYOUTS:
            args = ["cjpeg", "-precision", str(precision), "-lossless", f"{predictor},{point_transform}"]
            args += CJPEG_LAYOUTS[layout]
            if restart:
                args += ["-restart", str(restart)]
            jpeg = run(args, pnm(width, height, precision, layout == "gray"))
        else:
            sampling, rgb = ENCODER_LAYOUTS[layout]
            planes = texture_planes(width, height, precision, sampling)
            jpeg = encode_lossless(width, height, precision, [((0, 1, 2), predictor)], restart, planes, rgb)
        name = f"{layout}_{width}x{height}_{precision}bit_p{predictor}_pt{point_transform}"
        if restart:
            name += f"_rst{restart}"
        record(name, (layout, precision, predictor, point_transform, restart, width, height), jpeg, "-")

    with tempfile.TemporaryDirectory() as script_dir:
        for layout, precision, scans, restart, width, height in SPLIT_CASES:
            if layout in CJPEG_LAYOUTS:
                script = os.path.join(script_dir, "scans.txt")
                with open(script, "w", encoding="ascii") as handle:
                    handle.write(scan_script(scans))
                args = ["cjpeg", "-precision", str(precision), "-lossless", "1"]
                args += CJPEG_LAYOUTS[layout] + ["-scans", script]
                if restart:
                    args += ["-restart", str(restart)]
                jpeg = run(args, pnm(width, height, precision, False))
            else:
                if any(point_transform for _, _, point_transform in scans):
                    sys.exit(f"error: the reference encoder has no point transform ({layout})")
                sampling, rgb = ENCODER_LAYOUTS[layout]
                planes = texture_planes(width, height, precision, sampling)
                coded = [(group, predictor) for group, predictor, _ in scans]
                jpeg = encode_lossless(width, height, precision, coded, restart, planes, rgb)
            split = "/".join("".join(str(c) for c in group) for group, _, _ in scans)
            _, predictor, point_transform = scans[0]
            name = f"{layout}_{width}x{height}_{precision}bit_split{split.replace('/', '-')}"
            if restart:
                name += f"_rst{restart}"
            record(name, (layout, precision, predictor, point_transform, restart, width, height), jpeg, split)

    with open(os.path.join(OUT_DIR, "data.bin"), "wb") as handle:
        handle.write(data)
    with open(os.path.join(OUT_DIR, "cases.tsv"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines) - 2} cases, {len(data)} bytes")


if __name__ == "__main__":
    main()
