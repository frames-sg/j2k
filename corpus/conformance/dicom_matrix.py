#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Regenerate the DICOM interoperability matrix: JPEG codestreams as DCMTK
writes them into DICOM, with DCMTK's own decode as the reference.

Run manually (generate.sh invokes it when DCMTK is installed) and commit the
output. CI does not run it; tests read the committed files.

Each case is a synthetic phantom (no patient data) written as an
uncompressed Explicit VR Little Endian DICOM image, compressed with
`dcmcjpeg`, and decompressed with `dcmdjpeg`. The JPEG codestream is taken
from the encapsulated pixel data. Cases cover what DICOM archives hold:
lossless first-order prediction (SV1) and other predictors at 8 to 16 bits
stored, 12-bit extended and progressive DCT, and RGB.

Output (in crates/j2k-test-support/fixtures/dicom_matrix/):

- cases.tsv  one line per case: name, dcmcjpeg options, JPEG precision,
             channels, width, height, then offset:length of the JPEG and of
             the reference in data.bin.
- data.bin   the JPEG inputs and DCMTK references, concatenated. References
             above 8 bits are big-endian 16-bit samples.

Requires DCMTK `dcmcjpeg`, `dcmdjpeg` on PATH.
"""

import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "..", "crates", "j2k-test-support", "fixtures", "dicom_matrix")

# name, bits stored, samples per pixel, width, height, dcmcjpeg options
CASES = [
    ("ct12_sv1", 12, 1, 61, 47, ["+e1"]),
    ("ct12_p7", 12, 1, 61, 47, ["+el", "+sv", "7"]),
    ("ct10_p4", 10, 1, 53, 41, ["+el", "+sv", "4"]),
    ("mr14_sv1", 14, 1, 49, 37, ["+e1"]),
    ("cr16_p6", 16, 1, 57, 43, ["+el", "+sv", "6"]),
    ("mono8_sv1", 8, 1, 45, 33, ["+e1"]),
    ("rgb8_sv1", 8, 3, 45, 33, ["+e1"]),
    ("ct12_extended", 12, 1, 61, 47, ["+ee"]),
    ("ct12_progressive", 12, 1, 61, 47, ["+ep"]),
    ("rgb8_baseline", 8, 3, 45, 33, ["+eb"]),
    ("rgb8_progressive", 8, 3, 45, 33, ["+ep"]),
]

LONG_VRS = {"OB", "OW", "OF", "SQ", "UT", "UN"}


def element(group, elem, vr, value):
    if isinstance(value, str):
        value = value.encode()
        value += (b"\0" if vr == "UI" else b" ") * (len(value) % 2)
    tag = struct.pack("<HH", group, elem)
    if vr in LONG_VRS:
        return tag + vr.encode() + b"\0\0" + struct.pack("<I", len(value)) + value
    return tag + vr.encode() + struct.pack("<H", len(value)) + value


def us(value):
    return struct.pack("<H", value)


def phantom(width, height, bits, channels):
    """Deterministic CT-like rings plus gradients and low-bit noise."""
    maxval = (1 << bits) - 1
    out = []
    for y in range(height):
        for x in range(width):
            r = math.hypot(x - width / 2, y - height / 2) / (min(width, height) / 2)
            v = 0.5 + 0.35 * math.cos(r * 9) * math.exp(-r) + 0.1 * math.sin(x * 0.07 + y * 0.03)
            base = min(int(max(0.0, min(1.0, v)) * maxval) ^ ((x * 7 + y * 3) & 3), maxval)
            if channels == 1:
                out.append(base)
            else:
                out += [base, (x * 255 // max(width - 1, 1)) & maxval, (y * 9 + x) & maxval]
    return out


def write_dicom(path, width, height, bits, channels):
    sop_class = "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture Image Storage
    instance = f"1.2.826.0.1.3680043.2.1125.1.{width}.{height}.{bits}.{channels}"
    meta = (
        element(2, 1, "OB", b"\0\1")
        + element(2, 2, "UI", sop_class)
        + element(2, 3, "UI", instance)
        + element(2, 0x10, "UI", "1.2.840.10008.1.2.1")
        + element(2, 0x12, "UI", "1.2.826.0.1.3680043.2.1125.99")
    )
    meta = element(2, 0, "UL", struct.pack("<I", len(meta))) + meta
    dataset = element(8, 0x16, "UI", sop_class) + element(8, 0x18, "UI", instance)
    dataset += element(8, 0x60, "CS", "OT")
    dataset += element(0x10, 0x10, "PN", "Synthetic^Phantom") + element(0x10, 0x20, "LO", "SYN0001")
    dataset += element(0x20, 0x0D, "UI", instance + ".1") + element(0x20, 0x0E, "UI", instance + ".2")
    dataset += element(0x28, 2, "US", us(channels))
    dataset += element(0x28, 4, "CS", "RGB" if channels == 3 else "MONOCHROME2")
    if channels == 3:
        dataset += element(0x28, 6, "US", us(0))
    allocated = 8 if bits <= 8 else 16
    dataset += element(0x28, 0x10, "US", us(height)) + element(0x28, 0x11, "US", us(width))
    dataset += element(0x28, 0x100, "US", us(allocated)) + element(0x28, 0x101, "US", us(bits))
    dataset += element(0x28, 0x102, "US", us(bits - 1)) + element(0x28, 0x103, "US", us(0))
    pixels = phantom(width, height, bits, channels)
    data = bytes(pixels) if allocated == 8 else struct.pack(f"<{len(pixels)}H", *pixels)
    dataset += element(0x7FE0, 0x10, "OB" if allocated == 8 else "OW", data + b"\0" * (len(data) % 2))
    with open(path, "wb") as handle:
        handle.write(b"\0" * 128 + b"DICM" + meta + dataset)


def dataset_elements(path):
    """(group, element, value bytes or fragment list) of a DICOM file."""
    buf = open(path, "rb").read()
    meta_len = struct.unpack_from("<I", buf, 132 + 8)[0]
    pos = 132 + 12 + meta_len
    while pos + 8 <= len(buf):
        group, elem = struct.unpack_from("<HH", buf, pos)
        vr = buf[pos + 4:pos + 6].decode("latin1")
        if vr in LONG_VRS:
            length, start = struct.unpack_from("<I", buf, pos + 8)[0], pos + 12
        else:
            length, start = struct.unpack_from("<H", buf, pos + 6)[0], pos + 8
        if length == 0xFFFFFFFF:
            fragments, p = [], start
            while True:
                g, e, item_len = struct.unpack_from("<HHI", buf, p)
                if (g, e) == (0xFFFE, 0xE0DD):
                    break
                fragments.append(buf[p + 8:p + 8 + item_len])
                p += 8 + item_len
            yield group, elem, fragments
            return
        yield group, elem, buf[start:start + length]
        pos = start + length


def element_value(path, group, elem):
    for g, e, value in dataset_elements(path):
        if (g, e) == (group, elem):
            return value
    return None


def main():
    for tool in ("dcmcjpeg", "dcmdjpeg"):
        if shutil.which(tool) is None:
            sys.exit(f"error: DCMTK {tool} not on PATH")
    probe = subprocess.run(["dcmcjpeg", "--version"], capture_output=True, text=True, check=False)
    version = next((line.strip(" $") for line in probe.stdout.splitlines() if line.strip()), "DCMTK")
    os.makedirs(OUT_DIR, exist_ok=True)
    data = bytearray()
    lines = [
        f"# Generated by corpus/conformance/dicom_matrix.py with {version}",
        "# name\toptions\tprecision\tchannels\twidth\theight\tjpeg\treference",
    ]

    def append(blob):
        start = len(data)
        data.extend(blob)
        return f"{start}:{len(blob)}"

    with tempfile.TemporaryDirectory() as work:
        for name, bits, channels, width, height, options in CASES:
            source, compressed, decoded = (os.path.join(work, f"{name}{s}.dcm") for s in ("", "_jpeg", "_raw"))
            write_dicom(source, width, height, bits, channels)
            subprocess.run(["dcmcjpeg", *options, source, compressed], check=True)
            subprocess.run(["dcmdjpeg", compressed, decoded], check=True)
            fragments = element_value(compressed, 0x7FE0, 0x10)
            jpeg = b"".join(fragments[1:])  # item 0 is the basic offset table
            if element_value(decoded, 0x28, 6) not in (None, us(0)):
                sys.exit(f"error: {name}: DCMTK decoded to planar color")
            reference = element_value(decoded, 0x7FE0, 0x10)
            sof = next(i for i in range(len(jpeg) - 1) if jpeg[i] == 0xFF and jpeg[i + 1] in (0xC0, 0xC1, 0xC2, 0xC3))
            precision = jpeg[sof + 4]
            if precision > 8 or bits > 8:
                reference = b"".join(reference[i:i + 2][::-1] for i in range(0, len(reference) - 1, 2))
            expected = width * height * channels * (2 if bits > 8 else 1)
            if len(reference) < expected:
                sys.exit(f"error: {name}: reference has {len(reference)} bytes, expected {expected}")
            fields = [name, " ".join(options), str(precision), str(channels), str(width), str(height)]
            lines.append("\t".join(fields + [append(jpeg), append(reference[:expected])]))

    with open(os.path.join(OUT_DIR, "data.bin"), "wb") as handle:
        handle.write(data)
    with open(os.path.join(OUT_DIR, "cases.tsv"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines) - 2} cases, {len(data)} bytes")


if __name__ == "__main__":
    main()
