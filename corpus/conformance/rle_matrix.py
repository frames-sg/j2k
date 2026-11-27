#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Regenerate the DICOM RLE Lossless matrix: real and synthetic RLE frames
(DICOM PS3.5 Annex G) with reference samples, for the j2k-tilecodec tests.

Run manually and commit the output. CI does not run it; tests read the
committed files. Requires pydicom 2.4.4 and numpy:

    python3 -m venv venv && venv/bin/pip install pydicom==2.4.4 numpy
    venv/bin/python -I corpus/conformance/rle_matrix.py

Cases:

- Real frames: the encapsulated RLE frames of the RLE test files bundled with
  pydicom 2.4.4 (pydicom's test file README says GDCM's gdcmconv encoded
  them), with pydicom's decode as the reference. Where pydicom also ships the
  uncompressed original (MR_small.dcm, rtdose.dcm), the script checks that
  the reference equals it. Frames that repeat byte for byte in another file
  are kept once.
- Synthetic frames: small deterministic images (8, 16 and 32 bits allocated,
  1, 3, 5 and 15 samples per pixel, odd widths, runs longer than 128 bytes,
  rows with no repeats, single pixels) encoded with pydicom's RLE encoder.

Every case also records pydicom's encoding of its reference samples, so the
tests can compare encoders byte for byte.

Output (in crates/j2k-tilecodec/tests/fixtures/dicom_rle/):

- cases.tsv  one line per case: name, source, rows, columns, samples per
             pixel, bits allocated, then offset:length in data.bin of the RLE
             frame, of the reference samples, and of pydicom's encoding.
- data.bin   the RLE frames, references and pydicom encodings, concatenated.
             References are native little-endian samples, color-by-pixel
             (Planar Configuration 0).
"""

import hashlib
import os
import sys

import numpy as np
import pydicom
from pydicom.data import get_testdata_file
from pydicom.encaps import generate_pixel_data_frame
from pydicom.encoders.native import _encode_frame
from pydicom.pixel_data_handlers.rle_handler import _rle_decode_frame

PYDICOM_VERSION = "2.4.4"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "..", "crates", "j2k-tilecodec", "tests", "fixtures", "dicom_rle")

# Bundled RLE files and the uncompressed original when pydicom ships one.
# The two-frame SC_rgb variants are left out to keep the fixtures small.
REAL_FILES = [
    ("MR_small_RLE.dcm", "MR_small.dcm"),
    ("rtdose_rle.dcm", "rtdose.dcm"),
    ("rtdose_rle_1frame.dcm", "rtdose_1frame.dcm"),
    ("SC_rgb_rle.dcm", None),
    ("SC_rgb_rle_16bit.dcm", None),
    ("SC_rgb_rle_32bit.dcm", None),
]


def lcg(seed):
    state = seed
    while True:
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        yield state >> 8


def runs_row(spec):
    """A row from (value, count) runs."""
    out = []
    for value, count in spec:
        out += [value] * count
    return out


def synthetic_cases():
    """(name, rows, columns, samples, bits, interleaved sample values)."""
    noise = lcg(20261120)
    cases = []

    cases.append(("gray8_1x1", 1, 1, 1, 8, [0x5A]))
    cases.append(("gray16_1x1", 1, 1, 1, 16, [0xA55A]))
    cases.append(("gray32_1x1", 1, 1, 1, 32, [0x80FF0102]))
    cases.append(("rgb8_1x1", 1, 1, 3, 8, [0x10, 0x20, 0x30]))

    # Odd width; a two-byte repeat between literals; equal values on both
    # sides of row boundaries, which must not join into one run.
    mixed = [
        [1, 2, 3, 3, 4, 5, 9],
        [9, 9, 9, 7, 7, 8, 8],
        [8, 0, 255, 255, 255, 255, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 6, 5, 6, 5, 5, 5],
    ]
    cases.append(("gray8_5x7_mixed", 5, 7, 1, 8, sum(mixed, [])))

    # Runs of 128, 129 (128 plus a one-byte remainder), 130, 256 and 257
    # bytes, and literal stretches longer than 128.
    long_runs = [
        runs_row([(7, 129), (8, 130), (9, 2), (10, 1), (11, 38)]),
        runs_row([(20, 256), (21, 1), (22, 1), (23, 42)]),
        runs_row([(30, 257), (31, 43)]),
        runs_row([(40, 128), (41, 128), (42, 44)]),
    ]
    cases.append(("gray8_4x300_long_runs", 4, 300, 1, 8, sum(long_runs, [])))

    # No two neighbours equal: literal runs of 128, 128 and 5 bytes per row.
    literal = [((x * 7) + 3 * y) % 251 for y in range(2) for x in range(261)]
    cases.append(("gray8_2x261_literal", 2, 261, 1, 8, literal))

    # Constant high bytes and varying low bytes, and the reverse.
    gray16 = [(y << 12) | (x * 977 + y) & 0xFFF for y in range(4) for x in range(9)]
    cases.append(("gray16_4x9_ramp", 4, 9, 1, 16, gray16))

    gray32 = [
        (0x01000000 * y) | (next(noise) & 0xFFFF) if x % 3 else 0xDEADBEEF
        for y in range(2)
        for x in range(13)
    ]
    cases.append(("gray32_2x13", 2, 13, 1, 32, gray32))

    rgb8 = []
    for y in range(3):
        for x in range(11):
            rgb8 += [(x * 23) & 0xFF, 200 if x < 6 else 17, next(noise) & 0xFF]
    cases.append(("rgb8_3x11", 3, 11, 3, 8, rgb8))

    rgb16 = []
    for y in range(3):
        for x in range(5):
            rgb16 += [x * 13107, 0x0100 * y + 1, next(noise) & 0xFFFF]
    cases.append(("rgb16_3x5", 3, 5, 3, 16, rgb16))

    rgb32 = []
    for y in range(2):
        for x in range(3):
            rgb32 += [0xFFFFFFFF - x, 0x00010000 * (y + 1), next(noise) & 0xFFFFFFFF]
    cases.append(("rgb32_2x3_twelve_segments", 2, 3, 3, 32, rgb32))

    five = [(s * 4099 + x * 31 + y * 7) & 0xFFFF for y in range(2) for x in range(3) for s in range(5)]
    cases.append(("samples5_16bit_2x3", 2, 3, 5, 16, five))

    fifteen = [(s * 17 + x) & 0xFF if s % 2 else 0x42 for y in range(2) for x in range(3) for s in range(15)]
    cases.append(("samples15_8bit_2x3_fifteen_segments", 2, 3, 15, 8, fifteen))

    return cases


def sample_dtype(bits):
    return {8: "<u1", 16: "<u2", 32: "<u4"}[bits]


def pydicom_encode(reference, rows, columns, samples, bits):
    return _encode_frame(
        reference,
        rows=rows,
        columns=columns,
        samples_per_pixel=samples,
        bits_allocated=bits,
        byteorder="<",
    )


def pydicom_decode(frame, rows, columns, samples, bits):
    """pydicom's decode, converted from its planar output to color-by-pixel."""
    planar = _rle_decode_frame(frame, rows, columns, samples, bits)
    values = np.frombuffer(bytes(planar), dtype=sample_dtype(bits))
    return values.reshape(samples, rows * columns).T.tobytes()


def check_round_trip(name, frame, reference, rows, columns, samples, bits):
    decoded = pydicom_decode(frame, rows, columns, samples, bits)
    if decoded != reference:
        sys.exit(f"error: {name}: pydicom does not decode the frame to the reference")


def real_cases():
    seen = set()
    for file_name, original_name in REAL_FILES:
        ds = pydicom.dcmread(get_testdata_file(file_name))
        if ds.file_meta.TransferSyntaxUID != pydicom.uid.RLELossless:
            sys.exit(f"error: {file_name} is not RLE Lossless")
        count = int(getattr(ds, "NumberOfFrames", 1) or 1)
        rows, columns = int(ds.Rows), int(ds.Columns)
        samples, bits = int(ds.SamplesPerPixel), int(ds.BitsAllocated)
        pixels = ds.pixel_array.reshape(count, rows, columns, samples)
        if original_name is not None:
            original = pydicom.dcmread(get_testdata_file(original_name)).pixel_array
            if not np.array_equal(original.reshape(pixels.shape), pixels):
                sys.exit(f"error: {file_name} does not decode to {original_name}")
        for index, frame in enumerate(generate_pixel_data_frame(ds.PixelData, count)):
            digest = hashlib.sha256(frame).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            reference = pixels[index].astype(sample_dtype(bits)).tobytes()
            stem = os.path.splitext(file_name)[0]
            name = stem if count == 1 else f"{stem}_frame{index}"
            yield name, f"{file_name}#{index}", rows, columns, samples, bits, bytes(frame), reference


def main():
    if pydicom.__version__ != PYDICOM_VERSION:
        sys.exit(f"error: needs pydicom {PYDICOM_VERSION}, found {pydicom.__version__}")
    os.makedirs(OUT_DIR, exist_ok=True)
    data = bytearray()
    blobs = {}
    lines = [
        f"# Generated by corpus/conformance/rle_matrix.py with pydicom {pydicom.__version__}"
        f" and numpy {np.__version__}",
        "# name\tsource\trows\tcolumns\tsamples\tbits\trle\treference\tpydicom_rle",
    ]

    def append(blob):
        if blob in blobs:
            return blobs[blob]
        field = f"{len(data)}:{len(blob)}"
        data.extend(blob)
        blobs[blob] = field
        return field

    def add(name, source, rows, columns, samples, bits, frame, reference):
        check_round_trip(name, frame, reference, rows, columns, samples, bits)
        encoded = pydicom_encode(reference, rows, columns, samples, bits)
        check_round_trip(name + " (pydicom encoding)", encoded, reference, rows, columns, samples, bits)
        fields = [name, source, str(rows), str(columns), str(samples), str(bits)]
        lines.append("\t".join(fields + [append(frame), append(reference), append(encoded)]))

    for case in real_cases():
        add(*case)

    for name, rows, columns, samples, bits, values in synthetic_cases():
        if len(values) != rows * columns * samples:
            sys.exit(f"error: {name}: {len(values)} values for a {rows}x{columns}x{samples} frame")
        reference = np.array(values, dtype=np.uint64).astype(sample_dtype(bits)).tobytes()
        frame = pydicom_encode(reference, rows, columns, samples, bits)
        add(name, "synthetic", rows, columns, samples, bits, frame, reference)

    with open(os.path.join(OUT_DIR, "data.bin"), "wb") as handle:
        handle.write(data)
    with open(os.path.join(OUT_DIR, "cases.tsv"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines) - 2} cases, {len(data)} bytes")


if __name__ == "__main__":
    main()
