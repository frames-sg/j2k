#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Regenerate the JPEG-LS matrix: CharLS-encoded codestreams with CharLS's own
decode as the reference.

Run manually and commit the output. CI does not run it; tests read the
committed files.

Each case is a small synthetic image (no patient data) encoded by CharLS
through its C API: flat, gradient, smooth, noise, and mixed content at 2 to 16 bits,
1 to 5 components, every interleave mode, NEAR 0 to 7, and non-default preset
coding parameters. CharLS cannot write restart markers, so the restart cases
are encoded by j2k-jpegls (`examples/encode_raw.rs`) and checked by CharLS.
The interleaved two-component cases are j2k-jpegls output too: CharLS 2.4
does not code those scans (see `charls_supports`).
Every case is also encoded by j2k-jpegls, and the script stops unless CharLS
decodes that output to the expected samples. With `--dicom FILE`, the
JPEG-LS codestream in a DICOM file's encapsulated pixel data is added as a
real-world case.

Output (in crates/j2k-jpegls/fixtures/jpegls_matrix/):

- cases.tsv  one line per case: name, source (charls, j2k-jpegls, or
             dicom), CharLS encoding options, width, height, components, bits
             per sample, interleave mode, NEAR, restart interval, preset
             parameters, then offset:length of the input samples, the
             codestream, and the reference samples in data.bin.
- data.bin   the samples and codestreams, concatenated with duplicates
             shared. Samples are interleaved by pixel; above 8 bits they are
             little-endian 16-bit.

Requires the CharLS 2.x shared library: pass --charls PATH, set
CHARLS_LIBRARY, install it where ctypes can find it, or install the Python
imagecodecs package, which bundles it. Requires cargo to build the example.
"""

import argparse
import ctypes
import ctypes.util
import glob
import os
import subprocess
import struct
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(ROOT, "crates", "j2k-jpegls", "fixtures", "jpegls_matrix")

CHARLS_OPTIONS_NONE = 0
# CharLS's default options write an LSE segment with the default parameters
# for more than 12 bits per sample (for Java Advanced Imaging).
CHARLS_OPTIONS_JAI = 4

# name, pattern, width, height, components, bits, ilv, near, restart,
# preset (MAXVAL, T1, T2, T3, RESET; 0 is default) or None, CharLS options.
# A restart interval selects the j2k-jpegls encoder.
CASES = [
    ("g8_flat", "flat", 37, 23, 1, 8, 0, 0, 0, None, 0),
    ("g8_gradient", "gradient", 37, 23, 1, 8, 0, 0, 0, None, 0),
    ("g8_noise", "noise", 37, 23, 1, 8, 0, 0, 0, None, 0),
    ("g8_mixed", "mixed", 41, 29, 1, 8, 0, 0, 0, None, 0),
    ("g2_mixed", "mixed", 33, 17, 1, 2, 0, 0, 0, None, 0),
    ("g5_mixed", "mixed", 33, 17, 1, 5, 0, 0, 0, None, 0),
    ("g10_mixed", "mixed", 37, 23, 1, 10, 0, 0, 0, None, 0),
    ("g12_mixed", "mixed", 37, 23, 1, 12, 0, 0, 0, None, 0),
    ("g12_noise", "noise", 37, 23, 1, 12, 0, 0, 0, None, 0),
    ("g13_gradient_jai", "gradient", 37, 23, 1, 13, 0, 0, 0, None, CHARLS_OPTIONS_JAI),
    ("g16_mixed", "mixed", 37, 23, 1, 16, 0, 0, 0, None, 0),
    ("g16_mixed_jai", "mixed", 37, 23, 1, 16, 0, 0, 0, None, CHARLS_OPTIONS_JAI),
    ("g16_noise", "noise", 37, 23, 1, 16, 0, 0, 0, None, 0),
    ("g8_1x1", "noise", 1, 1, 1, 8, 0, 0, 0, None, 0),
    ("g8_1x17", "mixed", 1, 17, 1, 8, 0, 0, 0, None, 0),
    ("g8_29x1", "mixed", 29, 1, 1, 8, 0, 0, 0, None, 0),
    ("g16_1x13", "noise", 1, 13, 1, 16, 0, 0, 0, None, 0),
    ("g12_17x1", "gradient", 17, 1, 1, 12, 0, 0, 0, None, 0),
    ("g8_mixed_near1", "mixed", 41, 29, 1, 8, 0, 1, 0, None, 0),
    ("g8_mixed_near3", "mixed", 41, 29, 1, 8, 0, 3, 0, None, 0),
    ("g8_mixed_near7", "mixed", 41, 29, 1, 8, 0, 7, 0, None, 0),
    ("g8_noise_near3", "noise", 37, 23, 1, 8, 0, 3, 0, None, 0),
    ("g2_mixed_near1", "mixed", 33, 17, 1, 2, 0, 1, 0, None, 0),
    ("g10_noise_near3", "noise", 37, 23, 1, 10, 0, 3, 0, None, 0),
    ("g12_mixed_near3", "mixed", 37, 23, 1, 12, 0, 3, 0, None, 0),
    ("g16_mixed_near7", "mixed", 37, 23, 1, 16, 0, 7, 0, None, 0),
    ("rgb8_mixed_ilv0", "mixed", 37, 23, 3, 8, 0, 0, 0, None, 0),
    ("rgb8_mixed_ilv1", "mixed", 37, 23, 3, 8, 1, 0, 0, None, 0),
    ("rgb8_mixed_ilv2", "mixed", 37, 23, 3, 8, 2, 0, 0, None, 0),
    ("rgb8_flat_ilv2", "flat", 37, 23, 3, 8, 2, 0, 0, None, 0),
    ("rgb8_mixed_ilv0_near3", "mixed", 37, 23, 3, 8, 0, 3, 0, None, 0),
    ("rgb8_mixed_ilv1_near3", "mixed", 37, 23, 3, 8, 1, 3, 0, None, 0),
    ("rgb8_mixed_ilv2_near3", "mixed", 37, 23, 3, 8, 2, 3, 0, None, 0),
    ("rgb12_mixed_ilv0", "mixed", 29, 19, 3, 12, 0, 0, 0, None, 0),
    ("rgb12_mixed_ilv1", "mixed", 29, 19, 3, 12, 1, 0, 0, None, 0),
    ("rgb12_mixed_ilv2", "mixed", 29, 19, 3, 12, 2, 0, 0, None, 0),
    ("rgb16_gradient_ilv2", "gradient", 29, 19, 3, 16, 2, 0, 0, None, 0),
    ("rgb10_noise_ilv2_near1", "noise", 29, 19, 3, 10, 2, 1, 0, None, 0),
    ("rgb8_1x19_ilv2", "mixed", 1, 19, 3, 8, 2, 0, 0, None, 0),
    ("rgb8_23x1_ilv1", "mixed", 23, 1, 3, 8, 1, 0, 0, None, 0),
    ("c4_8_mixed_ilv1", "mixed", 29, 19, 4, 8, 1, 0, 0, None, 0),
    ("c4_8_mixed_ilv2", "mixed", 29, 19, 4, 8, 2, 0, 0, None, 0),
    ("c2_8_mixed_ilv1", "mixed", 29, 19, 2, 8, 1, 0, 0, None, 0),
    ("c2_8_mixed_ilv2", "mixed", 29, 19, 2, 8, 2, 0, 0, None, 0),
    ("c5_8_mixed_ilv0", "mixed", 23, 13, 5, 8, 0, 0, 0, None, 0),
    ("g8_preset_thresholds", "mixed", 41, 29, 1, 8, 0, 0, 0, (0, 4, 8, 22, 32), 0),
    ("g8_preset_reset3", "mixed", 41, 29, 1, 8, 0, 2, 0, (0, 0, 0, 0, 3), 0),
    # CharLS 2.4 writes a MAXVAL below 2^P - 1 into the LSE segment but does
    # not code with it as T.87 A.2.1 and A.4.2 specify (lossless: errors
    # modulo 2^P, predictions clamped to 2^P - 1). These two cases use content
    # and parameters for which both give the same codestream.
    ("g10_maxval1000_smooth", "smooth", 37, 23, 1, 10, 0, 0, 0, (1000, 0, 0, 0, 0), 0),
    ("g12_maxval4000_near1", "mixed", 37, 23, 1, 12, 0, 1, 0, (4000, 0, 0, 0, 0), 0),
    ("g8_mixed_restart5", "mixed", 41, 29, 1, 8, 0, 0, 5, None, 0),
    ("g16_mixed_restart1", "mixed", 37, 23, 1, 16, 0, 0, 1, None, 0),
    ("g12_mixed_near3_restart6", "mixed", 37, 23, 1, 12, 0, 3, 6, None, 0),
    ("rgb8_mixed_ilv0_restart7", "mixed", 37, 23, 3, 8, 0, 0, 7, None, 0),
    ("rgb8_mixed_ilv1_restart3", "mixed", 37, 23, 3, 8, 1, 0, 3, None, 0),
    ("rgb8_flat_ilv2_restart4", "flat", 37, 23, 3, 8, 2, 0, 4, None, 0),
]


class FrameInfo(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("bits_per_sample", ctypes.c_int32),
        ("component_count", ctypes.c_int32),
    ]


class PresetParameters(ctypes.Structure):
    _fields_ = [
        ("maximum_sample_value", ctypes.c_int32),
        ("threshold1", ctypes.c_int32),
        ("threshold2", ctypes.c_int32),
        ("threshold3", ctypes.c_int32),
        ("reset_value", ctypes.c_int32),
    ]


def load_charls(path):
    candidates = [path, os.environ.get("CHARLS_LIBRARY"), ctypes.util.find_library("charls")]
    try:
        import imagecodecs  # noqa: PLC0415 - optional source of a bundled CharLS

        bundled = os.path.join(os.path.dirname(imagecodecs.__file__), ".dylibs", "libcharls*")
        candidates += sorted(glob.glob(bundled))
        bundled = os.path.join(os.path.dirname(imagecodecs.__file__), "..", "imagecodecs.libs", "libcharls*")
        candidates += sorted(glob.glob(bundled))
    except ImportError:
        pass
    for candidate in candidates:
        if not candidate:
            continue
        try:
            lib = ctypes.CDLL(candidate)
        except OSError:
            continue
        lib.charls_get_version_string.restype = ctypes.c_char_p
        lib.charls_get_error_message.restype = ctypes.c_char_p
        lib.charls_jpegls_encoder_create.restype = ctypes.c_void_p
        lib.charls_jpegls_decoder_create.restype = ctypes.c_void_p
        return lib
    sys.exit("error: CharLS not found; pass --charls PATH or set CHARLS_LIBRARY")


class CharlsError(Exception):
    pass


def check(lib, err, what):
    if err != 0:
        raise CharlsError(f"CharLS {what}: {lib.charls_get_error_message(err).decode()}")


def lcg(seed):
    state = seed & 0xFFFFFFFF
    while True:
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        yield state >> 8


def pattern(kind, width, height, components, bits):
    """Deterministic integer-only test content, interleaved by pixel."""
    maxval = (1 << bits) - 1
    noise = lcg(width * 131 + height * 7 + components * 3 + bits)
    out = []
    for y in range(height):
        for x in range(width):
            for c in range(components):
                if kind == "flat":
                    spot = (x * 7 + y * 5 + c * 3) % 61 == 0
                    v = maxval * 2 // 3 if spot else maxval // 3 + (1 if (x // 9 + c) % 5 == 0 else 0)
                elif kind == "gradient":
                    v = (x * (maxval + 1) // max(width, 1) + y * (maxval + 1) // max(2 * height, 1) + c * 11) % (maxval + 1)
                elif kind == "smooth":
                    v = min(x * 13 + y * 11 + c * 17, maxval)
                elif kind == "noise":
                    v = next(noise) % (maxval + 1)
                else:  # mixed: flat left third, gradient middle, noise right, a stripe
                    third = max(width // 3, 1)
                    if y % 11 == 5:
                        v = maxval if (x + c) % 2 else 0
                    elif x < third:
                        v = (maxval // 4) * (1 + c % 3) // 2
                    elif x < 2 * third:
                        v = ((x - third) * maxval // max(third, 1) + y * 3 + c) % (maxval + 1)
                    else:
                        v = (maxval // 2 + next(noise) % max(maxval // 8, 2)) % (maxval + 1)
                out.append(v)
    return out


def limit_to_maxval(samples, preset):
    if preset and preset[0]:
        return [v % (preset[0] + 1) for v in samples]
    return samples


def pack(samples, bits):
    if bits > 8:
        return struct.pack(f"<{len(samples)}H", *samples)
    return bytes(samples)


def to_planar(data, components, sample_bytes):
    pixel = components * sample_bytes
    planes = [bytearray() for _ in range(components)]
    for offset in range(0, len(data), pixel):
        for c in range(components):
            start = offset + c * sample_bytes
            planes[c] += data[start:start + sample_bytes]
    return b"".join(planes)


def to_interleaved(data, components, sample_bytes):
    plane = len(data) // components
    out = bytearray()
    for offset in range(0, plane, sample_bytes):
        for c in range(components):
            start = c * plane + offset
            out += data[start:start + sample_bytes]
    return bytes(out)


def charls_encode(lib, data, width, height, components, bits, ilv, near, preset, options):
    encoder = ctypes.c_void_p(lib.charls_jpegls_encoder_create())
    try:
        frame = FrameInfo(width, height, bits, components)
        check(lib, lib.charls_jpegls_encoder_set_frame_info(encoder, ctypes.byref(frame)), "frame info")
        check(lib, lib.charls_jpegls_encoder_set_near_lossless(encoder, near), "NEAR")
        check(lib, lib.charls_jpegls_encoder_set_interleave_mode(encoder, ilv), "interleave mode")
        check(lib, lib.charls_jpegls_encoder_set_encoding_options(encoder, options), "options")
        if preset:
            params = PresetParameters(*preset)
            check(lib, lib.charls_jpegls_encoder_set_preset_coding_parameters(encoder, ctypes.byref(params)), "preset")
        out = ctypes.create_string_buffer(len(data) * 2 + 4096)
        check(lib, lib.charls_jpegls_encoder_set_destination_buffer(encoder, out, ctypes.c_size_t(len(out))), "destination")
        source = to_planar(data, components, 2 if bits > 8 else 1) if ilv == 0 and components > 1 else data
        src = ctypes.create_string_buffer(source, len(source))
        check(lib, lib.charls_jpegls_encoder_encode_from_buffer(encoder, src, ctypes.c_size_t(len(source)), ctypes.c_uint32(0)), "encode")
        written = ctypes.c_size_t()
        check(lib, lib.charls_jpegls_encoder_get_bytes_written(encoder, ctypes.byref(written)), "bytes written")
        return out.raw[:written.value]
    finally:
        lib.charls_jpegls_encoder_destroy(encoder)


def first_scan(codestream):
    """(ILV, NEAR, restart interval) of the first scan."""
    pos, restart = 2, 0
    while pos + 4 <= len(codestream):
        marker = codestream[pos + 1]
        length = codestream[pos + 2] << 8 | codestream[pos + 3]
        if marker == 0xDD:
            restart = int.from_bytes(codestream[pos + 4:pos + 2 + length], "big")
        if marker == 0xDA:
            count = codestream[pos + 4]
            near = codestream[pos + 4 + 2 * count + 1]
            return codestream[pos + 4 + 2 * count + 2], near, restart
        pos += 2 + length
    sys.exit("error: no SOS in codestream")


def charls_decode(lib, codestream):
    """Decode with CharLS; returns (frame info, samples interleaved by pixel)."""
    decoder = ctypes.c_void_p(lib.charls_jpegls_decoder_create())
    try:
        src = ctypes.create_string_buffer(codestream, len(codestream))
        check(lib, lib.charls_jpegls_decoder_set_source_buffer(decoder, src, ctypes.c_size_t(len(codestream))), "source")
        check(lib, lib.charls_jpegls_decoder_read_header(decoder), "read header")
        frame = FrameInfo()
        check(lib, lib.charls_jpegls_decoder_get_frame_info(decoder, ctypes.byref(frame)), "frame info")
        size = ctypes.c_size_t()
        check(lib, lib.charls_jpegls_decoder_get_destination_size(decoder, ctypes.c_uint32(0), ctypes.byref(size)), "size")
        out = ctypes.create_string_buffer(size.value)
        check(lib, lib.charls_jpegls_decoder_decode_to_buffer(decoder, out, size, ctypes.c_uint32(0)), "decode")
        data = out.raw[:size.value]
    finally:
        lib.charls_jpegls_decoder_destroy(decoder)
    sample_bytes = 2 if frame.bits_per_sample > 8 else 1
    if frame.component_count > 1 and first_scan(codestream)[0] == 0:
        data = to_interleaved(data, frame.component_count, sample_bytes)
    return frame, data


def build_example():
    subprocess.run(["cargo", "build", "--quiet", "--release", "-p", "j2k-jpegls", "--example", "encode_raw"], cwd=ROOT, check=True)
    target = os.environ.get("CARGO_TARGET_DIR", os.path.join(ROOT, "target"))
    return os.path.join(target, "release", "examples", "encode_raw")


def j2k_encode(example, work, data, width, height, components, bits, ilv, near, restart, preset):
    source, output = os.path.join(work, "input.raw"), os.path.join(work, "output.jls")
    with open(source, "wb") as handle:
        handle.write(data)
    preset_arg = ",".join(str(v) for v in preset) if preset else "-"
    args = [example, source, output, str(width), str(height), str(components), str(bits), str(near), str(ilv), str(restart), preset_arg]
    subprocess.run(args, check=True)
    with open(output, "rb") as handle:
        return handle.read()


def dicom_codestream(path):
    """The first frame of an Explicit VR Little Endian DICOM file's
    encapsulated pixel data."""
    buf = open(path, "rb").read()
    if buf[128:132] != b"DICM":
        sys.exit(f"error: {path} is not a DICOM file")
    pos = 132
    long_vrs = {b"OB", b"OW", b"OF", b"SQ", b"UT", b"UN"}
    while pos + 8 <= len(buf):
        group, elem = struct.unpack_from("<HH", buf, pos)
        vr = buf[pos + 4:pos + 6]
        if vr in long_vrs:
            length, start = struct.unpack_from("<I", buf, pos + 8)[0], pos + 12
        else:
            length, start = struct.unpack_from("<H", buf, pos + 6)[0], pos + 8
        if (group, elem) == (0x7FE0, 0x0010):
            if length != 0xFFFFFFFF:
                sys.exit("error: pixel data is not encapsulated")
            fragments, p = [], start
            while True:
                g, e, item_len = struct.unpack_from("<HHI", buf, p)
                if (g, e) == (0xFFFE, 0xE0DD):
                    break
                fragments.append(buf[p + 8:p + 8 + item_len])
                p += 8 + item_len
            codestream = b"".join(fragments[1:])  # item 0 is the basic offset table
            end = codestream.rfind(b"\xff\xd9")
            return codestream[:end + 2]
        pos = start + length
    sys.exit("error: no pixel data")


def check_within_near(name, expected, actual, bits, near):
    if len(expected) != len(actual):
        sys.exit(f"error: {name}: decoded length {len(actual)}, expected {len(expected)}")
    fmt = "<{}H" if bits > 8 else "{}B"
    count = len(expected) // (2 if bits > 8 else 1)
    a = struct.unpack(fmt.format(count), expected)
    b = struct.unpack(fmt.format(count), actual)
    worst = max((abs(x - y) for x, y in zip(a, b)), default=0)
    if worst > near:
        sys.exit(f"error: {name}: decoded error {worst} exceeds NEAR {near}")


def charls_supports(components, ilv):
    """CharLS 2.4 rejects interleaved two-component scans when decoding, and
    its encoder writes them as an all-ones bit stream. Those cases use the
    j2k-jpegls codestream with the input as the reference, so they only guard
    against regressions."""
    return not (components == 2 and ilv != 0)


def add_case(lib, example, work, case, append):
    """Encode one case; returns its cases.tsv line and whether the j2k-jpegls
    codestream is identical to CharLS's."""
    name, kind, width, height, components, bits, ilv, near, restart, preset, options = case
    raw = pack(limit_to_maxval(pattern(kind, width, height, components, bits), preset), bits)
    ours = j2k_encode(example, work, raw, width, height, components, bits, ilv, near, restart, preset)
    if not charls_supports(components, ilv):
        if near != 0:
            sys.exit(f"error: {name}: no reference for a near-lossless two-component scan")
        source, codestream, reference = "j2k-jpegls", ours, raw
    else:
        _, ours_decoded = charls_decode(lib, ours)
        check_within_near(name + " (j2k-jpegls output)", raw, ours_decoded, bits, near)
        if restart:
            source, codestream = "j2k-jpegls", ours
        else:
            source = "charls"
            codestream = charls_encode(lib, raw, width, height, components, bits, ilv, near, preset, options)
        _, reference = charls_decode(lib, codestream)
    check_within_near(name, raw, reference, bits, near)
    if near == 0 and reference != raw:
        sys.exit(f"error: {name}: lossless decode differs from the input")
    preset_field = ",".join(str(v) for v in preset) if preset else "-"
    fields = [name, source, str(options), str(width), str(height), str(components), str(bits), str(ilv), str(near), str(restart), preset_field]
    line = "\t".join(fields + [append(raw), append(codestream), append(reference)])
    return line, source == "charls" and codestream == ours


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--charls", help="path to the CharLS shared library")
    parser.add_argument("--dicom", help="DICOM file with JPEG-LS pixel data to add as a case")
    args = parser.parse_args()

    lib = load_charls(args.charls)
    version = lib.charls_get_version_string().decode()
    if not version.startswith("2."):
        sys.exit(f"error: CharLS 2.x required, found {version}")
    example = build_example()
    data = bytearray()
    offsets = {}
    lines = [
        f"# Generated by corpus/conformance/jpegls_matrix.py with CharLS {version}",
        "# name\tsource\tcharls_options\twidth\theight\tcomponents\tbits\tilv\tnear\trestart\tpreset\tinput\tcodestream\treference",
    ]

    def append(blob):
        blob = bytes(blob)
        if blob not in offsets:
            offsets[blob] = len(data)
            data.extend(blob)
        return f"{offsets[blob]}:{len(blob)}"

    differing = []
    with tempfile.TemporaryDirectory() as work:
        for case in CASES:
            try:
                line, identical = add_case(lib, example, work, case, append)
            except CharlsError as error:
                sys.exit(f"error: {case[0]}: {error}")
            lines.append(line)
            if not case[8] and charls_supports(case[4], case[6]) and not identical:
                differing.append(case[0])

        if args.dicom:
            codestream = dicom_codestream(args.dicom)
            try:
                frame, reference = charls_decode(lib, codestream)
            except CharlsError as error:
                sys.exit(f"error: {args.dicom}: {error}")
            name = os.path.splitext(os.path.basename(args.dicom))[0]
            ilv, near, restart = first_scan(codestream)
            fields = [name, "dicom", "-", str(frame.width), str(frame.height), str(frame.component_count), str(frame.bits_per_sample), str(ilv), str(near), str(restart), "-"]
            lines.append("\t".join(fields + [append(reference), append(codestream), append(reference)]))

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "data.bin"), "wb") as handle:
        handle.write(data)
    with open(os.path.join(OUT_DIR, "cases.tsv"), "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines) - 2} cases, {len(data)} bytes")
    if differing:
        print("j2k-jpegls output differs from CharLS in: " + ", ".join(differing))


if __name__ == "__main__":
    main()
