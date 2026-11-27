# SPDX-License-Identifier: MIT OR Apache-2.0

"""JPEG-LS and DICOM RLE frames through the Python binding.

Run after installing the wheel: python -m unittest discover -s crates/j2k-python/tests
"""

import pathlib
import unittest
import warnings

import numpy as np

import j2k


REPO = pathlib.Path(__file__).resolve().parents[3]
JPEGLS_FIXTURES = REPO / "crates/j2k-jpegls/fixtures/jpegls_matrix"
RLE_FIXTURES = REPO / "crates/j2k-tilecodec/tests/fixtures/dicom_rle"


def fixture_cases(directory):
    """Rows of cases.tsv, with offset:length fields resolved against data.bin."""
    data = (directory / "data.bin").read_bytes()
    lines = (directory / "cases.tsv").read_text().splitlines()
    header = next(line for line in lines if line.startswith("# name"))[2:].split("\t")
    for line in lines:
        if line.startswith("#"):
            continue
        row = dict(zip(header, line.split("\t")))
        for key, value in row.items():
            if ":" in value and value.replace(":", "").isdigit():
                start, length = map(int, value.split(":"))
                row[key] = data[start:start + length]
        yield row


def samples(width, height, components, bits, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.indices((height, width))
    base = (x * 13 + y * 7)[..., None] + np.arange(components) * 29
    noisy = base + rng.integers(0, 9, size=base.shape)
    dtype = "u1" if bits <= 8 else "<u2"
    return (noisy % (1 << bits)).astype(dtype)


class JpegLsTests(unittest.TestCase):
    def test_round_trips_lossless_across_depths_and_interleave_modes(self):
        for bits, components, interleave, restart in [
            (8, 1, "none", None),
            (12, 1, "none", 5),
            (16, 1, "none", None),
            (8, 3, "none", None),
            (8, 3, "line", None),
            (10, 3, "sample", 7),
        ]:
            with self.subTest(bits=bits, components=components, interleave=interleave):
                pixels = samples(37, 23, components, bits)
                encoded = j2k.encode_jpegls(
                    pixels.tobytes(), 37, 23, components=components, bit_depth=bits,
                    interleave=interleave, restart_interval=restart,
                )
                info = j2k.inspect_jpegls(encoded)
                self.assertEqual(
                    (info["width"], info["height"], info["components"], info["bit_depth"]),
                    (37, 23, components, bits),
                )
                self.assertEqual(info["near"], 0)
                self.assertEqual(info["interleave"], interleave if components > 1 else "none")
                self.assertEqual(info["restart_interval"], restart)
                self.assertEqual(j2k.decode_jpegls(encoded), pixels.tobytes())

    def test_near_lossless_stays_within_near(self):
        pixels = samples(40, 30, 1, 12)
        encoded = j2k.encode_jpegls(pixels.tobytes(), 40, 30, bit_depth=12, near=3)
        self.assertEqual(j2k.inspect_jpegls(encoded)["near"], 3)
        decoded = np.frombuffer(j2k.decode_jpegls(encoded), dtype="<u2").reshape(pixels.shape)
        error = np.abs(decoded.astype(int) - pixels.astype(int))
        self.assertLessEqual(error.max(), 3)

    def test_decodes_the_charls_and_dicom_fixtures(self):
        count = 0
        for case in fixture_cases(JPEGLS_FIXTURES):
            with self.subTest(case=case["name"]):
                self.assertEqual(j2k.decode_jpegls(case["codestream"]), case["reference"])
                count += 1
        self.assertGreater(count, 50)

    def test_accepts_buffers_and_rejects_bad_input(self):
        pixels = samples(16, 8, 1, 8)
        encoded = j2k.encode_jpegls(memoryview(pixels.tobytes()), 16, 8)
        self.assertEqual(j2k.decode_jpegls(bytearray(encoded)), pixels.tobytes())
        with self.assertRaises(ValueError):
            j2k.decode_jpegls(encoded[: len(encoded) // 2])
        with self.assertRaises(ValueError):
            j2k.decode_jpegls(b"\xff\xd8\xff\xc0")
        with self.assertRaises(ValueError):
            j2k.encode_jpegls(pixels.tobytes(), 16, 8, interleave="planar")
        with self.assertRaises(ValueError):
            j2k.encode_jpegls(pixels.tobytes()[:-1], 16, 8)
        with self.assertRaises(ValueError):
            j2k.encode_jpegls(b"\x00" * 31, 4, 4, bit_depth=12)


class DicomRleTests(unittest.TestCase):
    def test_matches_pydicom_on_the_fixtures(self):
        count = 0
        for case in fixture_cases(RLE_FIXTURES):
            with self.subTest(case=case["name"]):
                layout = dict(
                    samples_per_pixel=int(case["samples"]), bits_allocated=int(case["bits"]),
                )
                rows, columns = int(case["rows"]), int(case["columns"])
                decoded = j2k.decode_rle(case["rle"], rows, columns, **layout)
                self.assertEqual(decoded, case["reference"])
                self.assertEqual(j2k.encode_rle(case["reference"], rows, columns, **layout),
                                 case["pydicom_rle"])
                count += 1
        self.assertGreater(count, 20)

    def test_round_trips_both_planar_configurations(self):
        for bits in (8, 16, 32):
            for planar in (0, 1):
                with self.subTest(bits=bits, planar=planar):
                    frame = samples(19, 11, 3, 16, seed=bits).astype(f"<u{bits // 8}")
                    if planar:
                        frame = np.ascontiguousarray(frame.transpose(2, 0, 1))
                    layout = dict(samples_per_pixel=3, bits_allocated=bits,
                                  planar_configuration=planar)
                    encoded = j2k.encode_rle(frame.tobytes(), 11, 19, **layout)
                    self.assertEqual(j2k.decode_rle(encoded, 11, 19, **layout), frame.tobytes())

    def test_lenient_decode_warns_about_what_it_accepted(self):
        frame = samples(8, 4, 1, 8).tobytes()
        encoded = j2k.encode_rle(frame, 4, 8)
        # Another replicate run after the segment's rows are complete.
        padded = encoded + b"\xff\x07"
        with self.assertRaises(ValueError):
            j2k.decode_rle(padded, 4, 8)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(j2k.decode_rle(padded, 4, 8, lenient=True), frame)
        self.assertEqual([warning.category for warning in caught], [j2k.RecoveredImageWarning])
        self.assertIn("segment(s) 0", str(caught[0].message))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertEqual(j2k.decode_rle(encoded, 4, 8, lenient=True), frame)

    def test_rejects_bad_layouts_and_frames(self):
        frame = bytes(32)
        encoded = j2k.encode_rle(frame, 4, 8)
        with self.assertRaises(ValueError):
            j2k.decode_rle(encoded[:40], 4, 8)
        with self.assertRaises(ValueError):
            j2k.decode_rle(encoded, 4, 8, bits_allocated=12)
        with self.assertRaises(ValueError):
            j2k.decode_rle(encoded, 4, 8, planar_configuration=2)
        with self.assertRaises(ValueError):
            j2k.decode_rle(encoded, 4, 8, samples_per_pixel=3)
        with self.assertRaises(ValueError):
            j2k.encode_rle(frame[:-1], 4, 8)
        with self.assertRaises(OverflowError):
            j2k.decode_rle(encoded, 70000, 8)


if __name__ == "__main__":
    unittest.main()
