# SPDX-License-Identifier: MIT OR Apache-2.0

"""General encoding and colour inspection through the Python binding.

Run after installing the wheel: python -m unittest discover -s crates/j2k-python/tests
"""

import unittest

import numpy as np

import j2k


WIDTH, HEIGHT = 40, 28
PROFILE = b"test-icc-profile"


def rgb():
    index = np.arange(WIDTH * HEIGHT * 3)
    return ((index * 37 + index // 97 * 11) % 251).astype(np.uint8)


def decode_native(data):
    return j2k.Decoder(layout="nhwc").decode(data).to_numpy()


def has_marker(data, code):
    return code.to_bytes(2, "big") in data


class EncodeTests(unittest.TestCase):
    def test_lossless_parameters_round_trip_native_samples(self):
        pixels = rgb()
        deep = (np.arange(WIDTH * HEIGHT) * 91 % 4096 - 2048).astype("<i2")
        for samples, components, bit_depth, signed, codec in [
            (pixels, 3, 8, False, "j2k"),
            (pixels, 3, 8, False, "htj2k"),
            (deep, 1, 12, True, "htj2k"),
        ]:
            with self.subTest(components=components, codec=codec):
                data = j2k.encode(
                    samples.tobytes(), WIDTH, HEIGHT, components=components, bit_depth=bit_depth,
                    signed=signed, codec=codec, levels=2, progression="rpcl", tile_size=(24, 16),
                    tile_part_packets=3, markers=["sop", "eph", "tlm", "plt"], quality_layers=2,
                    color_transform="auto" if components != 3 else "none",
                )
                self.assertTrue(has_marker(data, 0xFF91) and has_marker(data, 0xFF55))
                decoded = decode_native(data)
                self.assertEqual(decoded.shape, (HEIGHT, WIDTH, components))
                np.testing.assert_array_equal(decoded.reshape(-1), samples)

    def test_lossy_encode_matches_encode_rgb_for_the_same_options(self):
        pixels = rgb().tobytes()
        for codec in ("j2k", "htj2k"):
            with self.subTest(codec=codec):
                general = j2k.encode(pixels, WIDTH, HEIGHT, codec=codec, mode="lossy",
                                     levels=3, target_bpp=2.0)
                research = j2k.encode_rgb(pixels, WIDTH, HEIGHT, codec=codec, levels=3, target_bpp=2.0)
                self.assertEqual(general, research)

    def test_lossy_targets_report_what_the_codec_measured(self):
        pixels = rgb().tobytes()
        data, report = j2k.encode(pixels, WIDTH, HEIGHT, mode="lossy", target_bytes=600,
                                  return_report=True)
        self.assertEqual(report["codestream_bytes"], len(data))
        self.assertLessEqual(len(data), 600)
        self.assertAlmostEqual(report["bits_per_pixel"], len(data) * 8 / (WIDTH * HEIGHT))
        layered, report = j2k.encode(pixels, WIDTH, HEIGHT, codec="j2k", mode="lossy",
                                     target_bpp=[0.75, 1.5, 3.0], return_report=True)
        self.assertEqual(report["quality_layers"], 3)
        self.assertEqual(report["codestream_bytes"], len(layered))
        _, report = j2k.encode(pixels, WIDTH, HEIGHT, codec="j2k", mode="lossy",
                               target_psnr=32.0, return_report=True)
        self.assertGreaterEqual(report["psnr_db"], 32.0 - 0.25)
        tuned = j2k.encode(pixels, WIDTH, HEIGHT, mode="lossy", qfactor=80)
        self.assertEqual(decode_native(tuned).shape, (HEIGHT, WIDTH, 3))
        _, report = j2k.encode(pixels, WIDTH, HEIGHT, return_report=True)
        self.assertEqual((report["quality_layers"], report["psnr_db"]), (1, None))

    def test_files_carry_the_icc_profile_inspect_reports(self):
        pixels = rgb().tobytes()
        for codec, brand in [("j2k", b"jp2 "), ("htj2k", b"jph ")]:
            with self.subTest(codec=codec):
                data = j2k.encode(pixels, WIDTH, HEIGHT, codec=codec, file=True, icc_profile=PROFILE)
                self.assertEqual(data[20:24], brand)
                info = j2k.inspect(data)
                self.assertEqual(info["color_specification"], "icc")
                self.assertEqual(info["icc_profile"], PROFILE)
                self.assertIsNone(info["enumerated_colorspace"])
        info = j2k.inspect(j2k.encode(pixels, WIDTH, HEIGHT, file=True))
        self.assertEqual((info["color_specification"], info["enumerated_colorspace"]), ("enumerated", 16))
        self.assertIsNone(info["icc_profile"])
        info = j2k.inspect(j2k.encode(pixels, WIDTH, HEIGHT))
        self.assertEqual((info["color_specification"], info["icc_profile"]), ("none", None))

    def test_invalid_arguments_raise_value_error(self):
        pixels = rgb().tobytes()
        for case, options in [
            ("codec", {"codec": "jpeg"}),
            ("progression", {"progression": "lrpc"}),
            ("marker", {"markers": ["soc"]}),
            ("two target units", {"mode": "lossy", "target_bpp": 1.0, "target_psnr": 30.0}),
            ("lossless target", {"target_bpp": 1.0}),
            ("lossless qfactor", {"qfactor": 50}),
            ("lossy layer count", {"mode": "lossy", "quality_layers": 2}),
            ("lossy without colour transform", {"mode": "lossy", "color_transform": "none"}),
            ("mode", {"mode": "near-lossless"}),
            ("colour transform", {"color_transform": "rct"}),
            ("ICC without a file", {"icc_profile": PROFILE}),
            ("classic qfactor", {"codec": "j2k", "mode": "lossy", "qfactor": 50}),
            ("qfactor with a rate", {"mode": "lossy", "qfactor": 50, "target_bpp": 1.0}),
            ("too few samples", {"components": 4}),
        ]:
            with self.subTest(case=case), self.assertRaises(ValueError):
                j2k.encode(pixels, WIDTH, HEIGHT, **options)


if __name__ == "__main__":
    unittest.main()
