# SPDX-License-Identifier: MIT OR Apache-2.0

"""Image-server facts: inspect metadata for IIIF info.json."""

import array
import mmap
import os
import struct
import tempfile
import unittest

import numpy as np

import j2k

from test_adapter import ROOT

ORIGIN = ROOT / "crates/j2k-test-support/fixtures/openjpeg_origin"


def gray(width, height):
    return (np.arange(width * height) * 7 % 251).astype(np.uint8).tobytes()


def tiled(levels=4, **options):
    return j2k.encode(gray(150, 90), 150, 90, components=1, codec="j2k", levels=levels,
                      tile_size=(64, 48), **options)


def with_component_override(codestream, levels):
    """Insert a main-header COC giving component 0 fewer decomposition levels."""
    offset = 2
    while True:
        marker, length = struct.unpack_from(">HH", codestream, offset)
        if marker == 0xFF52:
            break
        offset += 2 + length
    cod = codestream[offset + 4:offset + 2 + length]
    # Scoc keeps only the precinct flag; SPcoc repeats SPcod with new levels.
    body = bytes([0, cod[0] & 1, levels]) + cod[6:]
    coc = struct.pack(">HH", 0xFF53, len(body) + 2) + body
    end = offset + 2 + length
    return codestream[:end] + coc + codestream[end:]


def info_json(data, identifier):
    """The README's info.json construction, executed so it stays correct."""
    info = j2k.inspect(data)
    decoder = j2k.Decoder()
    reductions = range(info["max_reduction"] + 1)
    sizes = []
    for reduction in reversed(reductions):
        plan = decoder.plan_region(data, reduction=reduction)
        sizes.append({"width": plan["width"], "height": plan["height"]})
    return {
        "@context": "http://iiif.io/api/image/3/context.json",
        "id": identifier,
        "type": "ImageService3",
        "protocol": "http://iiif.io/api/image",
        "profile": "level1",
        "width": info["width"],
        "height": info["height"],
        "tiles": [{"width": info["tile_width"], "height": info["tile_height"],
                   "scaleFactors": [2**reduction for reduction in reductions]}],
        "sizes": sizes,
    }


class InspectTests(unittest.TestCase):
    def test_tile_grid_levels_and_coding_style(self):
        data = tiled(levels=4, progression="rpcl", quality_layers=3)
        info = j2k.inspect(data)
        self.assertEqual((info["width"], info["height"]), (150, 90))
        self.assertEqual(info["resolution_levels"], 5)
        self.assertEqual(info["decomposition_levels"], 4)
        self.assertEqual(info["max_reduction"], 4)
        self.assertEqual((info["tile_width"], info["tile_height"]), (64, 48))
        self.assertEqual((info["tiles_x"], info["tiles_y"]), (3, 2))
        self.assertEqual((info["image_x0"], info["image_y0"]), (0, 0))
        self.assertEqual((info["tile_x0"], info["tile_y0"]), (0, 0))
        self.assertEqual(info["quality_layers"], 3)
        self.assertEqual(info["progression_order"], "rpcl")
        self.assertEqual((info["code_block_width"], info["code_block_height"]), (64, 64))
        # A file reports the same codestream facts.
        self.assertEqual(j2k.inspect(j2k.encode(gray(150, 90), 150, 90, components=1, codec="j2k",
                                                levels=4, tile_size=(64, 48), file=True))["max_reduction"], 4)

    def test_max_reduction_follows_component_overrides(self):
        data = with_component_override(tiled(levels=4), levels=2)
        info = j2k.inspect(data)
        # The main COD still declares four levels; every component supports two.
        self.assertEqual(info["decomposition_levels"], 4)
        self.assertEqual(info["max_reduction"], 2)
        decoder = j2k.Decoder()
        self.assertEqual(decoder.plan_region(data)["max_reduction"], info["max_reduction"])
        decoder.decode(data, reduction=2)
        with self.assertRaises(ValueError):
            decoder.decode(data, reduction=3)

    def test_image_and_tile_origins(self):
        info = j2k.inspect((ORIGIN / "rgb_23x17_origin_3_5.j2k").read_bytes())
        self.assertEqual((info["width"], info["height"]), (23, 17))
        self.assertEqual((info["image_x0"], info["image_y0"]), (3, 5))
        self.assertLessEqual(info["tile_x0"], 3)
        self.assertLessEqual(info["tile_y0"], 5)

    def test_info_json_tiles_and_sizes(self):
        data = tiled(levels=3)
        document = info_json(data, "https://example.org/iiif/image")
        self.assertEqual(document["tiles"],
                         [{"width": 64, "height": 48, "scaleFactors": [1, 2, 4, 8]}])
        self.assertEqual(document["sizes"], [
            {"width": 19, "height": 12}, {"width": 38, "height": 23},
            {"width": 75, "height": 45}, {"width": 150, "height": 90}])
        # Each listed size is what a full decode at that reduction returns.
        decoder = j2k.Decoder(layout="nhwc")
        for reduction, size in enumerate(reversed(document["sizes"])):
            shape = decoder.decode(data, reduction=reduction).to_numpy().shape
            self.assertEqual((shape[1], shape[0]), (size["width"], size["height"]))
        # A nonzero image origin moves the reduced grid: sizes come from the plan.
        origin = info_json((ORIGIN / "rgb_23x17_origin_3_5.j2k").read_bytes(), "origin")
        self.assertEqual(origin["sizes"][-1], {"width": 23, "height": 17})


class BufferInputTests(unittest.TestCase):
    """Every decode, inspect and plan entry point takes any bytes-like input."""

    @classmethod
    def setUpClass(cls):
        cls.data = tiled(levels=3)
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.directory.name, "image.j2c")
        with open(cls.path, "wb") as file:
            file.write(cls.data)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def mapped(self, access):
        with open(self.path, "r+b") as file:
            return mmap.mmap(file.fileno(), 0, access=access)

    def buffers(self):
        data = self.data
        padded = b"head" + data + b"tail"
        return {
            "bytes": data,
            "bytearray": bytearray(data),
            "memoryview": memoryview(data),
            "memoryview slice": memoryview(padded)[4:-4],
            "read-only memoryview of bytearray": memoryview(bytearray(data)).toreadonly(),
            "array": array.array("B", data),
            "numpy": np.frombuffer(data, dtype=np.uint8),
            "read-only mmap": self.mapped(mmap.ACCESS_READ),
            "writable mmap": self.mapped(mmap.ACCESS_WRITE),
            "copy-on-write mmap": self.mapped(mmap.ACCESS_COPY),
        }

    def test_every_entry_point_accepts_every_buffer_type(self):
        decoder = j2k.Decoder(layout="nhwc")
        expected = decoder.decode(self.data, region=(5, 7, 70, 50), reduction=1).to_numpy()
        expected_srgb = decoder.decode_srgb8(self.data, reduction=1).to_numpy()
        expected_plan = decoder.plan_region(self.data, region=(5, 7, 70, 50), reduction=1)
        expected_info = j2k.inspect(self.data)
        expected_bands = [band.tensor.to_numpy() for component in
                          j2k.decode_wavelets(self.data, resolutions=[0]).components
                          for band in component.bands]
        for name, buffer in self.buffers().items():
            with self.subTest(buffer=name):
                self.assertEqual(j2k.inspect(buffer), expected_info)
                self.assertEqual(
                    decoder.plan_region(buffer, region=(5, 7, 70, 50), reduction=1), expected_plan)
                np.testing.assert_array_equal(
                    decoder.decode(buffer, region=(5, 7, 70, 50), reduction=1).to_numpy(), expected)
                np.testing.assert_array_equal(
                    decoder.decode_srgb8(buffer, reduction=1).to_numpy(), expected_srgb)
                request = j2k.ImageRequest(buffer, region=(5, 7, 70, 50), reduction=1)
                np.testing.assert_array_equal(decoder.decode(request).to_numpy(), expected)
                batch = decoder.decode_batch([buffer, request])
                self.assertFalse(batch.errors or batch.group_errors)
                prepared = decoder.prepare([buffer], region=(5, 7, 70, 50), reduction=1)
                np.testing.assert_array_equal(
                    decoder.decode_prepared(prepared).groups[0].tensor.to_numpy()[0], expected)
                bands = [band.tensor.to_numpy() for component in
                         j2k.decode_wavelets(buffer, resolutions=[0]).components
                         for band in component.bands]
                self.assertEqual(len(bands), len(expected_bands))
                for band, expected_band in zip(bands, expected_bands):
                    np.testing.assert_array_equal(band, expected_band)
                if isinstance(buffer, mmap.mmap):
                    # No buffer export outlives the calls: the map can close.
                    buffer.close()

    def test_rgb_helper_accepts_buffers(self):
        pixels = (np.arange(24 * 16 * 3) % 251).astype(np.uint8).tobytes()
        data = j2k.encode_rgb(pixels, 24, 16, codec="j2k")
        expected = j2k.decode_rgb(data)
        for buffer in (bytearray(data), memoryview(data), np.frombuffer(data, np.uint8)):
            self.assertEqual(j2k.decode_rgb(buffer), expected)

    def test_requests_own_their_bytes(self):
        source = bytearray(self.data)
        request = j2k.ImageRequest(source)
        source[:] = bytes(len(source))
        decoder = j2k.Decoder(layout="nhwc")
        np.testing.assert_array_equal(
            decoder.decode(request).to_numpy(), decoder.decode(self.data).to_numpy())

    def test_other_layouts_and_types_are_rejected_clearly(self):
        decoder = j2k.Decoder()
        strided = memoryview(self.data)[::2]
        two_d = memoryview(self.data[:150 * 2]).cast("B", (2, 150))
        for name, call in [
            ("inspect", j2k.inspect),
            ("decode", decoder.decode),
            ("decode_srgb8", decoder.decode_srgb8),
            ("plan_region", decoder.plan_region),
            ("ImageRequest", j2k.ImageRequest),
            ("decode_wavelets", j2k.decode_wavelets),
        ]:
            with self.subTest(entry=name):
                with self.assertRaisesRegex(BufferError, "one-dimensional C-contiguous"):
                    call(strided)
                with self.assertRaisesRegex(BufferError, "one-dimensional C-contiguous"):
                    call(two_d)
                with self.assertRaisesRegex(TypeError, "unsigned bytes"):
                    call(array.array("i", [1, 2, 3]))
                with self.assertRaisesRegex(TypeError, "bytes-like"):
                    call("image.jp2")
        with self.assertRaisesRegex(TypeError, "input 1: expected bytes, a bytes-like object"):
            decoder.decode_batch([self.data, "image.jp2"])
        with self.assertRaisesRegex(BufferError, "input 0: .*C-contiguous"):
            decoder.prepare([strided])


if __name__ == "__main__":
    unittest.main()
