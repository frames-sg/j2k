# SPDX-License-Identifier: MIT OR Apache-2.0

"""Image-server facts: inspect metadata for IIIF info.json."""

import struct
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

if __name__ == "__main__":
    unittest.main()
