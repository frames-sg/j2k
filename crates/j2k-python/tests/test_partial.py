# SPDX-License-Identifier: MIT OR Apache-2.0

"""Decoding the start of a codestream, as a progressive viewer does."""

import unittest

import numpy as np
import j2k


def ct_like_codestream():
    y, x = np.indices((72, 96))
    pixels = (((x * 37 + y * 11) ^ (x * y)) % 4096).astype("<u2")
    return j2k.encode(
        pixels.tobytes(),
        96,
        72,
        components=1,
        bit_depth=12,
        codec="htj2k",
        levels=4,
        progression="rpcl",
    )


class PartialCodestreamTest(unittest.TestCase):
    def test_prefixes_decode_at_the_reported_reduction(self):
        full = ct_like_codestream()
        complete = j2k.Decoder(layout="nhwc")
        references = {
            reduction: complete.decode(full, reduction=reduction).to_numpy()
            for reduction in range(5)
        }
        decoder = j2k.Decoder(layout="nhwc", partial=True)
        seen = set()
        previous = None
        for length in range(16, len(full) + 1, max(1, len(full) // 120)):
            prefix = full[:length]
            try:
                report = j2k.completeness(prefix)
            except ValueError:
                self.assertIsNone(previous, f"prefix {length} failed after a shorter one opened")
                continue
            reduction = report["min_exact_reduction"]
            if previous is not None:
                self.assertIsNotNone(reduction)
                self.assertLessEqual(reduction, previous)
            if reduction is None:
                with self.assertRaisesRegex(ValueError, "partial codestream"):
                    decoder.decode(prefix, reduction=4)
                continue
            previous = reduction
            seen.add(reduction)
            np.testing.assert_array_equal(
                decoder.decode(prefix, reduction=reduction).to_numpy(),
                references[reduction],
            )
            if reduction > 0:
                with self.assertRaisesRegex(ValueError, "partial codestream"):
                    decoder.decode(prefix, reduction=reduction - 1)
        self.assertTrue({1, 2, 3, 4} <= seen, seen)

        report = j2k.completeness(full)
        self.assertTrue(report["complete"])
        self.assertEqual(report["min_exact_reduction"], 0)
        self.assertEqual(report["tile_min_exact_reduction"], [0])

    def test_partial_decoding_is_cpu_only(self):
        with self.assertRaisesRegex(ValueError, "CPU only"):
            j2k.Decoder(device="mps", partial=True)


if __name__ == "__main__":
    unittest.main()
