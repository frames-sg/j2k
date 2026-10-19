# SPDX-License-Identifier: MIT OR Apache-2.0

"""Optional direct CUDA tensor checks; required runners set J2K_REQUIRE_CUDA=1."""

import gc
import os
import unittest

import torch
import j2k
from test_adapter import encoded

AVAILABLE = j2k.cuda_enabled and torch.cuda.is_available()
if os.environ.get("J2K_REQUIRE_CUDA") == "1" and not AVAILABLE:
    raise RuntimeError("CUDA adapter build and CUDA hardware are required")


@unittest.skipUnless(AVAILABLE, "requires the cuda extension build and CUDA hardware")
class CudaTests(unittest.TestCase):
    def test_nondefault_stream_direct_output_matches_cpu_and_outlives_decoder(self):
        cpu = j2k.Decoder()
        decoder = j2k.Decoder(device="cuda:0")
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for name in ["gray_u8_53", "gray_u12_53", "gray_s16_53", "rgb_u8_53", "rgb_u12_53"]:
                with self.subTest(name=name):
                    data = encoded(name)
                    expected = cpu.decode(data).to_torch()
                    image = decoder.decode(data).to_torch()
                    self.assertEqual(image.device.type, "cuda")
                    self.assertEqual(image.dtype, expected.dtype)
                    self.assertTrue(torch.equal(image.cpu(), expected))
                    result = image.float().mean()
            del decoder, image
            gc.collect()
            self.assertTrue(torch.isfinite(result).item())

    def test_default_stream_prepared_reuse_and_partial_failure(self):
        decoder = j2k.Decoder(device="cuda")
        prepared = decoder.prepare([encoded("rgb_u8_53"), b"invalid", encoded("rgb_u8_53")])
        first = decoder.decode_prepared(prepared)
        self.assertEqual([e[0] for e in first.errors], [1])
        self.assertFalse(first.group_errors)
        self.assertEqual(first.groups[0].source_indices, [0, 2])
        tensor = first.groups[0].tensor.to_torch()
        saved = tensor.clone()
        for _ in range(8):
            output = decoder.decode_prepared(prepared)
            self.assertFalse(output.group_errors)
            output.groups[0].tensor.to_torch().float().sum().item()
        del decoder, prepared, first
        gc.collect()
        self.assertTrue(torch.equal(tensor, saved))
