# SPDX-License-Identifier: MIT OR Apache-2.0

"""Optional direct CUDA tensor checks; required runners set J2K_REQUIRE_CUDA=1."""

import gc
import os
import unittest
from unittest.mock import patch

import numpy as np
import torch
import j2k
from test_adapter import encoded

AVAILABLE = j2k.cuda_enabled and torch.cuda.is_available()
if os.environ.get("J2K_REQUIRE_CUDA") == "1" and not AVAILABLE:
    raise RuntimeError("CUDA adapter build and CUDA hardware are required")


@unittest.skipUnless(AVAILABLE, "requires the cuda extension build and CUDA hardware")
class CudaTests(unittest.TestCase):
    def test_bare_device_binds_current_device_once(self):
        ordinal = torch.cuda.current_device()
        with patch.object(torch.cuda, "current_device", return_value=ordinal) as current:
            decoder = j2k.Decoder(device="cuda")
            current.assert_called_once_with()
        with patch.object(torch.cuda, "current_device", side_effect=AssertionError("decoder rebound")):
            output = decoder.decode(encoded("gray_u8_53"))
            self.assertEqual(output.__dlpack_device__(), (2, ordinal))
        tensor = output.to_torch()
        self.assertEqual(tensor.device.index, ordinal)
        if torch.cuda.device_count() > 1:
            other = (ordinal + 1) % torch.cuda.device_count()
            with torch.cuda.device(other):
                self.assertEqual(decoder.decode(encoded("gray_u8_53")).__dlpack_device__(), (2, ordinal))
                self.assertEqual(j2k.Decoder(device="cuda").decode(encoded("gray_u8_53")).__dlpack_device__(), (2, other))

    def test_single_image_request_preserves_shape_and_pixels(self):
        decoder = j2k.Decoder(device="cuda")
        data = encoded("rgb_u8_53")
        for options in ({}, {"region": (2, 3, 8, 7)}, {"reduction": 1}):
            expected = j2k.Decoder().decode(data, **options).to_torch()
            output = decoder.decode(j2k.ImageRequest(data, **options))
            with self.assertRaises(BufferError):
                np.from_dlpack(output)
            image = output.to_torch()
            self.assertEqual(image.shape, expected.shape)
            self.assertTrue(torch.equal(image.cpu(), expected))

    def test_foreign_import_rejection_preserves_the_tensor(self):
        data = encoded("gray_u8_53")
        output = j2k.Decoder(device="cuda").decode(data)
        for _ in range(3):
            with self.assertRaises(BufferError):
                np.from_dlpack(output)
            gc.collect()
        expected = j2k.Decoder().decode(data).to_torch()
        self.assertTrue(torch.equal(output.to_torch().cpu(), expected))

    def test_mixed_inputs_preserve_indices_and_gpu_export_after_numpy_rejection(self):
        data = encoded("gray_u8_53")
        jpeg = j2k.encode_rgb(bytes(range(192)), 8, 8, codec="jpeg", quality=90)
        decoder = j2k.Decoder(device="cuda")
        result = decoder.decode_batch([jpeg, data, jpeg, b"invalid"])
        self.assertEqual([index for index, _ in result.errors], [0, 2, 3])
        self.assertTrue(all("CUDA" in message for index, message in result.errors if index in (0, 2)))
        self.assertFalse(result.group_errors)
        self.assertEqual(result.groups[0].source_indices, [1])
        output = result.groups[0].tensor
        with self.assertRaises(BufferError):
            output.to_numpy()
        image = output.to_torch()
        self.assertEqual(image.device.type, "cuda")
        expected = j2k.Decoder().decode(data).to_torch()
        self.assertTrue(torch.equal(image[0].cpu(), expected))
        with self.assertRaisesRegex(ValueError, "CUDA"):
            decoder.decode(jpeg)

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
