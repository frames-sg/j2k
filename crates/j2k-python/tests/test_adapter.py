# SPDX-License-Identifier: MIT OR Apache-2.0

"""Public Python/PyTorch contract checks against existing independent fixtures.

Run after installing the wheel: python -m unittest discover -s crates/j2k-python/tests
"""

import gc
from pathlib import Path
import unittest

import numpy as np
import torch

import j2k


ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "crates/j2k-test-support/fixtures/htj2k/openjph_batch"


def encoded(name):
    return (FIXTURES / f"{name}.j2c").read_bytes()


class CodecDataset(torch.utils.data.Dataset):
    """Each spawned worker creates its own persistent CPU session."""

    def __init__(self):
        self.decoder = None

    def __len__(self):
        return 4

    def __getitem__(self, index):
        if self.decoder is None:
            self.decoder = j2k.Decoder(workers=1)
        return self.decoder.decode(encoded("gray_u8_53")).to_torch(), index


class AdapterTests(unittest.TestCase):
    def test_native_dtypes_and_values_match_independent_oracles(self):
        decoder = j2k.Decoder(layout="nhwc")
        for name, dtype in [("gray_u8_53", np.uint8), ("gray_u12_53", np.dtype("<u2")), ("gray_s16_53", np.dtype("<i2")), ("rgb_u8_53", np.uint8)]:
            with self.subTest(name=name):
                info = j2k.inspect(encoded(name))
                output = decoder.decode(encoded(name))
                array = output.to_numpy()
                expected = np.frombuffer((FIXTURES / f"{name}.oracle.raw").read_bytes(), dtype=dtype)
                np.testing.assert_array_equal(array.reshape(-1), expected)
                self.assertEqual(array.shape, (info["height"], info["width"], info["components"]))

    def test_prepared_mixed_batches_preserve_indices_and_errors(self):
        decoder = j2k.Decoder()
        prepared = decoder.prepare([encoded("gray_u8_53"), encoded("rgb_u8_53"), encoded("gray_u8_53"), b"invalid"])
        first = decoder.decode_prepared(prepared)
        self.assertEqual(sorted(i for group in first.groups for i in group.source_indices), [0, 1, 2])
        self.assertEqual([error[0] for error in first.errors], [3])
        gray = next(group for group in first.groups if group.source_indices == [0, 2])
        tensor = gray.tensor.to_torch()
        self.assertTrue(torch.equal(tensor[0], tensor[1]))
        saved = tensor.clone()
        # A consumer owns its output even after decoder/result destruction.
        del first, gray
        second = decoder.decode_prepared(prepared)
        del decoder, prepared, second
        gc.collect()
        self.assertTrue(torch.equal(tensor, saved))
        tensor.add_(1)

    def test_region_reduction_and_layout(self):
        data = encoded("rgb_u8_53")
        decoder = j2k.Decoder(layout="nhwc")
        full = decoder.decode(data).to_numpy()
        region = decoder.decode(data, region=(3, 5, 11, 7)).to_numpy()
        np.testing.assert_array_equal(region, full[5:12, 3:14])
        reduced = decoder.decode(data, reduction=1).to_torch()
        self.assertEqual(tuple(reduced.shape), ((full.shape[0] + 1) // 2, (full.shape[1] + 1) // 2, 3))
        mixed = decoder.decode_batch([data, j2k.ImageRequest(data, region=(3, 5, 11, 7)), j2k.ImageRequest(data, reduction=1)])
        self.assertFalse(mixed.errors or mixed.group_errors)
        self.assertEqual(sorted(i for group in mixed.groups for i in group.source_indices), [0, 1, 2])
        for group in mixed.groups:
            array = group.tensor.to_numpy()
            expected = [full, region, reduced.numpy()][group.source_indices[0]]
            np.testing.assert_array_equal(array[0], expected)
        channels_first = j2k.Decoder().decode(data).to_numpy()
        np.testing.assert_array_equal(channels_first, full.transpose(2, 0, 1))
        with self.assertRaises(ValueError):
            decoder.decode(data, reduction=99)
        with self.assertRaises(ValueError):
            decoder.decode(data, region=(3, 5, 11, 9))

    def test_dlpack_transfers_ownership_once_and_handles_unconsumed_capsule(self):
        output = j2k.Decoder().decode(encoded("gray_u8_53"))
        self.assertEqual(output.__dlpack_device__(), (1, 0))
        tensor = torch.from_dlpack(output)
        with self.assertRaises(RuntimeError):
            torch.from_dlpack(output)
        del output
        gc.collect()
        self.assertEqual(tensor.dtype, torch.uint8)
        # The capsule destructor must retire an export that no framework consumed.
        abandoned = j2k.Decoder().decode(encoded("gray_u8_53")).__dlpack__()
        del abandoned
        gc.collect()

    def test_wavelet_tensors_and_selection(self):
        data = encoded("gray_u12_53")
        full = j2k.decode_wavelets(data)
        self.assertEqual(full.components[0].transform, "5/3")
        selected = j2k.decode_wavelets(data, resolutions=[0], bands=["LL"])
        self.assertEqual(len(selected.components), len(full.components))
        for component in selected.components:
            self.assertEqual(len(component.bands), 1)
            band = component.bands[0]
            expected = next(c for c in full.components if (c.tile_index, c.index) == (component.tile_index, component.index)).bands[0]
            actual = band.tensor.to_torch()
            self.assertEqual(actual.dtype, torch.int64)
            self.assertTrue(torch.equal(actual, expected.tensor.to_torch()))

    def test_spawned_dataloader_workers(self):
        loader = torch.utils.data.DataLoader(CodecDataset(), batch_size=2, num_workers=2, multiprocessing_context="spawn")
        batches = list(loader)
        self.assertEqual(torch.cat([labels for _, labels in batches]).tolist(), [0, 1, 2, 3])
        self.assertEqual(sum(images.shape[0] for images, _ in batches), 4)


if __name__ == "__main__":
    unittest.main()
