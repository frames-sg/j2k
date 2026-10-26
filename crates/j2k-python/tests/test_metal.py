# SPDX-License-Identifier: MIT OR Apache-2.0

"""Direct MPS ownership checks; GPU runners set J2K_REQUIRE_METAL=1."""

from concurrent.futures import ThreadPoolExecutor
import gc
import os
from pathlib import Path
import unittest

import numpy as np
import torch
import j2k
from test_adapter import encoded

AVAILABLE = getattr(j2k, "metal_enabled", False) and torch.backends.mps.is_available()
if os.environ.get("J2K_REQUIRE_METAL") == "1" and not AVAILABLE:
    raise RuntimeError("Metal adapter build and MPS hardware are required")


class CompressedDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {"data": encoded("gray_u8_53"), "label": index}


def compressed_records(items):
    return items


@unittest.skipUnless(AVAILABLE, "requires the metal extension build and MPS hardware")
class MetalTests(unittest.TestCase):
    def test_reader_workers_send_records_to_main_process_mps_decoder(self):
        loader = torch.utils.data.DataLoader(CompressedDataset(), batch_size=2,
            num_workers=2, multiprocessing_context="spawn", pin_memory=False,
            collate_fn=compressed_records)
        decoder = j2k.Decoder(device="mps")
        expected = j2k.Decoder().decode(encoded("gray_u8_53")).to_torch()
        labels = []
        for records in loader:
            result = decoder.decode_batch([j2k.ImageRequest(record["data"]) for record in records])
            self.assertFalse(result.errors or result.group_errors)
            for group in result.groups:
                pixels = group.tensor.to_torch().cpu()
                for row, index in enumerate(group.source_indices):
                    labels.append(records[index]["label"])
                    self.assertTrue(torch.equal(pixels[row], expected))
        self.assertEqual(labels, [0, 1, 2, 3])

    def test_explicit_zero_device_alias(self):
        data = encoded("gray_u8_53")
        actual = j2k.Decoder(device="mps:0").decode(j2k.ImageRequest(data)).to_torch()
        expected = j2k.Decoder(device="mps").decode(data).to_torch()
        self.assertTrue(torch.equal(actual.cpu(), expected.cpu()))

    def test_foreign_import_rejection_preserves_the_tensor(self):
        data = encoded("gray_u8_53")
        output = j2k.Decoder(device="mps").decode(data)
        for _ in range(3):
            with self.assertRaises(BufferError):
                np.from_dlpack(output)
            gc.collect()
        expected = j2k.Decoder().decode(data).to_torch()
        self.assertTrue(torch.equal(output.to_torch().cpu(), expected))

    def test_native_types_layouts_and_requests_match_cpu(self):
        fixtures = [encoded(name) for name in (
            "gray_u8_53", "gray_u16_53", "gray_s16_53",
            "rgb_u8_53", "rgb_u12_53", "rgb_s16_53_single",
        )]
        # OpenJPEG via Pillow, lossless 19x13 RGBA, two decomposition levels.
        fixtures.append((Path(__file__).parent / "fixtures/rgba8.jp2").read_bytes())
        for layout in ("nchw", "nhwc"):
            cpu = j2k.Decoder(layout=layout)
            decoder = j2k.Decoder(device="mps", layout=layout)
            for index, data in enumerate(fixtures):
                for options in ({}, {"region": (3, 5, 11, 7)}, {"reduction": 1},
                                {"region": (3, 5, 11, 7), "reduction": 1}):
                    with self.subTest(layout=layout, fixture=index, options=options):
                        expected = cpu.decode(data, **options).to_torch()
                        output = decoder.decode(data, **options)
                        self.assertEqual(output.__dlpack_device__(), (8, 0))
                        image = output.to_torch()
                        self.assertEqual(image.device.type, "mps")
                        self.assertEqual(image.dtype, expected.dtype)
                        self.assertTrue(torch.equal(image.cpu(), expected))
                        self.assertTrue(torch.equal((image.float() * 2 + 1).cpu(), expected.float() * 2 + 1))

    def test_prepared_reuse_mutation_and_decoder_lifetime(self):
        decoder = j2k.Decoder(device="mps")
        data = encoded("rgb_u8_53")
        prepared = decoder.prepare([data, b"invalid", data, encoded("gray_u8_53")])
        expected = j2k.Decoder().decode_prepared(prepared)
        first = decoder.decode_prepared(prepared)
        self.assertEqual([e[0] for e in first.errors], [1])
        self.assertFalse(first.group_errors)
        for actual, reference in zip(first.groups, expected.groups):
            self.assertEqual(actual.source_indices, reference.source_indices)
            self.assertEqual(actual.decoded_rects, reference.decoded_rects)
            self.assertEqual(actual.warnings, reference.warnings)
            self.assertEqual((actual.precision, actual.signed, actual.color, actual.alpha, actual.layout),
                             (reference.precision, reference.signed, reference.color, reference.alpha, reference.layout))
        group = next(g for g in first.groups if g.source_indices == [0, 2])
        tensor = group.tensor.to_torch()
        self.assertTrue(torch.equal(tensor[0].cpu(), tensor[1].cpu()))
        tensor.fill_(17)
        for _ in range(3):
            output = decoder.decode_prepared(prepared)
            self.assertEqual([e[0] for e in output.errors], [1])
            self.assertFalse(output.group_errors)
            fresh = next(g for g in output.groups if g.source_indices == [0, 2]).tensor.to_torch()
            self.assertTrue(torch.equal(fresh[0].cpu(), j2k.Decoder().decode(data).to_torch()))
        del decoder, prepared, first, group, output, fresh
        gc.collect()
        self.assertTrue(torch.all(tensor.cpu() == 17).item())
        tensor.add_(1)
        self.assertTrue(torch.all(tensor.cpu() == 18).item())

    def test_export_once_rejections_and_worker_thread_destruction(self):
        output = j2k.Decoder(device="mps").decode(encoded("gray_u8_53"))
        with self.assertRaises(BufferError):
            output.to_numpy()
        for kwargs in ({"stream": 1}, {"copy": True}, {"dl_device": (1, 0)}):
            with self.assertRaises((ValueError, BufferError)):
                output.__dlpack__(**kwargs)
        tensor = output.to_torch()
        with self.assertRaises(RuntimeError):
            output.to_torch()
        pending = tensor.float().sum()
        with ThreadPoolExecutor(max_workers=1) as pool:
            holder = [tensor]
            del tensor, output
            pool.submit(lambda: holder.clear()).result()

            def decode_and_abandon():
                decoder = j2k.Decoder(device="mps")
                capsule = decoder.decode(encoded("gray_u8_53")).__dlpack__()
                del capsule, decoder
                gc.collect()

            pool.submit(decode_and_abandon).result()
        torch.mps.synchronize()
        expected = j2k.Decoder().decode(encoded("gray_u8_53")).to_torch().float().sum()
        self.assertEqual(pending.item(), expected.item())

    def test_gpu_decode_failure_discards_only_its_group_and_surfaces_single_error(self):
        data = encoded("gray_u8_53")
        # Zero the final HT cleanup segment's Scup trailer. Packet preparation
        # still succeeds, but the Metal decoder must reject the invalid payload.
        corrupt = data[:-4] + b"\x00\x00" + data[-2:]
        decoder = j2k.Decoder(device="mps")
        result = decoder.decode_batch([corrupt, encoded("rgb_u8_53")])
        self.assertFalse(result.errors)
        self.assertEqual([indices for indices, _ in result.group_errors], [[0]])
        self.assertIn("Metal kernel decode failure", result.group_errors[0][1])
        self.assertEqual(result.groups[0].source_indices, [1])
        expected = j2k.Decoder().decode(encoded("rgb_u8_53")).to_torch()
        self.assertTrue(torch.equal(result.groups[0].tensor.to_torch()[0].cpu(), expected))
        with self.assertRaisesRegex(ValueError, "Metal kernel decode failure"):
            decoder.decode(corrupt)

    def test_jpeg_is_an_indexed_error_without_cpu_fallback(self):
        jpeg = j2k.encode_rgb(bytes(range(192)), 8, 8, codec="jpeg", quality=90)
        decoder = j2k.Decoder(device="mps")
        result = decoder.decode_batch([jpeg, encoded("gray_u8_53"), jpeg])
        self.assertEqual([i for i, _ in result.errors], [0, 2])
        self.assertTrue(all("MPS" in message for _, message in result.errors))
        self.assertFalse(result.group_errors)
        self.assertEqual(result.groups[0].source_indices, [1])
        self.assertEqual(result.groups[0].tensor.to_torch().device.type, "mps")
        with self.assertRaisesRegex(ValueError, "MPS"):
            decoder.decode(jpeg)


if __name__ == "__main__":
    unittest.main()
