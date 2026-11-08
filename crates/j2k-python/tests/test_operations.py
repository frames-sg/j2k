# SPDX-License-Identifier: MIT OR Apache-2.0

"""Exact region, preparation, and operation-control binding contracts."""

import gc
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import j2k

from test_adapter import ROOT, encoded


def deep_image(offset=0):
    y, x, channel = np.indices((67, 65, 3))
    pixels = ((x * 5 + y * 7 + channel * 43 + offset) % 256).astype(np.uint8)
    return j2k.encode_rgb(pixels.tobytes(), 65, 67, codec="htj2k", levels=4)


def rows_by_index(batch):
    result = {}
    for group in batch.groups:
        pixels = group.tensor.to_numpy()
        for row, index in enumerate(group.source_indices):
            result[index] = (pixels[row], group.decoded_rects[row])
    return result


def device_rows_by_index(batch):
    """Per-source samples and rectangles of a batch whose tensors may be on a GPU."""
    result = {}
    for group in batch.groups:
        pixels = group.tensor.to_torch().cpu().numpy()
        for row, index in enumerate(group.source_indices):
            result[index] = (pixels[row], group.decoded_rects[row])
    return result


def assert_device_batch_matches_cpu(test, actual, expected):
    test.assertEqual(actual.errors, expected.errors)
    test.assertEqual(actual.group_errors, expected.group_errors)
    actual_rows, expected_rows = device_rows_by_index(actual), rows_by_index(expected)
    test.assertEqual(sorted(actual_rows), sorted(expected_rows))
    for index, (pixels, rect) in expected_rows.items():
        test.assertEqual(actual_rows[index][1], rect, index)
        test.assertTrue(np.array_equal(actual_rows[index][0], pixels), index)


def assert_exact_device_capabilities(test, device):
    """Shared device boundary check; the GPU suites supply real hardware.

    A device decoder returns the CPU decoder's samples and rectangles for
    exact operations and applies their output limits and cancellation; CPU-only
    features fail explicitly.
    """
    data = deep_image()
    decoder = j2k.Decoder(device=device)
    cpu = j2k.Decoder()
    prepared = cpu.prepare([j2k.ImageRequest(data, reduction=4)])
    mixed = [j2k.ImageRequest(data, reduction=4), encoded("gray_u8_53")]
    test.assertTrue(np.array_equal(
        decoder.decode(data, reduction=4).to_torch().cpu().numpy(),
        cpu.decode(data, reduction=4).to_numpy()))
    assert_device_batch_matches_cpu(
        test, decoder.decode_prepared(prepared), cpu.decode_prepared(prepared))
    roi = (17, 9, 19, 25)
    assert_device_batch_matches_cpu(
        test,
        decoder.decode_prepared(prepared, region=roi, reduction=2),
        cpu.decode_prepared(prepared, region=roi, reduction=2))
    assert_device_batch_matches_cpu(
        test, decoder.decode_batch(mixed), cpu.decode_batch(mixed))
    device_prepared = decoder.prepare([j2k.ImageRequest(data, region=roi, reduction=4)])
    assert_device_batch_matches_cpu(
        test,
        decoder.decode_prepared(device_prepared),
        cpu.decode_batch([j2k.ImageRequest(data, region=roi, reduction=4)]))
    with test.assertRaisesRegex(ValueError, "CPU"):
        decoder.decode_srgb8(data)
    with test.assertRaisesRegex(ValueError, "CPU"):
        decoder.plan_region(data, reduction=4)
    with test.assertRaisesRegex(ValueError, "CPU"):
        decoder.prepare([data], control=j2k.DecodeControl(max_output_bytes=1 << 20))
    # Devices apply output limits and cancellation; working limits are a
    # per-operation capability error, never silently ignored.
    bounded = j2k.DecodeControl(max_output_bytes=1 << 20)
    assert_device_batch_matches_cpu(
        test,
        decoder.decode_prepared(prepared, control=bounded),
        cpu.decode_prepared(prepared, control=bounded))
    # A device's own legacy preparation decodes under a control as operations.
    gray = encoded("gray_u8_53")
    assert_device_batch_matches_cpu(
        test,
        decoder.decode_prepared(decoder.prepare([gray]), control=bounded),
        cpu.decode_prepared(cpu.prepare([gray]), control=bounded))
    tiny = decoder.decode_prepared(prepared, control=j2k.DecodeControl(max_output_bytes=1))
    test.assertEqual([index for index, _ in tiny.errors], [0])
    test.assertEqual(tiny.groups, [])
    working = decoder.decode_prepared(prepared, control=j2k.DecodeControl(max_working_memory_bytes=8 << 20))
    test.assertEqual([index for index, _ in working.errors], [0])
    test.assertIn("working-memory", working.errors[0][1])
    token = j2k.CancellationToken()
    token.cancel()
    stopped = decoder.decode_prepared(prepared, control=j2k.DecodeControl(cancellation=token))
    test.assertEqual([index for index, _ in stopped.errors], [0])
    test.assertIn("cancel", stopped.errors[0][1].lower())


class OperationTests(unittest.TestCase):
    def test_exact_requests_plan_single_batch_and_prepared_agree(self):
        data, other = deep_image(), deep_image(71)
        decoder = j2k.Decoder(layout="nhwc", workers=1)
        roi = (17, 9, 19, 25)
        plan = decoder.plan_region(data, region=roi, reduction=4)
        self.assertEqual(plan["source_dimensions"], (65, 67))
        self.assertEqual(plan["output_region"], (1, 0, 2, 3))
        self.assertEqual((plan["reduction"], plan["max_reduction"]), (4, 4))
        self.assertEqual((plan["sample_type"], plan["interpretation"], plan["route"]), ("uint8", "native", "native_cpu"))
        full = decoder.decode(data, reduction=4).to_numpy()
        self.assertEqual(full.shape, (5, 5, 3))
        expected = full[:3, 1:3].copy()
        actual_region = decoder.decode(data, region=roi, reduction=4).to_numpy()
        np.testing.assert_array_equal(actual_region, expected)
        self.assertEqual(tuple(plan["shape"]), actual_region.shape)
        self.assertEqual(tuple(plan["strides"]), actual_region.strides)

        inputs = [j2k.ImageRequest(data, region=roi, reduction=4), b"invalid", j2k.ImageRequest(other, region=roi, reduction=4), j2k.ImageRequest(data, reduction=5)]
        prepared = decoder.prepare(inputs)
        self.assertEqual([index for index, _ in prepared.errors], [1, 3])
        self.assertEqual(prepared.group_count, 1)
        one_shot = decoder.decode_batch(inputs)
        reused = decoder.decode_prepared(prepared)
        self.assertEqual([index for index, _ in reused.errors], [1, 3])
        actual, reference = rows_by_index(reused), rows_by_index(one_shot)
        self.assertEqual(set(actual), {0, 2})
        for index in actual:
            np.testing.assert_array_equal(actual[index][0], reference[index][0])
            self.assertEqual(actual[index][1], (1, 0, 2, 3))
        np.testing.assert_array_equal(actual[0][0], expected)
        restored = rows_by_index(decoder.decode_prepared(prepared, region=(0, 0, 65, 67), reduction=0))
        np.testing.assert_array_equal(restored[0][0], decoder.decode(data).to_numpy())
        decoder.set_operation_cache_limit(0)
        again = rows_by_index(decoder.decode_prepared(prepared))
        del decoder, prepared, reused
        gc.collect()
        np.testing.assert_array_equal(again[0][0], expected)
        with self.assertRaises(ValueError):
            j2k.ImageRequest(data, reduction=32)

    def test_prepared_native_widths_controls_and_independent_consumers(self):
        for name, dtype in [("gray_u8_53", np.uint8), ("gray_u12_53", np.uint16), ("gray_s16_53", np.int16)]:
            data = encoded(name)
            maker = j2k.Decoder(layout="nhwc", workers=1)
            prepared = maker.prepare([data])
            expected = maker.decode(data, reduction=1).to_numpy().copy()
            plan = maker.plan_region(data, reduction=1)
            self.assertEqual(plan["sample_type"], np.dtype(dtype).name)
            self.assertEqual(tuple(plan["shape"]), expected.shape)
            control = j2k.DecodeControl(max_working_memory_bytes=8 << 20, max_output_bytes=expected.nbytes)

            def consume():
                receiver = j2k.Decoder(layout="nhwc", workers=1)
                batch = receiver.decode_prepared(prepared, reduction=1, control=control)
                self.assertEqual(batch.errors, [])
                return batch.groups[0].tensor.to_numpy()

            with ThreadPoolExecutor(max_workers=2) as pool:
                arrays = list(pool.map(lambda _: consume(), range(2)))
            for array in arrays:
                self.assertEqual(array.dtype, np.dtype(dtype))
                np.testing.assert_array_equal(array[0], expected)
            tiny = maker.decode_prepared(prepared, reduction=1, control=j2k.DecodeControl(max_output_bytes=expected.nbytes - 1))
            self.assertEqual([index for index, _ in tiny.errors], [0])
            self.assertEqual(tiny.groups, [])
            working = maker.decode_prepared(prepared, reduction=1, control=j2k.DecodeControl(max_working_memory_bytes=1))
            self.assertEqual([index for index, _ in working.errors], [0])
            self.assertIn("resource", working.errors[0][1].lower())
            token = j2k.CancellationToken()
            cancelled = j2k.DecodeControl(cancellation=token)
            token.cancel()
            self.assertTrue(token.cancelled)
            stopped = maker.decode_prepared(prepared, control=cancelled)
            self.assertIn("cancel", stopped.errors[0][1].lower())
            self.assertEqual(stopped.groups, [])
            np.testing.assert_array_equal(maker.decode_prepared(prepared, reduction=1).groups[0].tensor.to_numpy()[0], expected)
            with self.assertRaisesRegex(ValueError, "layout"):
                j2k.Decoder(layout="nchw").decode_prepared(prepared, reduction=1, control=control)

    def test_explicit_srgb_and_native_plans_have_distinct_interpretation(self):
        data = encoded("gray_u12_53")
        decoder = j2k.Decoder(layout="nhwc", workers=1)
        native = decoder.plan_region(data)
        normalized = decoder.plan_region(data, interpretation="srgb8")
        self.assertEqual((native["sample_type"], native["precision"]), ("uint16", 12))
        self.assertEqual((normalized["sample_type"], normalized["precision"]), ("uint8", 8))
        self.assertEqual(normalized["route"], "region_planes")
        full = decoder.decode_srgb8(data).to_numpy()
        region = decoder.decode_srgb8(data, region=(3, 5, 11, 7)).to_numpy()
        np.testing.assert_array_equal(region, full[5:12, 3:14])
        chw = j2k.Decoder(workers=1).decode_srgb8(data).to_numpy()
        np.testing.assert_array_equal(chw, full.transpose(2, 0, 1))
        rgb = deep_image()
        chw_decoder = j2k.Decoder(workers=1)
        chw_plan = chw_decoder.plan_region(rgb, reduction=4, interpretation="srgb8")
        chw_rgb = chw_decoder.decode_srgb8(rgb, reduction=4).to_numpy()
        self.assertEqual(tuple(chw_plan["shape"]), chw_rgb.shape)
        self.assertEqual(tuple(chw_plan["strides"]), chw_rgb.strides)
        with self.assertRaises(ValueError):
            decoder.decode_srgb8(data, control=j2k.DecodeControl(max_output_bytes=1))
        # Raw sRGB calls parse and decode their input under the control.
        bounded = j2k.DecodeControl(max_working_memory_bytes=8 << 20)
        np.testing.assert_array_equal(decoder.decode_srgb8(data, control=bounded).to_numpy(), full)
        np.testing.assert_array_equal(
            j2k.Decoder(workers=1).decode_srgb8(data, control=bounded).to_numpy(),
            full.transpose(2, 0, 1))
        with self.assertRaisesRegex(ValueError, "resource"):
            decoder.decode_srgb8(data, control=j2k.DecodeControl(max_working_memory_bytes=1))
        token = j2k.CancellationToken()
        token.cancel()
        with self.assertRaisesRegex(ValueError, "cancel"):
            decoder.decode_srgb8(data, control=j2k.DecodeControl(cancellation=token))
        origin = (ROOT / "crates/j2k-test-support/fixtures/openjpeg_origin/rgb_23x17_origin_3_5_ht.j2c").read_bytes()
        with self.assertRaisesRegex(ValueError, "no samples"):
            decoder.plan_region(origin, region=(0, 0, 1, 1), reduction=1)

    def test_controlled_preparation_reports_limits_and_cancellation_per_input(self):
        data, other = deep_image(), deep_image(71)
        decoder = j2k.Decoder(layout="nhwc", workers=1)
        for inputs in ([data, other], [j2k.ImageRequest(data, reduction=4), j2k.ImageRequest(other, reduction=4)]):
            ordinary = decoder.prepare(inputs)
            controlled = decoder.prepare(inputs, control=j2k.DecodeControl(max_working_memory_bytes=64 << 20))
            self.assertEqual(controlled.errors, [])
            self.assertEqual(controlled.group_count, ordinary.group_count)
            expected, actual = rows_by_index(decoder.decode_prepared(ordinary)), rows_by_index(decoder.decode_prepared(controlled))
            self.assertEqual(sorted(actual), [0, 1])
            for index in actual:
                np.testing.assert_array_equal(actual[index][0], expected[index][0])
                self.assertEqual(actual[index][1], expected[index][1])
            starved = decoder.prepare(inputs, control=j2k.DecodeControl(max_working_memory_bytes=1))
            self.assertEqual([index for index, _ in starved.errors], [0, 1])
            self.assertTrue(all("resource" in message.lower() for _, message in starved.errors), starved.errors)
            token = j2k.CancellationToken()
            token.cancel()
            stopped = decoder.prepare(inputs, control=j2k.DecodeControl(cancellation=token))
            self.assertEqual([index for index, _ in stopped.errors], [0, 1])
            self.assertTrue(all("cancel" in message.lower() for _, message in stopped.errors), stopped.errors)
        # Full 65 x 67 RGB output is 13065 bytes; the reduced plan is 75.
        bounded = decoder.prepare([data, j2k.ImageRequest(other, reduction=4)], control=j2k.DecodeControl(max_output_bytes=75))
        self.assertEqual([index for index, _ in bounded.errors], [0])

    def test_deep_jpeg_requests_do_not_silently_use_native_j2k(self):
        # A valid deep J2K request can coexist with a rejected JPEG request.
        data = deep_image()
        jpeg = (ROOT / "crates/j2k-test-support/fixtures/conformance/baseline_420_16x16.jpg").read_bytes()
        batch = j2k.Decoder().decode_batch([j2k.ImageRequest(jpeg, reduction=4), j2k.ImageRequest(data, reduction=4)])
        self.assertEqual([index for index, _ in batch.errors], [0])
        self.assertEqual(batch.groups[0].source_indices, [1])


if __name__ == "__main__":
    unittest.main()
