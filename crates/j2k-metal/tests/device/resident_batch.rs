// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn prepared_ht_rgb_u8_nhwc_resident_group_is_exact_and_uses_one_allocation() {
    if !should_run_metal_runtime() {
        return;
    }

    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let roi = Rect {
        x: 2,
        y: 2,
        w: 4,
        h: 4,
    };
    let inputs = vec![
        EncodedImage::new(
            Arc::from(fixture_ht_rgb_u8_sized(8, 8, 0)),
            DecodeRequest::RegionReduced {
                roi,
                scale: Downscale::Half,
            },
        ),
        EncodedImage::new(
            Arc::from(fixture_ht_rgb_u8_sized(8, 8, 17)),
            DecodeRequest::RegionReduced {
                roi,
                scale: Downscale::Half,
            },
        ),
    ];
    let mut cpu = CpuBatchDecoder::new(options);
    let expected = cpu.decode(inputs.clone()).expect("CPU RGB U8 oracle");
    let CpuBatchSamples::U8(expected) = expected.groups()[0].samples() else {
        panic!("six-bit RGB must use U8 batch storage")
    };

    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options).expect("persistent Metal decoder");
    let prepared = decoder
        .prepare(inputs)
        .expect("prepare resident HT RGB U8 group");
    let result = decoder
        .decode_prepared(&prepared)
        .expect("decode resident HT RGB U8 group");
    assert!(result.group_errors().is_empty());
    let group = &result.groups()[0];
    assert_eq!(group.info().layout, BatchLayout::Nhwc);
    assert_eq!(group.surfaces().len(), 2);
    let image_bytes = expected.len() / group.surfaces().len();
    for (index, surface) in group.surfaces().iter().enumerate() {
        assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
        assert_eq!(
            surface.as_bytes().expect("resident RGB U8 bytes").as_ref(),
            &expected[index * image_bytes..(index + 1) * image_bytes]
        );
    }
    let first_range = group.surfaces()[0]
        .memory_range()
        .expect("first resident RGB U8 range");
    let second_range = group.surfaces()[1]
        .memory_range()
        .expect("second resident RGB U8 range");
    assert_eq!(first_range.allocation, second_range.allocation);
    assert_eq!(first_range.offset, 0);
    assert_eq!(second_range.offset, image_bytes);
    assert_eq!(first_range.len, image_bytes);
    assert_eq!(second_range.len, image_bytes);
}

#[test]
fn prepared_ht_rgb_u16_nhwc_resident_group_is_exact_and_uses_one_allocation() {
    if !should_run_metal_runtime() {
        return;
    }

    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let inputs = vec![
        EncodedImage::full(Arc::from(fixture_ht_rgb_u16_sized(8, 8, 0))),
        EncodedImage::full(Arc::from(fixture_ht_rgb_u16_sized(8, 8, 257))),
    ];
    let mut cpu = CpuBatchDecoder::new(options);
    let expected = cpu.decode(inputs.clone()).expect("CPU RGB U16 oracle");
    let CpuBatchSamples::U16(expected) = expected.groups()[0].samples() else {
        panic!("twelve-bit RGB must use U16 batch storage")
    };
    let expected_bytes = expected
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect::<Vec<_>>();

    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options).expect("persistent Metal decoder");
    let prepared = decoder
        .prepare(inputs)
        .expect("prepare resident HT RGB U16 group");
    let result = decoder
        .decode_prepared(&prepared)
        .expect("decode resident HT RGB U16 group");
    assert!(result.group_errors().is_empty());
    let group = &result.groups()[0];
    assert_eq!(group.info().layout, BatchLayout::Nhwc);
    assert_eq!(group.surfaces().len(), 2);
    let image_bytes = expected_bytes.len() / group.surfaces().len();
    for (index, surface) in group.surfaces().iter().enumerate() {
        assert_eq!(surface.pixel_format(), PixelFormat::Rgb16);
        assert_eq!(
            surface.as_bytes().expect("resident RGB U16 bytes").as_ref(),
            &expected_bytes[index * image_bytes..(index + 1) * image_bytes]
        );
    }
    let first_range = group.surfaces()[0]
        .memory_range()
        .expect("first resident RGB U16 range");
    let second_range = group.surfaces()[1]
        .memory_range()
        .expect("second resident RGB U16 range");
    assert_eq!(first_range.allocation, second_range.allocation);
    assert_eq!(first_range.offset, 0);
    assert_eq!(second_range.offset, image_bytes);
    assert_eq!(first_range.len, image_bytes);
    assert_eq!(second_range.len, image_bytes);
}

#[test]
fn metal_prepared_batch_continues_after_one_group_execution_failure() {
    if !should_run_metal_runtime() {
        return;
    }

    let options = BatchDecodeOptions {
        layout: BatchLayout::Nchw,
        ..BatchDecodeOptions::default()
    };
    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options).expect("persistent Metal decoder");
    let prepared = decoder
        .prepare(vec![
            EncodedImage::full(unsupported_classic_roi_rgb()),
            EncodedImage::full(Arc::from(fixture_ht_gray8())),
        ])
        .expect("prepare two distinct Metal groups");
    assert_eq!(prepared.groups().len(), 2);

    let result = decoder
        .decode_prepared(&prepared)
        .expect("unsupported RGN group must not abort the reusable Metal session");
    assert!(result.errors().is_empty());
    assert_eq!(result.groups().len(), 1);
    assert_eq!(result.groups()[0].source_indices(), &[1]);
    assert_eq!(result.group_errors().len(), 1);
    assert_eq!(result.group_errors()[0].source_indices(), &[0]);
    assert!(matches!(
        result.group_errors()[0].source(),
        Error::UnsupportedMetalRequest { .. }
    ));
}

#[test]
fn prepared_ht_rgb_resident_session_reuse_soak_stabilizes_scratch_retention() {
    if !should_run_metal_runtime() {
        return;
    }

    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options).expect("persistent Metal decoder");
    let prepared = decoder
        .prepare(vec![EncodedImage::full(Arc::from(
            fixture_ht_rgb_u8_sized(8, 8, 0),
        ))])
        .expect("prepare reusable resident HT RGB group");

    for _ in 0..16 {
        let result = decoder
            .decode_prepared(&prepared)
            .expect("warm resident HT RGB session");
        assert!(result.group_errors().is_empty());
    }
    let stable = decoder
        .backend_session()
        .buffer_pool_diagnostics()
        .expect("warm scratch-pool diagnostics");
    let retention = |diagnostics: j2k_metal::MetalBufferPoolsDiagnostics| {
        (
            diagnostics.private.cached_bytes,
            diagnostics.private.cached_buffers,
            diagnostics.private.metadata_capacity,
            diagnostics.private.peak_cached_bytes,
            diagnostics.private.peak_cached_buffers,
            diagnostics.shared.cached_bytes,
            diagnostics.shared.cached_buffers,
            diagnostics.shared.metadata_capacity,
            diagnostics.shared.peak_cached_bytes,
            diagnostics.shared.peak_cached_buffers,
        )
    };
    let expected_retention = retention(stable);

    for iteration in 0..1_000 {
        let result = decoder
            .decode_prepared(&prepared)
            .expect("soak resident HT RGB session");
        assert!(result.group_errors().is_empty());
        drop(result);
        if iteration % 100 == 99 {
            let diagnostics = decoder
                .backend_session()
                .buffer_pool_diagnostics()
                .expect("periodic scratch-pool diagnostics");
            assert_eq!(
                retention(diagnostics),
                expected_retention,
                "scratch retention grew after warmup at submission {}",
                iteration + 1
            );
        }
    }
}
