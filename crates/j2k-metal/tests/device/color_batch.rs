// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn submitted_prepared_ht_rgb_u8_stores_exact_native_nhwc_and_nchw_for_all_requests() {
    if !should_run_metal_runtime() {
        return;
    }

    let first = Arc::<[u8]>::from(fixture_ht_rgb_u8_sized(8, 8, 0));
    let second = Arc::<[u8]>::from(fixture_ht_rgb_u8_sized(8, 8, 17));
    let roi = Rect {
        x: 2,
        y: 2,
        w: 4,
        h: 4,
    };
    let requests = [
        DecodeRequest::Full,
        DecodeRequest::Region { roi },
        DecodeRequest::Reduced {
            scale: Downscale::Half,
        },
        DecodeRequest::RegionReduced {
            roi,
            scale: Downscale::Half,
        },
    ];

    for layout in [BatchLayout::Nhwc, BatchLayout::Nchw] {
        for request in requests {
            let options = BatchDecodeOptions {
                layout,
                ..BatchDecodeOptions::default()
            };
            let inputs = vec![
                EncodedImage::new(first.clone(), request),
                EncodedImage::new(second.clone(), request),
            ];
            let mut cpu = CpuBatchDecoder::new(options);
            let expected = cpu.decode(inputs.clone()).expect("CPU RGB U8 oracle");
            let CpuBatchSamples::U8(expected) = expected.groups()[0].samples() else {
                panic!("six-bit RGB must use U8 batch storage")
            };

            let mut decoder = MetalBatchDecoder::system_default_with_options(options)
                .expect("persistent Metal decoder");
            let prepared = decoder
                .prepare(inputs)
                .expect("prepare exact-native HT RGB U8 group");
            assert!(prepared.errors().is_empty());
            assert_eq!(prepared.groups().len(), 1);
            let group = &prepared.groups()[0];
            assert_eq!(group.info().sample_type, NativeSampleType::U8);
            let (width, height) = group.info().dimensions;
            let samples_per_image =
                usize::try_from(width).unwrap() * usize::try_from(height).unwrap() * 3;
            let output_len = samples_per_image * group.images().len();
            let buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(
                decoder.backend_session().device(),
                output_len + 8,
            )
            .expect("RGB U8 destination buffer");
            let row_bytes = usize::try_from(width).unwrap() * 3;
            let destination_layout = MetalImageLayout::new_batch(
                4,
                (width, height),
                row_bytes,
                PixelFormat::Rgb8,
                group.images().len(),
                samples_per_image,
            )
            .expect("dense RGB U8 destination layout");
            // SAFETY: this fresh allocation is retained exclusively by the
            // pending submission until its explicit completion wait.
            let destination = unsafe {
                MetalImageDestination::from_exclusive_buffer(buffer.clone(), destination_layout)
                    .expect("RGB U8 destination")
            };
            decoder
                .submit_prepared_group_into(group, destination)
                .expect("submit exact-native HT RGB U8 group")
                .wait()
                .expect("complete exact-native HT RGB U8 group");

            // SAFETY: codec completion released exclusive destination access.
            let actual =
                unsafe { j2k_metal_support::checked_buffer_read_vec::<u8>(&buffer, 4, output_len) }
                    .expect("RGB U8 destination samples");
            assert_eq!(
                actual.as_slice(),
                expected.as_slice(),
                "{layout:?} {request:?}"
            );
            assert!(
                actual.iter().all(|sample| *sample <= 0x3f),
                "six-bit samples must not be scaled to the full U8 range"
            );
        }
    }
}

#[test]
fn submitted_prepared_ht_rgb_u16_stores_exact_native_nhwc_and_nchw() {
    if !should_run_metal_runtime() {
        return;
    }

    let first = Arc::<[u8]>::from(fixture_ht_rgb_u16_sized(8, 8, 0));
    let second = Arc::<[u8]>::from(fixture_ht_rgb_u16_sized(8, 8, 257));
    for layout in [BatchLayout::Nhwc, BatchLayout::Nchw] {
        let options = BatchDecodeOptions {
            layout,
            ..BatchDecodeOptions::default()
        };
        let inputs = vec![
            EncodedImage::full(first.clone()),
            EncodedImage::full(second.clone()),
        ];
        let mut cpu = CpuBatchDecoder::new(options);
        let expected = cpu.decode(inputs.clone()).expect("CPU RGB U16 oracle");
        let CpuBatchSamples::U16(expected) = expected.groups()[0].samples() else {
            panic!("twelve-bit RGB must use U16 batch storage")
        };

        let mut decoder = MetalBatchDecoder::system_default_with_options(options)
            .expect("persistent Metal decoder");
        let prepared = decoder
            .prepare(inputs)
            .expect("prepare exact-native HT RGB U16 group");
        assert!(prepared.errors().is_empty());
        let group = &prepared.groups()[0];
        let (width, height) = group.info().dimensions;
        let samples_per_image =
            usize::try_from(width).unwrap() * usize::try_from(height).unwrap() * 3;
        let output_len = samples_per_image * group.images().len();
        let buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(
            decoder.backend_session().device(),
            output_len * 2 + 8,
        )
        .expect("RGB U16 destination buffer");
        let row_bytes = usize::try_from(width).unwrap() * 3 * 2;
        let destination_layout = MetalImageLayout::new_batch(
            4,
            (width, height),
            row_bytes,
            PixelFormat::Rgb16,
            group.images().len(),
            samples_per_image * 2,
        )
        .expect("dense RGB U16 destination layout");
        // SAFETY: this fresh allocation remains exclusive until completion.
        let destination = unsafe {
            MetalImageDestination::from_exclusive_buffer(buffer.clone(), destination_layout)
                .expect("RGB U16 destination")
        };
        decoder
            .submit_prepared_group_into(group, destination)
            .expect("submit exact-native HT RGB U16 group")
            .wait()
            .expect("complete exact-native HT RGB U16 group");

        // SAFETY: codec completion released exclusive destination access.
        let actual =
            unsafe { j2k_metal_support::checked_buffer_read_vec::<u16>(&buffer, 4, output_len) }
                .expect("RGB U16 destination samples");
        assert_eq!(actual.as_slice(), expected.as_slice(), "{layout:?}");
        assert!(
            actual.iter().all(|sample| *sample <= 0x0fff),
            "twelve-bit samples must not be scaled to the full U16 range"
        );
    }
}

fn assert_signed_rgb_request(
    name: &str,
    encoded: &Arc<[u8]>,
    layout: BatchLayout,
    request: DecodeRequest,
) {
    let options = BatchDecodeOptions {
        layout,
        ..BatchDecodeOptions::default()
    };
    let inputs = vec![
        EncodedImage::new(encoded.clone(), request),
        EncodedImage::new(encoded.clone(), request),
    ];
    let mut cpu = CpuBatchDecoder::new(options);
    let expected = cpu
        .decode(inputs.clone())
        .unwrap_or_else(|error| panic!("{name} CPU oracle: {error}"));
    let CpuBatchSamples::I16(expected) = expected.groups()[0].samples() else {
        panic!("{name} must use I16 batch storage")
    };

    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options).expect("persistent Metal decoder");
    let prepared = decoder
        .prepare(inputs)
        .unwrap_or_else(|error| panic!("{name} prepare: {error}"));
    assert!(prepared.errors().is_empty(), "{name}");
    assert_eq!(prepared.groups().len(), 1, "{name}");
    let group = &prepared.groups()[0];
    assert_eq!(group.info().sample_type, NativeSampleType::I16);
    assert!(group
        .images()
        .iter()
        .all(|image| image.preparation_depth() == PreparationDepth::Htj2kOffsetPlan));
    let (width, height) = group.info().dimensions;
    let samples_per_image = usize::try_from(width).unwrap() * usize::try_from(height).unwrap() * 3;
    let output_len = samples_per_image * group.images().len();
    let buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(
        decoder.backend_session().device(),
        output_len * 2 + 8,
    )
    .expect("signed RGB destination buffer");
    let destination_layout = MetalImageLayout::new_batch(
        4,
        (width, height),
        usize::try_from(width).unwrap() * 3 * 2,
        PixelFormat::RgbI16,
        group.images().len(),
        samples_per_image * 2,
    )
    .expect("dense signed RGB destination layout");
    // SAFETY: this fresh allocation remains exclusively owned by the pending
    // codec submission until its completion wait.
    let destination = unsafe {
        MetalImageDestination::from_exclusive_buffer(buffer.clone(), destination_layout)
            .expect("signed RGB destination")
    };
    decoder
        .submit_prepared_group_into(group, destination)
        .unwrap_or_else(|error| panic!("{name} submit: {error}"))
        .wait()
        .unwrap_or_else(|error| panic!("{name} completion: {error}"));

    // SAFETY: completion released exclusive destination access.
    let actual =
        unsafe { j2k_metal_support::checked_buffer_read_vec::<i16>(&buffer, 4, output_len) }
            .expect("signed RGB destination samples");
    assert_eq!(
        actual.as_slice(),
        expected.as_slice(),
        "{name} {layout:?} {request:?}"
    );
}

#[test]
fn independent_openjph_signed_rgb_stores_exact_i16_for_all_requests_and_layouts() {
    if !should_run_metal_runtime() {
        return;
    }

    let roi = Rect {
        x: 3,
        y: 2,
        w: 9,
        h: 7,
    };
    let requests = [
        DecodeRequest::Full,
        DecodeRequest::Region { roi },
        DecodeRequest::Reduced {
            scale: Downscale::Half,
        },
        DecodeRequest::RegionReduced {
            roi,
            scale: Downscale::Half,
        },
    ];
    let fixtures = j2k_test_support::openjph_batch_fixtures()
        .iter()
        .filter(|fixture| {
            matches!(
                fixture.name,
                "openjph-rgb-s8-53-single-raw"
                    | "openjph-rgb-s12-53-single-raw"
                    | "openjph-rgb-s16-53-single-raw"
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(fixtures.len(), 3);

    for fixture in fixtures {
        let encoded = Arc::<[u8]>::from(fixture.encoded);
        for layout in [BatchLayout::Nhwc, BatchLayout::Nchw] {
            for request in requests {
                assert_signed_rgb_request(fixture.name, &encoded, layout, request);
            }
        }
    }
}

#[test]
fn dropped_pending_prepared_ht_rgb_group_reuses_session_and_prepared_plan() {
    if !should_run_metal_runtime() {
        return;
    }

    let first = Arc::<[u8]>::from(fixture_ht_rgb_u8_sized(8, 8, 0));
    let second = Arc::<[u8]>::from(fixture_ht_rgb_u8_sized(8, 8, 17));
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nchw,
        ..BatchDecodeOptions::default()
    };
    let inputs = vec![
        EncodedImage::full(first.clone()),
        EncodedImage::full(second.clone()),
    ];
    let mut cpu = CpuBatchDecoder::new(options);
    let expected = cpu.decode(inputs.clone()).expect("CPU RGB reuse oracle");
    let CpuBatchSamples::U8(expected) = expected.groups()[0].samples() else {
        panic!("six-bit RGB must use U8 batch storage")
    };
    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options).expect("persistent Metal decoder");
    let prepared = decoder.prepare(inputs).expect("prepare reusable RGB group");
    let group = &prepared.groups()[0];
    let (width, height) = group.info().dimensions;
    let image_len = width as usize * height as usize * 3;
    let layout = MetalImageLayout::new_batch(
        4,
        (width, height),
        width as usize * 3,
        PixelFormat::Rgb8,
        2,
        image_len,
    )
    .expect("reusable RGB group layout");

    let dropped_buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(
        decoder.backend_session().device(),
        image_len * 2 + 8,
    )
    .expect("dropped RGB destination");
    // SAFETY: the fresh range is retained exclusively by the pending owner.
    let dropped_destination = unsafe {
        MetalImageDestination::from_exclusive_buffer(dropped_buffer, layout)
            .expect("dropped RGB destination guard")
    };
    drop(
        decoder
            .submit_prepared_group_into(group, dropped_destination)
            .expect("submit disposable RGB group"),
    );

    let completed_buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(
        decoder.backend_session().device(),
        image_len * 2 + 8,
    )
    .expect("completed RGB destination");
    // SAFETY: the fresh range stays exclusive through explicit completion.
    let completed_destination = unsafe {
        MetalImageDestination::from_exclusive_buffer(completed_buffer.clone(), layout)
            .expect("completed RGB destination guard")
    };
    decoder
        .submit_prepared_group_into(group, completed_destination)
        .expect("reuse RGB session after pending drop")
        .wait()
        .expect("complete reused RGB group");

    // SAFETY: completion released the exclusive destination owner.
    let actual = unsafe {
        j2k_metal_support::checked_buffer_read_vec::<u8>(&completed_buffer, 4, image_len * 2)
    }
    .expect("completed RGB samples");
    assert_eq!(actual.as_slice(), expected.as_slice());
    assert_eq!(
        decoder.submissions().expect("RGB submission count"),
        2,
        "a dropped pending group must retire exactly once and leave the session reusable"
    );
}
