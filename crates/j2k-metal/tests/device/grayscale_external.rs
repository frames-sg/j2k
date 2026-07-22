// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn submitted_prepared_sub8_ht_grayscale_preserves_native_samples_for_single_and_stacked_groups() {
    if !should_run_metal_runtime() {
        return;
    }

    let first_pixels = (0_u8..16).collect::<Vec<_>>();
    let second_pixels = (0_u8..16).rev().collect::<Vec<_>>();
    let encode_fixture = |pixels: &[u8]| {
        encode_htj2k(
            pixels,
            4,
            4,
            1,
            4,
            false,
            &EncodeOptions {
                reversible: true,
                num_decomposition_levels: 1,
                ..EncodeOptions::default()
            },
        )
        .expect("encode reversible HT Gray4 fixture")
    };
    let encoded = [
        Arc::<[u8]>::from(encode_fixture(&first_pixels)),
        Arc::<[u8]>::from(encode_fixture(&second_pixels)),
    ];
    let options = BatchDecodeOptions::default();
    let mut cpu = CpuBatchDecoder::new(options);
    let mut metal =
        MetalBatchDecoder::system_default_with_options(options).expect("persistent Metal decoder");
    let mut expected_groups = Vec::new();
    let mut actual_groups = Vec::new();

    for batch_len in [1_usize, 2] {
        let inputs = encoded
            .iter()
            .take(batch_len)
            .cloned()
            .map(EncodedImage::full)
            .collect::<Vec<_>>();
        let expected = cpu.decode(inputs.clone()).expect("CPU Gray4 oracle");
        assert!(expected.errors().is_empty());
        assert_eq!(expected.groups().len(), 1);
        let expected_group = &expected.groups()[0];
        assert_eq!(expected_group.info().precision, 4);
        let CpuBatchSamples::U8(expected_samples) = expected_group.samples() else {
            panic!("Gray4 must use native U8 batch storage")
        };
        expected_groups.push(expected_samples.clone());

        let prepared = metal.prepare(inputs).expect("prepare Metal Gray4 group");
        assert!(prepared.errors().is_empty());
        assert_eq!(prepared.groups().len(), 1);
        assert_eq!(prepared.groups()[0].info().precision, 4);
        let output_len = 16 * batch_len;
        let buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(
            metal.backend_session().device(),
            output_len,
        )
        .expect("Gray4 destination buffer");
        let layout = MetalImageLayout::new_batch(0, (4, 4), 4, PixelFormat::Gray8, batch_len, 16)
            .expect("Gray4 destination layout");
        // SAFETY: the pending submission exclusively retains this fresh range.
        let destination = unsafe {
            MetalImageDestination::from_exclusive_buffer(buffer.clone(), layout)
                .expect("Gray4 destination")
        };
        metal
            .submit_prepared_group_into(&prepared.groups()[0], destination)
            .expect("submit prepared Gray4 group")
            .wait()
            .expect("complete prepared Gray4 group");

        // SAFETY: codec completion released exclusive destination access.
        actual_groups.push(
            unsafe { j2k_metal_support::checked_buffer_read_vec::<u8>(&buffer, 0, output_len) }
                .expect("Gray4 output samples"),
        );
    }

    assert_eq!(actual_groups, expected_groups);
}

#[test]
fn submitted_prepared_ht_grayscale_roi_and_reduction_match_cpu_oracle() {
    if !should_run_metal_runtime() {
        return;
    }

    let encoded = Arc::<[u8]>::from(j2k_test_support::openhtj2k_refinement_odd_fixture());
    let roi = Rect {
        x: 3,
        y: 5,
        w: 9,
        h: 21,
    };
    let requests = [
        DecodeRequest::Region { roi },
        DecodeRequest::Reduced {
            scale: Downscale::Half,
        },
        DecodeRequest::RegionReduced {
            roi,
            scale: Downscale::Half,
        },
    ];

    for request in requests {
        let options = BatchDecodeOptions::default();
        let mut cpu = CpuBatchDecoder::new(options);
        let expected = cpu
            .decode(vec![EncodedImage::new(encoded.clone(), request)])
            .expect("CPU request oracle");
        let CpuBatchSamples::U8(expected) = expected.groups()[0].samples() else {
            panic!("odd OpenHT fixture must decode to U8")
        };

        let mut decoder = MetalBatchDecoder::system_default_with_options(options)
            .expect("persistent Metal decoder");
        let prepared = decoder
            .prepare(vec![EncodedImage::new(encoded.clone(), request)])
            .expect("prepare Metal request");
        assert!(prepared.errors().is_empty());
        assert_eq!(prepared.groups().len(), 1);
        let group = &prepared.groups()[0];
        let (width, height) = group.info().dimensions;
        let image_len = usize::try_from(width)
            .unwrap()
            .checked_mul(usize::try_from(height).unwrap())
            .unwrap();
        let buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(
            decoder.backend_session().device(),
            image_len + 8,
        )
        .expect("request destination buffer");
        let layout = MetalImageLayout::new_batch(
            4,
            (width, height),
            width as usize,
            PixelFormat::Gray8,
            1,
            image_len,
        )
        .expect("request destination layout");
        // SAFETY: the pending submission exclusively retains this fresh range.
        let destination = unsafe {
            MetalImageDestination::from_exclusive_buffer(buffer.clone(), layout)
                .expect("request destination")
        };
        decoder
            .submit_prepared_group_into(group, destination)
            .expect("submit prepared request")
            .wait()
            .expect("complete prepared request");

        // SAFETY: codec completion released exclusive destination access.
        let actual =
            unsafe { j2k_metal_support::checked_buffer_read_vec::<u8>(&buffer, 4, image_len) }
                .expect("request pixels");
        assert_eq!(
            actual.as_slice(),
            expected.as_slice(),
            "request {request:?}"
        );
    }
}

#[test]
fn independent_openht_sigprop_overlap_matches_openht_oracle_within_one_lsb() {
    const WIDTH: u32 = 512;
    const HEIGHT: u32 = 64;

    if !should_run_metal_runtime() {
        return;
    }
    let encoded = Arc::<[u8]>::from(j2k_test_support::openhtj2k_sigprop_overlap_fixture());
    let expected = j2k_test_support::openhtj2k_sigprop_overlap_pixels();
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options).expect("persistent Metal decoder");
    let prepared = decoder
        .prepare(vec![EncodedImage::full(encoded.clone())])
        .expect("prepare independent SigProp-overlap fixture");
    assert!(
        prepared.errors().is_empty(),
        "prepare errors: {:?}",
        prepared.errors()
    );
    assert_eq!(prepared.groups().len(), 1);
    let group = &prepared.groups()[0];
    assert_eq!(group.info().route, BatchCodecRoute::Htj2k);
    assert_eq!(group.info().dimensions, (WIDTH, HEIGHT));

    let buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(
        decoder.backend_session().device(),
        expected.len() + 8,
    )
    .expect("SigProp-overlap destination buffer");
    let layout = MetalImageLayout::new_batch(
        4,
        (WIDTH, HEIGHT),
        WIDTH as usize * 3,
        PixelFormat::Rgb8,
        1,
        expected.len(),
    )
    .expect("SigProp-overlap destination layout");
    // SAFETY: this fresh range remains exclusively retained by the submitted
    // codec work until its explicit completion wait.
    let destination = unsafe {
        MetalImageDestination::from_exclusive_buffer(buffer.clone(), layout)
            .expect("SigProp-overlap destination")
    };
    decoder
        .submit_prepared_group_into(group, destination)
        .expect("submit independent SigProp-overlap fixture")
        .wait()
        .expect("complete independent SigProp-overlap fixture");

    // SAFETY: codec completion released exclusive destination access.
    let actual = unsafe {
        j2k_metal_support::checked_buffer_read_vec::<u8>(&buffer, 4, expected.len())
            .expect("SigProp-overlap pixels")
    };
    let mut cpu = vec![0_u8; expected.len()];
    J2kDecoder::new(&encoded)
        .expect("SigProp-overlap CPU decoder")
        .decode_into(&mut cpu, WIDTH as usize * 3, PixelFormat::Rgb8)
        .expect("SigProp-overlap CPU decode");
    assert_eq!(actual, cpu, "Metal must match the corrected scalar reader");
    let max_difference = actual
        .iter()
        .zip(expected.iter())
        .map(|(&actual, &expected)| actual.abs_diff(expected))
        .max()
        .unwrap_or(0);
    assert!(
        max_difference <= 1,
        "Metal differs from the independent OpenHTJ2K oracle by {max_difference} LSB"
    );
}

#[test]
fn submitted_prepared_classic_grayscale_writes_external_group_without_staging() {
    if !should_run_metal_runtime() {
        return;
    }

    let mut decoder = MetalBatchDecoder::system_default().expect("persistent Metal decoder");
    let prepared = decoder
        .prepare(vec![EncodedImage::full(Arc::<[u8]>::from(fixture_gray8()))])
        .expect("prepare classic grayscale group");
    assert!(prepared.errors().is_empty());
    assert_eq!(prepared.groups()[0].info().route, BatchCodecRoute::Classic);

    let buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(
        decoder.backend_session().device(),
        24,
    )
    .expect("classic destination buffer");
    let layout = MetalImageLayout::new_batch(4, (4, 4), 4, PixelFormat::Gray8, 1, 16)
        .expect("classic destination layout");
    // SAFETY: the pending submission exclusively retains this fresh range.
    let destination = unsafe {
        MetalImageDestination::from_exclusive_buffer(buffer.clone(), layout)
            .expect("classic destination")
    };
    decoder
        .submit_prepared_group_into(&prepared.groups()[0], destination)
        .expect("submit classic prepared group")
        .wait()
        .expect("complete classic prepared group");

    // SAFETY: codec completion released exclusive destination access.
    let actual = unsafe { j2k_metal_support::checked_buffer_read_vec::<u8>(&buffer, 4, 16) }
        .expect("classic pixels");
    assert_eq!(actual, (0..16).collect::<Vec<u8>>());
}

#[test]
fn submitted_prepared_classic_signed_gray12_preserves_native_i16_samples() {
    if !should_run_metal_runtime() {
        return;
    }

    let (encoded, expected) = fixture_classic_signed_gray12();
    let mut decoder = MetalBatchDecoder::system_default().expect("persistent Metal decoder");
    let prepared = decoder
        .prepare(vec![EncodedImage::full(Arc::<[u8]>::from(encoded))])
        .expect("prepare classic signed group");
    assert_eq!(
        prepared.groups()[0].info().sample_type,
        NativeSampleType::I16
    );

    let buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(
        decoder.backend_session().device(),
        40,
    )
    .expect("classic signed destination buffer");
    let layout = MetalImageLayout::new_batch(4, (4, 4), 8, PixelFormat::GrayI16, 1, 32)
        .expect("classic signed destination layout");
    // SAFETY: the pending submission exclusively retains this fresh range.
    let destination = unsafe {
        MetalImageDestination::from_exclusive_buffer(buffer.clone(), layout)
            .expect("classic signed destination")
    };
    decoder
        .submit_prepared_group_into(&prepared.groups()[0], destination)
        .expect("submit classic signed group")
        .wait()
        .expect("complete classic signed group");

    // SAFETY: codec completion released exclusive destination access.
    let bytes = unsafe { j2k_metal_support::checked_buffer_read_vec::<u8>(&buffer, 4, 32) }
        .expect("classic signed pixels");
    let actual = bytes
        .chunks_exact(2)
        .map(|sample| i16::from_le_bytes([sample[0], sample[1]]))
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

#[test]
fn dropped_pending_metal_batch_can_be_followed_by_a_successful_decode() {
    if !should_run_metal_runtime() {
        return;
    }

    let bytes = Arc::<[u8]>::from(fixture_ht_gray8());
    let mut decoder = MetalBatchDecoder::system_default().expect("persistent Metal decoder");
    let pending = decoder
        .submit_batch(vec![
            EncodedImage::full(bytes.clone()),
            EncodedImage::full(bytes.clone()),
        ])
        .expect("pending batch");
    assert_eq!(pending.len(), 1);
    drop(pending);

    let result = decoder
        .decode_batch(vec![
            EncodedImage::full(bytes.clone()),
            EncodedImage::full(bytes),
        ])
        .expect("decoder reuse after pending drop");
    assert!(result.errors().is_empty());
    assert!(result.group_errors().is_empty());
    assert_eq!(result.groups().len(), 1);
    assert_eq!(result.groups()[0].surfaces().len(), 2);
    assert!(result.groups()[0]
        .surfaces()
        .iter()
        .all(|surface| surface.residency() == SurfaceResidency::MetalResidentDecode));
}
