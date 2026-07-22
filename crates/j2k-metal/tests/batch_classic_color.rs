// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(target_os = "macos")]

use std::sync::Arc;

use j2k::{
    BatchColor, BatchDecodeOptions, BatchLayout, CpuBatchDecoder, CpuBatchSamples, DecodeRequest,
    EncodedImage, NativeSampleType,
};
use j2k_core::{Downscale, PixelFormat, Rect};
use j2k_metal::{MetalBatchDecoder, MetalImageDestination, MetalImageLayout};
use j2k_native::{encode, EncodeOptions};

#[derive(Debug, Clone, Copy)]
enum ClassicRgbProfile {
    U8,
    U8Irreversible97,
    U12,
    I12,
}

fn should_run_metal_runtime() -> bool {
    j2k_test_support::metal_runtime_gate(module_path!())
}

fn classic_rgb_fixture(profile: ClassicRgbProfile) -> Vec<u8> {
    const WIDTH: u32 = 8;
    const HEIGHT: u32 = 8;
    let pixel_count = usize::try_from(WIDTH)
        .expect("fixture width fits usize")
        .checked_mul(usize::try_from(HEIGHT).expect("fixture height fits usize"))
        .expect("fixture area fits usize");
    let (pixels, bit_depth, signed, reversible) = match profile {
        ClassicRgbProfile::U8 | ClassicRgbProfile::U8Irreversible97 => {
            let mut pixels = Vec::with_capacity(pixel_count * 3);
            for y in 0..HEIGHT {
                for x in 0..WIDTH {
                    for value in [x * 17 + y * 3, x * 5 + y * 19 + 7, x * 11 + y * 13 + 29] {
                        pixels.push(value.to_le_bytes()[0]);
                    }
                }
            }
            (pixels, 8, false, matches!(profile, ClassicRgbProfile::U8))
        }
        ClassicRgbProfile::U12 => {
            let mut pixels = Vec::with_capacity(pixel_count * 3 * 2);
            for y in 0..HEIGHT {
                for x in 0..WIDTH {
                    for value in [
                        x * 193 + y * 257,
                        x * 313 + y * 97 + 31,
                        x * 71 + y * 401 + 63,
                    ] {
                        let value = u16::try_from(value & 0x0fff).expect("twelve-bit RGB sample");
                        pixels.extend_from_slice(&value.to_le_bytes());
                    }
                }
            }
            (pixels, 12, false, true)
        }
        ClassicRgbProfile::I12 => {
            let mut pixels = Vec::with_capacity(pixel_count * 3 * 2);
            for y in 0..HEIGHT {
                for x in 0..WIDTH {
                    for value in [
                        x * 193 + y * 257,
                        x * 313 + y * 97 + 31,
                        x * 71 + y * 401 + 63,
                    ] {
                        let value = i16::try_from(
                            i32::try_from(value & 0x0fff).expect("twelve-bit RGB sample fits i32")
                                - 2048,
                        )
                        .expect("signed twelve-bit RGB sample");
                        pixels.extend_from_slice(&value.to_le_bytes());
                    }
                }
            }
            (pixels, 12, true, true)
        }
    };
    encode(
        &pixels,
        WIDTH,
        HEIGHT,
        3,
        bit_depth,
        signed,
        &EncodeOptions {
            reversible,
            use_mct: true,
            num_decomposition_levels: 2,
            ..EncodeOptions::default()
        },
    )
    .unwrap_or_else(|error| panic!("encode {profile:?} classic RGB fixture: {error}"))
}

fn rgb_format(sample_type: NativeSampleType) -> PixelFormat {
    match sample_type {
        NativeSampleType::U8 => PixelFormat::Rgb8,
        NativeSampleType::U16 => PixelFormat::Rgb16,
        NativeSampleType::I16 => PixelFormat::RgbI16,
        _ => panic!("unsupported RGB sample type"),
    }
}

fn assert_native_samples(actual: &[u8], expected: &CpuBatchSamples, max_lsb_diff: u8) {
    match expected {
        CpuBatchSamples::U8(expected) => assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(actual, expected)| actual.abs_diff(*expected) <= max_lsb_diff),
            "U8 reconstruction differs by more than {max_lsb_diff} LSB"
        ),
        CpuBatchSamples::U16(expected) => {
            let actual = actual
                .chunks_exact(2)
                .map(|sample| u16::from_le_bytes([sample[0], sample[1]]))
                .collect::<Vec<_>>();
            assert!(
                actual
                    .iter()
                    .zip(expected)
                    .all(|(actual, expected)| actual.abs_diff(*expected) <= u16::from(max_lsb_diff)),
                "U16 reconstruction differs by more than {max_lsb_diff} LSB"
            );
        }
        CpuBatchSamples::I16(expected) => {
            let actual = actual
                .chunks_exact(2)
                .map(|sample| i16::from_le_bytes([sample[0], sample[1]]))
                .collect::<Vec<_>>();
            assert!(
                actual
                    .iter()
                    .zip(expected)
                    .all(|(actual, expected)| actual.abs_diff(*expected) <= u16::from(max_lsb_diff)),
                "I16 reconstruction differs by more than {max_lsb_diff} LSB"
            );
        }
        _ => panic!("unsupported CPU RGB oracle type"),
    }
}

fn assert_classic_rgb_layout(
    profile: ClassicRgbProfile,
    encoded: &Arc<[u8]>,
    layout: BatchLayout,
    requests: &[DecodeRequest],
) {
    let options = BatchDecodeOptions {
        layout,
        ..BatchDecodeOptions::default()
    };
    let max_lsb_diff = u8::from(matches!(profile, ClassicRgbProfile::U8Irreversible97));
    let inputs = requests
        .iter()
        .copied()
        .map(|request| EncodedImage::new(encoded.clone(), request))
        .collect::<Vec<_>>();
    let mut cpu = CpuBatchDecoder::new(options);
    let expected = cpu.decode(inputs.clone()).expect("CPU classic RGB oracle");
    assert!(
        expected.errors().is_empty(),
        "{profile:?} {layout:?} CPU errors: {:?}",
        expected.errors()
    );

    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options).expect("persistent Metal decoder");
    let prepared = decoder
        .prepare(inputs)
        .expect("prepare classic Metal RGB request matrix");
    assert!(
        prepared.errors().is_empty(),
        "{profile:?} {layout:?} prepare errors: {:?}",
        prepared.errors()
    );

    for group in prepared.groups() {
        assert_eq!(group.info().color, BatchColor::Rgb);
        let expected_group = expected
            .groups()
            .iter()
            .find(|expected| expected.source_indices() == group.source_indices())
            .expect("matching CPU classic RGB group");
        let fmt = rgb_format(group.info().sample_type);
        let bytes_per_sample = fmt.bytes_per_sample();
        let (width, height) = group.info().dimensions;
        let row_bytes =
            usize::try_from(width).expect("classic RGB width fits usize") * 3 * bytes_per_sample;
        let image_bytes =
            row_bytes * usize::try_from(height).expect("classic RGB height fits usize");
        let output_bytes = image_bytes * group.images().len();
        let buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(
            decoder.backend_session().device(),
            output_bytes + 8,
        )
        .expect("classic RGB destination buffer");
        let destination_layout = MetalImageLayout::new_batch(
            4,
            (width, height),
            row_bytes,
            fmt,
            group.images().len(),
            image_bytes,
        )
        .expect("classic RGB destination layout");
        // SAFETY: this fresh output range remains exclusively retained by the
        // pending codec submission until completion.
        let destination = unsafe {
            MetalImageDestination::from_exclusive_buffer(buffer.clone(), destination_layout)
                .expect("classic RGB destination")
        };
        decoder
            .submit_prepared_group_into(group, destination)
            .unwrap_or_else(|error| {
                panic!("submit {profile:?} {layout:?} classic RGB group: {error}")
            })
            .wait()
            .unwrap_or_else(|error| {
                panic!("complete {profile:?} {layout:?} classic RGB group: {error}")
            });

        // SAFETY: codec completion released exclusive output access.
        let actual = unsafe {
            j2k_metal_support::checked_buffer_read_vec::<u8>(&buffer, 4, output_bytes)
                .expect("classic RGB output samples")
        };
        assert_native_samples(&actual, expected_group.samples(), max_lsb_diff);
    }
}

#[test]
fn prepared_classic_rgb_matches_cpu_for_native_types_requests_and_layouts() {
    if !should_run_metal_runtime() {
        return;
    }

    let roi = Rect {
        x: 1,
        y: 2,
        w: 5,
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
    for profile in [
        ClassicRgbProfile::U8,
        ClassicRgbProfile::U8Irreversible97,
        ClassicRgbProfile::U12,
        ClassicRgbProfile::I12,
    ] {
        let encoded = Arc::<[u8]>::from(classic_rgb_fixture(profile));
        for layout in [BatchLayout::Nchw, BatchLayout::Nhwc] {
            assert_classic_rgb_layout(profile, &encoded, layout, &requests);
        }
    }
}
