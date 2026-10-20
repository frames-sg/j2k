// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{BackendRequest, PixelFormat};
use j2k_native::{encode_htj2k, EncodeOptions};

use super::runtime::should_run_metal_runtime;
use crate::engine::test_counters::{fused_idwt_stores_for_test, reset_fused_idwt_stores_for_test};
use crate::{J2kDecoder, MetalDecodeRequest};

/// A `width` x `height` HTJ2K image of `bit_depth`-bit samples, signed or
/// not, with structure and noise at every scale.
fn image(width: u32, height: u32, bit_depth: u8, signed: bool, reversible: bool) -> Vec<u8> {
    seeded_image(width, height, bit_depth, signed, reversible, 0)
}

fn seeded_image(
    width: u32,
    height: u32,
    bit_depth: u8,
    signed: bool,
    reversible: bool,
    seed: u32,
) -> Vec<u8> {
    let range = 1_i32 << bit_depth;
    let mut state = 0x2545_F491_u32 ^ seed.wrapping_mul(0x9E37_79B9);
    let mut samples = Vec::with_capacity(width as usize * height as usize * 2);
    for y in 0..height {
        for x in 0..width {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let smooth = i32::try_from((x * 13 + y * 7 + seed * 31) % 251).expect("fits i32");
            let noise = i32::try_from(state % 17).expect("fits i32");
            let value = (smooth * range / 251 + noise).rem_euclid(range);
            let value = if signed { value - range / 2 } else { value };
            if bit_depth <= 8 {
                samples.push(value.to_le_bytes()[0]);
            } else {
                samples.extend_from_slice(&value.to_le_bytes()[..2]);
            }
        }
    }
    let options = EncodeOptions {
        reversible,
        num_decomposition_levels: 3,
        ..EncodeOptions::default()
    };
    encode_htj2k(&samples, width, height, 1, bit_depth, signed, &options).expect("encode")
}

#[test]
fn fused_last_level_stores_match_the_cpu_for_every_gray_output() {
    if !should_run_metal_runtime() {
        return;
    }
    let (width, height) = (157_u32, 93_u32);
    for reversible in [true, false] {
        for (bit_depth, signed, fmt) in [
            (7, false, PixelFormat::Gray8),
            (7, false, PixelFormat::Gray16),
            (7, true, PixelFormat::Gray16),
            (8, false, PixelFormat::Gray8),
            (9, false, PixelFormat::Gray8),
            (9, false, PixelFormat::Gray16),
            (9, true, PixelFormat::Gray16),
            (16, false, PixelFormat::Gray8),
            (16, false, PixelFormat::Gray16),
            (16, true, PixelFormat::Gray16),
            (8, false, PixelFormat::Gray16),
            (12, false, PixelFormat::Gray16),
            (12, true, PixelFormat::Gray16),
            (12, true, PixelFormat::Gray8),
            (8, true, PixelFormat::Gray16),
        ] {
            let label = format!("{bit_depth}-bit signed={signed} {fmt:?} reversible={reversible}");
            let bytes = image(width, height, bit_depth, signed, reversible);
            let row_bytes = width as usize * fmt.bytes_per_pixel();
            let mut expected = vec![0; row_bytes * height as usize];
            j2k::J2kDecoder::new(&bytes)
                .expect("CPU decoder")
                .decode_into(&mut expected, row_bytes, fmt)
                .expect("CPU decode");

            reset_fused_idwt_stores_for_test();
            let surface = J2kDecoder::new(&bytes)
                .expect("decoder")
                .decode_request_to_device(MetalDecodeRequest::full(fmt, BackendRequest::Metal))
                .unwrap_or_else(|error| panic!("{label}: {error}"));
            assert_eq!(
                fused_idwt_stores_for_test(),
                1,
                "{label}: fused store not used"
            );
            let actual = surface.as_bytes().expect("surface bytes");
            assert_eq!(actual.len(), expected.len(), "{label}");
            if let Some(index) = actual.iter().zip(&expected).position(|(a, b)| a != b) {
                panic!(
                    "{label}: byte {index}: GPU={}, CPU={}",
                    actual[index], expected[index]
                );
            }
        }
    }
}

fn cpu_samples_bytes(samples: &j2k::CpuBatchSamples) -> Vec<u8> {
    match samples {
        j2k::CpuBatchSamples::U8(values) => values.clone(),
        j2k::CpuBatchSamples::U16(values) => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        j2k::CpuBatchSamples::I16(values) => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        _ => panic!("unexpected CPU batch sample type"),
    }
}

#[test]
fn stacked_batches_store_their_last_level_directly_and_match_the_cpu() {
    if !should_run_metal_runtime() {
        return;
    }
    let options = j2k::BatchDecodeOptions {
        layout: j2k::BatchLayout::Nhwc,
        ..j2k::BatchDecodeOptions::default()
    };
    let mut metal = crate::MetalBatchDecoder::system_default_with_options(options)
        .expect("Metal batch decoder");
    let mut cpu = j2k::CpuBatchDecoder::new(options);
    for reversible in [true, false] {
        for (bit_depth, signed) in [
            (7, false),
            (7, true),
            (8, false),
            (9, false),
            (9, true),
            (12, false),
            (12, true),
        ] {
            let label = format!("{bit_depth}-bit signed={signed} reversible={reversible}");
            // Distinct images of one geometry form one stacked group.
            let inputs = || {
                (0..4)
                    .map(|seed| j2k::EncodedImage {
                        bytes: std::sync::Arc::from(seeded_image(
                            131, 77, bit_depth, signed, reversible, seed,
                        )),
                        request: j2k::DecodeRequest::Full,
                    })
                    .collect::<Vec<_>>()
            };
            let expected: Vec<u8> = {
                let prepared = cpu.prepare(inputs()).expect("CPU prepare");
                let result = cpu.decode_prepared(&prepared).expect("CPU decode");
                assert!(result.errors().is_empty(), "{label}: CPU errors");
                result
                    .groups()
                    .iter()
                    .flat_map(|group| cpu_samples_bytes(group.samples()))
                    .collect()
            };
            let prepared = metal.prepare(inputs()).expect("Metal prepare");
            reset_fused_idwt_stores_for_test();
            let result = metal.decode_prepared(&prepared).expect("Metal decode");
            assert!(
                result.errors().is_empty() && result.group_errors().is_empty(),
                "{label}: Metal errors"
            );
            let mut actual = Vec::new();
            for group in result.groups() {
                for surface in group.surfaces() {
                    actual.extend_from_slice(&surface.as_bytes().expect("surface bytes"));
                }
            }
            assert_eq!(actual, expected, "{label}");
            assert!(
                fused_idwt_stores_for_test() > 0,
                "{label}: fused store not used"
            );
        }
    }
}

/// Restores allocation-sized stacked runs even if a test panics.
struct DefaultStackedRuns;

impl Drop for DefaultStackedRuns {
    fn drop(&mut self) {
        crate::engine::test_counters::set_stacked_run_len_for_test(0);
    }
}

#[test]
fn stacked_groups_split_into_runs_keep_every_image_in_place() {
    if !should_run_metal_runtime() {
        return;
    }
    let _restore = DefaultStackedRuns;
    let options = j2k::BatchDecodeOptions {
        layout: j2k::BatchLayout::Nhwc,
        ..j2k::BatchDecodeOptions::default()
    };
    let mut metal = crate::MetalBatchDecoder::system_default_with_options(options)
        .expect("Metal batch decoder");
    let mut cpu = j2k::CpuBatchDecoder::new(options);
    // Five images in runs of two leave a last image decoded on its own.
    let inputs = || {
        (0..5)
            .map(|seed| j2k::EncodedImage {
                bytes: std::sync::Arc::from(seeded_image(96, 64, 12, false, true, seed)),
                request: j2k::DecodeRequest::Full,
            })
            .collect::<Vec<_>>()
    };
    let expected: Vec<u8> = {
        let prepared = cpu.prepare(inputs()).expect("CPU prepare");
        let result = cpu.decode_prepared(&prepared).expect("CPU decode");
        result
            .groups()
            .iter()
            .flat_map(|group| cpu_samples_bytes(group.samples()))
            .collect()
    };
    crate::engine::test_counters::set_stacked_run_len_for_test(2);
    let prepared = metal.prepare(inputs()).expect("Metal prepare");
    let result = metal.decode_prepared(&prepared).expect("Metal decode");
    assert!(result.errors().is_empty() && result.group_errors().is_empty());
    let mut actual = Vec::new();
    for group in result.groups() {
        for surface in group.surfaces() {
            actual.extend_from_slice(&surface.as_bytes().expect("surface bytes"));
        }
    }
    assert_eq!(actual, expected);
}

#[test]
fn separate_signed_seven_bit_store_uses_requested_gray16_widening() {
    if !should_run_metal_runtime() {
        return;
    }
    let source = [-64_i8, -1, 0, 63];
    let pixels: Vec<_> = source
        .into_iter()
        .map(|value| value.to_le_bytes()[0])
        .collect();
    let bytes = encode_htj2k(
        &pixels,
        4,
        1,
        1,
        7,
        true,
        &EncodeOptions {
            num_decomposition_levels: 0,
            ..Default::default()
        },
    )
    .unwrap();
    let header = j2k_native::inspect_j2k_codestream_header(&bytes).unwrap();
    assert_eq!(header.resolution_levels, 1);
    assert_eq!(header.component_info[0].bit_depth, 7);
    assert!(header.component_info[0].signed);
    reset_fused_idwt_stores_for_test();
    let surface = J2kDecoder::new(&bytes)
        .unwrap()
        .decode_request_to_device(MetalDecodeRequest::full(
            PixelFormat::Gray16,
            BackendRequest::Metal,
        ))
        .unwrap();
    assert_eq!(fused_idwt_stores_for_test(), 0);
    let expected: Vec<_> = [65535_u16, 65535, 0, 32509]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    assert_eq!(surface.as_bytes().unwrap().as_ref(), expected);
}
