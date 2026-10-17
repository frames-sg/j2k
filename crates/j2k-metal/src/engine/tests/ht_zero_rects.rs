// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{BackendRequest, PixelFormat};
use j2k_native::{encode_htj2k, DecodeSettings, DecoderContext, EncodeOptions, Image};

use super::super::prepare_direct_grayscale_plan;
use super::runtime::should_run_metal_runtime;
use crate::{J2kDecoder, MetalDecodeRequest};

/// A 16-bit image that is zero across a region spanning whole code blocks in
/// every sub-band, so those blocks code no passes and get no decode job.
fn image_with_empty_blocks() -> (Vec<u8>, u32) {
    let (width, height) = (512_u32, 384_u32);
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 2);
    for y in 0..height {
        for x in 0..width {
            let value = if (96..416).contains(&x) && (96..288).contains(&y) {
                0
            } else {
                u16::try_from((x * 37 + y * 101) % 4093 + 7).expect("fits u16")
            };
            pixels.extend_from_slice(&value.to_le_bytes());
        }
    }
    let options = EncodeOptions {
        num_decomposition_levels: 2,
        ..EncodeOptions::default()
    };
    let bytes = encode_htj2k(&pixels, width, height, 1, 16, false, &options).expect("encode");
    (bytes, width)
}

#[test]
fn ht_blocks_without_jobs_decode_to_zero_after_the_arena_was_used() {
    if !should_run_metal_runtime() {
        return;
    }
    let (bytes, width) = image_with_empty_blocks();
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse");
    let plan = image
        .build_direct_grayscale_plan_with_context(&mut DecoderContext::default())
        .expect("native plan");
    let prepared = prepare_direct_grayscale_plan(&plan).expect("prepared plan");
    assert!(
        prepared
            .ht_groups
            .iter()
            .any(|group| !group.zero_rects.is_empty()),
        "the fixture must leave code blocks without jobs"
    );

    let row_bytes = width as usize * PixelFormat::Gray16.bytes_per_pixel();
    let mut cpu = j2k::J2kDecoder::new(&bytes).expect("CPU decoder");
    let mut expected = vec![0; row_bytes * cpu.info().dimensions.1 as usize];
    cpu.decode_into(&mut expected, row_bytes, PixelFormat::Gray16)
        .expect("CPU decode");
    // Other images first fill the runtime's pooled coefficient scratch with
    // nonzero values, so a skipped zero fill would show.
    let busy = super::classic::encoded_classic(512, 384, 1);
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    for pass in 0..3 {
        let mut other = J2kDecoder::new(&busy).expect("other decoder");
        other
            .decode_request_to_device(MetalDecodeRequest::full(
                PixelFormat::Gray16,
                BackendRequest::Metal,
            ))
            .expect("other decode");
        let surface = decoder
            .decode_request_to_device(MetalDecodeRequest::full(
                PixelFormat::Gray16,
                BackendRequest::Metal,
            ))
            .expect("Metal decode");
        assert_eq!(
            surface.as_bytes().expect("surface bytes").as_ref(),
            expected.as_slice(),
            "pass {pass}"
        );
    }
}
