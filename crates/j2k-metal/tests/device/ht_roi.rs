// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use j2k_native::{encode_with_roi_regions, EncodeRoiRegion};

#[test]
fn reversible_htj2k_roi_decode_uses_metal_and_matches_cpu_exactly() {
    if !should_run_metal_runtime() {
        return;
    }

    let width = 64_u32;
    let height = 64_u32;
    let pixels = j2k_test_support::gradient_u8(width, height, 1);
    let bytes = encode_with_roi_regions(
        &pixels,
        width,
        height,
        1,
        8,
        false,
        &EncodeOptions {
            reversible: true,
            use_ht_block_coding: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        },
        &[EncodeRoiRegion {
            component: 0,
            x: 8,
            y: 12,
            width: 40,
            height: 36,
            shift: 12,
        }],
    )
    .expect("encode reversible HTJ2K ROI fixture");

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = vec![0_u8; (width * height) as usize];
    host_decoder
        .decode_into(&mut host, width as usize, PixelFormat::Gray8)
        .expect("host decode");

    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options).expect("Metal decoder");
    let batch = decoder
        .decode_batch(vec![EncodedImage::full(Arc::from(bytes))])
        .expect("Metal decode");

    assert!(batch.errors().is_empty(), "{:?}", batch.errors());
    assert!(
        batch.group_errors().is_empty(),
        "{:?}",
        batch.group_errors()
    );
    let group = &batch.groups()[0];
    assert!(
        group.dispatch_report().tier1 > 0,
        "ROI fixture must execute Metal Tier-1"
    );
    assert_eq!(
        group.surfaces()[0]
            .as_bytes()
            .expect("decoded surface bytes"),
        host
    );
}

#[test]
fn single_image_htj2k_roi_with_rgn_matches_cpu_on_metal() {
    if !should_run_metal_runtime() {
        return;
    }

    let bytes = fixture_ht_roi_rgb();
    let roi = Rect {
        x: 0,
        y: 0,
        w: 1,
        h: 2,
    };
    let session = MetalBackendSession::system_default().expect("Metal session");
    let mut cpu_decoder = J2kDecoder::new(&bytes).expect("CPU decoder");
    let cpu = cpu_decoder
        .decode_request_to_device_with_session(
            MetalDecodeRequest::region(PixelFormat::Rgb8, roi, BackendRequest::Cpu),
            &session,
        )
        .expect("CPU ROI decode");
    let mut metal_decoder = J2kDecoder::new(&bytes).expect("Metal decoder");
    let metal = metal_decoder
        .decode_request_to_device_with_session(
            MetalDecodeRequest::region(PixelFormat::Rgb8, roi, BackendRequest::Metal),
            &session,
        )
        .expect("generic Metal ROI fallback");

    assert_eq!(metal.backend_kind(), BackendKind::Metal);
    assert_eq!(
        metal.as_bytes().expect("Metal ROI output"),
        cpu.as_bytes().expect("CPU ROI output")
    );
}
