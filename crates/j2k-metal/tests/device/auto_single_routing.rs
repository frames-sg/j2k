// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

fn auto_selected_backend(codestream: &[u8], request: MetalDecodeRequest) -> (BackendKind, Vec<u8>) {
    let mut decoder = J2kDecoder::new(codestream).expect("Auto decoder");
    let routed = decoder
        .decode_request_to_device_with_report(request)
        .expect("Auto decode");
    let bytes = routed.surface.as_bytes().expect("Auto bytes").into_owned();
    (routed.report.selected_backend, bytes)
}

fn cpu_bytes(codestream: &[u8], request: MetalDecodeRequest) -> Vec<u8> {
    J2kDecoder::new(codestream)
        .expect("CPU decoder")
        .decode_request_to_device(MetalDecodeRequest {
            backend: BackendRequest::Cpu,
            ..request
        })
        .expect("CPU decode")
        .as_bytes()
        .expect("CPU bytes")
        .into_owned()
}

#[test]
fn auto_promotes_ht_full_decodes_and_keeps_half_scale_on_cpu() {
    if !should_run_metal_runtime() {
        return;
    }

    let (width, height) = (640, 480);
    let pixels = j2k_test_support::gradient_u8(width, height, 3);
    let encode = |reversible| {
        encode_htj2k(
            &pixels,
            width,
            height,
            3,
            8,
            false,
            &EncodeOptions {
                reversible,
                num_decomposition_levels: 6,
                ..EncodeOptions::default()
            },
        )
        .expect("encode HTJ2K routing fixture")
    };
    let full = MetalDecodeRequest::full(PixelFormat::Rgb8, BackendRequest::Auto);
    let half = MetalDecodeRequest::scaled(PixelFormat::Rgb8, Downscale::Half, BackendRequest::Auto);

    // Lossless full decodes at the measured 640x480 threshold run on Metal
    // and match the CPU byte for byte.
    let lossless = encode(true);
    let (backend, actual) = auto_selected_backend(&lossless, full);
    assert_eq!(backend, BackendKind::Metal);
    assert_eq!(actual, cpu_bytes(&lossless, full));

    // Lossy (9/7) full decodes qualify at the same measured threshold and are
    // byte-identical; no half-scale cell qualified, so that stays on the CPU.
    let lossy = encode(false);
    for (request, expected) in [(full, BackendKind::Metal), (half, BackendKind::Cpu)] {
        let (backend, actual) = auto_selected_backend(&lossy, request);
        assert_eq!(backend, expected, "{request:?}");
        assert_eq!(actual, cpu_bytes(&lossy, request), "{request:?}");
    }
}
