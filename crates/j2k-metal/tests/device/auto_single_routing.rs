// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn auto_promotes_only_the_qualified_scaled_htj2k_cell() {
    if !should_run_metal_runtime() {
        return;
    }

    let width = 640;
    let height = 480;
    let pixels = j2k_test_support::gradient_u8(width, height, 3);
    let codestream = encode_htj2k(
        &pixels,
        width,
        height,
        3,
        8,
        false,
        &EncodeOptions {
            reversible: false,
            num_decomposition_levels: 6,
            ..EncodeOptions::default()
        },
    )
    .expect("encode qualified HTJ2K routing fixture");

    let mut full = J2kDecoder::new(&codestream).expect("full decoder");
    let full = full
        .decode_request_to_device_with_report(MetalDecodeRequest::full(
            PixelFormat::Rgb8,
            BackendRequest::Auto,
        ))
        .expect("Auto full decode");
    assert_eq!(full.report.selected_backend, BackendKind::Cpu);

    let request =
        MetalDecodeRequest::scaled(PixelFormat::Rgb8, Downscale::Half, BackendRequest::Auto);
    let mut cpu = J2kDecoder::new(&codestream).expect("CPU decoder");
    let expected = cpu
        .decode_request_to_device(MetalDecodeRequest {
            backend: BackendRequest::Cpu,
            ..request
        })
        .expect("CPU scaled decode")
        .as_bytes()
        .expect("CPU bytes")
        .into_owned();
    let mut auto = J2kDecoder::new(&codestream).expect("Auto decoder");
    let actual = auto
        .decode_request_to_device_with_report(request)
        .expect("Auto scaled decode");

    assert_eq!(actual.report.selected_backend, BackendKind::Metal);
    assert_eq!(
        actual.surface.as_bytes().expect("Metal bytes").as_ref(),
        expected
    );
}
