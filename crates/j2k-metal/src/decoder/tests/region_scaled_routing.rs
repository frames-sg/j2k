// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{wrap_j2k_codestream, J2kFileWrapOptions};
use j2k_core::{BackendRequest, Downscale, PixelFormat, Rect};

use super::{
    hybrid, should_run_metal_runtime, J2kDecoder, MetalBackendSession, MetalDecodeRequest,
};

#[test]
fn jph_region_and_scaled_requests_use_prepared_direct_plans() {
    if !should_run_metal_runtime() {
        return;
    }
    let Ok(device) = j2k_metal_support::system_default_device() else {
        j2k_test_support::metal_device_unavailable_is_skip(module_path!());
        return;
    };

    let pixels = j2k_test_support::gradient_u8(64, 64, 3);
    let codestream = j2k_native::encode_htj2k(
        &pixels,
        64,
        64,
        3,
        8,
        false,
        &j2k_native::EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            ..j2k_native::EncodeOptions::default()
        },
    )
    .expect("encode HTJ2K routing fixture");
    let jph = wrap_j2k_codestream(&codestream, J2kFileWrapOptions::jph())
        .expect("wrap HTJ2K routing fixture as JPH");
    let roi = Rect {
        x: 8,
        y: 8,
        w: 32,
        h: 32,
    };
    let requests = [
        (
            MetalDecodeRequest::region(PixelFormat::Rgb8, roi, BackendRequest::Cpu),
            MetalDecodeRequest::region(PixelFormat::Rgb8, roi, BackendRequest::Metal),
        ),
        (
            MetalDecodeRequest::scaled(PixelFormat::Rgb8, Downscale::Half, BackendRequest::Cpu),
            MetalDecodeRequest::scaled(PixelFormat::Rgb8, Downscale::Half, BackendRequest::Metal),
        ),
    ];
    let _counter_guard = hybrid::region_scaled_color_plan_test_lock_for_test();
    let session = MetalBackendSession::new(device);
    let mut plan_builds = Vec::new();

    for (cpu_request, metal_request) in requests {
        hybrid::reset_region_scaled_color_plan_builds_for_test();
        let mut cpu_decoder = J2kDecoder::new(&jph).expect("CPU decoder");
        let cpu = cpu_decoder
            .decode_request_to_device_with_session(cpu_request, &session)
            .expect("CPU reference decode");
        let mut metal_decoder = J2kDecoder::new(&jph).expect("Metal decoder");
        let metal = metal_decoder
            .decode_request_to_device_with_session(metal_request, &session)
            .expect("Metal decode");

        assert_eq!(
            metal.as_bytes().expect("Metal output"),
            cpu.as_bytes().expect("CPU output")
        );
        plan_builds.push(hybrid::region_scaled_color_plan_builds_for_test());
    }

    assert_eq!(plan_builds, [1, 1]);
}
