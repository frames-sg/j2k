// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k_core::{Downscale, PixelFormat, Rect};
#[cfg(target_os = "macos")]
use metal::Device;

use super::{
    decode_region_scaled_color_batch_direct_to_device,
    decode_region_scaled_color_batch_direct_to_device_routed,
    decode_region_scaled_direct_to_surface_with_session,
    decode_region_scaled_grayscale_batch_direct_to_device_routed,
    decode_repeated_region_scaled_color_batch_direct_to_device,
    decode_repeated_region_scaled_color_batch_direct_to_device_routed,
    region_scaled_color_plan_builds_for_test, region_scaled_color_plan_test_lock_for_test,
    reset_region_scaled_color_plan_builds_for_test, reset_region_scaled_color_plan_cache_for_test,
};
use crate::{Error, MetalBackendSession};

#[cfg(target_os = "macos")]
fn should_run_metal_runtime() -> bool {
    j2k_test_support::metal_runtime_gate(module_path!())
}

fn encoded_rgb8_tile_for_region_scaled_plan_cache(seed: u8) -> Arc<[u8]> {
    let mut pixels = j2k_test_support::gradient_u8(64, 64, 3);
    for pixel in pixels.chunks_exact_mut(3) {
        pixel[0] = pixel[0].wrapping_add(seed);
        pixel[1] = pixel[1].wrapping_add(seed.wrapping_mul(3));
        pixel[2] = pixel[2].wrapping_add(seed.wrapping_mul(5));
    }
    let options = j2k_native::EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        ..j2k_native::EncodeOptions::default()
    };
    Arc::<[u8]>::from(
        j2k_native::encode(&pixels, 64, 64, 3, 8, false, &options).expect("encode rgb8"),
    )
}

fn region_scaled_plan_cache_roi() -> Rect {
    Rect {
        x: 8,
        y: 8,
        w: 32,
        h: 32,
    }
}

#[cfg(target_os = "macos")]
fn session_runtime_fixture() -> Option<(Device, MetalBackendSession, usize, Rect)> {
    if !should_run_metal_runtime() {
        return None;
    }
    let Some(device) = Device::system_default() else {
        j2k_test_support::metal_device_unavailable_is_skip(module_path!());
        return None;
    };
    let session = MetalBackendSession::new(device.clone());
    let runtime = session.runtime().expect("explicit session runtime");
    Some((
        device,
        session,
        Arc::as_ptr(&runtime).addr(),
        region_scaled_plan_cache_roi(),
    ))
}

#[cfg(target_os = "macos")]
fn assert_plan_build_uses_runtime(
    device: &Device,
    expected_runtime: usize,
    label: &str,
    decode: impl FnOnce() -> Result<(), Error>,
) {
    crate::compute::reset_direct_tier1_input_buffer_prepares_for_test();
    crate::compute::with_isolated_runtime_for_device_for_test(device, decode).expect(label);
    assert_eq!(
        crate::compute::direct_tier1_input_buffer_runtime_for_test(),
        expected_runtime,
        "{label} must prepare with the explicit session runtime"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn explicit_session_gray_region_scaled_plans_use_session_runtime() {
    let Some((device, session, expected_runtime, roi)) = session_runtime_fixture() else {
        return;
    };
    let gray = Arc::<[u8]>::from(
        j2k_native::encode(
            &j2k_test_support::gradient_u8(64, 64, 1),
            64,
            64,
            1,
            8,
            false,
            &j2k_native::EncodeOptions {
                reversible: true,
                num_decomposition_levels: 2,
                ..j2k_native::EncodeOptions::default()
            },
        )
        .expect("encode gray8 region-scaled session fixture"),
    );

    assert_plan_build_uses_runtime(&device, expected_runtime, "single grayscale plan", || {
        decode_region_scaled_direct_to_surface_with_session(
            gray.as_ref(),
            PixelFormat::Gray8,
            roi,
            Downscale::Half,
            &session,
        )
        .map(drop)
    });
    assert_plan_build_uses_runtime(&device, expected_runtime, "grayscale batch plan", || {
        decode_region_scaled_grayscale_batch_direct_to_device_routed(
            &[(gray, roi, Downscale::Half)],
            PixelFormat::Gray8,
            Some(&session),
        )
        .map(drop)
    });
}

#[cfg(target_os = "macos")]
#[test]
fn explicit_session_color_region_scaled_plans_use_session_runtime() {
    let Some((device, session, expected_runtime, roi)) = session_runtime_fixture() else {
        return;
    };
    let color = encoded_rgb8_tile_for_region_scaled_plan_cache(71);
    assert_plan_build_uses_runtime(&device, expected_runtime, "single color plan", || {
        decode_region_scaled_direct_to_surface_with_session(
            color.as_ref(),
            PixelFormat::Rgb8,
            roi,
            Downscale::Half,
            &session,
        )
        .map(drop)
    });
}

#[cfg(target_os = "macos")]
#[test]
fn explicit_session_color_batches_use_session_runtime() {
    let Some((device, session, expected_runtime, roi)) = session_runtime_fixture() else {
        return;
    };
    let first = encoded_rgb8_tile_for_region_scaled_plan_cache(83);
    let second = encoded_rgb8_tile_for_region_scaled_plan_cache(89);
    let distinct = [
        (first, roi, Downscale::Half),
        (second, roi, Downscale::Half),
    ];
    assert_plan_build_uses_runtime(&device, expected_runtime, "distinct color batch", || {
        decode_region_scaled_color_batch_direct_to_device_routed(
            &distinct,
            PixelFormat::Rgb8,
            Some(&session),
        )
        .map(drop)
    });

    let repeated_input = encoded_rgb8_tile_for_region_scaled_plan_cache(101);
    let repeated = [
        (repeated_input.clone(), roi, Downscale::Half),
        (repeated_input, roi, Downscale::Half),
    ];
    assert_plan_build_uses_runtime(&device, expected_runtime, "repeated color batch", || {
        decode_region_scaled_color_batch_direct_to_device_routed(
            &repeated,
            PixelFormat::Rgb8,
            Some(&session),
        )
        .map(drop)
    });
}

#[cfg(target_os = "macos")]
#[test]
fn explicit_session_repeated_color_plan_uses_session_runtime() {
    let Some((device, session, expected_runtime, roi)) = session_runtime_fixture() else {
        return;
    };
    let color = encoded_rgb8_tile_for_region_scaled_plan_cache(97);
    assert_plan_build_uses_runtime(
        &device,
        expected_runtime,
        "explicit repeated color plan",
        || {
            decode_repeated_region_scaled_color_batch_direct_to_device_routed(
                color.as_ref(),
                roi,
                Downscale::Half,
                PixelFormat::Rgb8,
                2,
                Some(&session),
            )
            .map(drop)
        },
    );
}

#[test]
fn known_repeated_region_scaled_color_batch_builds_one_plan() {
    let _guard = region_scaled_color_plan_test_lock_for_test();
    reset_region_scaled_color_plan_builds_for_test();
    let input = Arc::<[u8]>::from([1_u8, 2, 3, 4]);
    let roi = Rect {
        x: 0,
        y: 0,
        w: 64,
        h: 64,
    };

    let result = decode_repeated_region_scaled_color_batch_direct_to_device(
        input.as_ref(),
        roi,
        Downscale::Half,
        PixelFormat::Rgb8,
        4,
    );

    assert!(result.is_err());
    assert_eq!(region_scaled_color_plan_builds_for_test(), 1);
}

#[test]
fn known_repeated_region_scaled_color_batch_rejects_zero_count() {
    let _guard = region_scaled_color_plan_test_lock_for_test();
    reset_region_scaled_color_plan_builds_for_test();
    let result = decode_repeated_region_scaled_color_batch_direct_to_device(
        &[1_u8, 2, 3, 4],
        Rect {
            x: 0,
            y: 0,
            w: 64,
            h: 64,
        },
        Downscale::Half,
        PixelFormat::Rgb8,
        0,
    );

    assert!(matches!(result, Err(Error::MetalKernel { .. })));
    assert_eq!(region_scaled_color_plan_builds_for_test(), 0);
}

#[cfg(target_os = "macos")]
#[test]
fn known_repeated_region_scaled_color_batch_reuses_cached_plan_across_calls() {
    if !should_run_metal_runtime() {
        return;
    }

    if Device::system_default().is_none() {
        j2k_test_support::metal_device_unavailable_is_skip(module_path!());
        return;
    }

    let _guard = region_scaled_color_plan_test_lock_for_test();
    let input = encoded_rgb8_tile_for_region_scaled_plan_cache(17);
    let roi = region_scaled_plan_cache_roi();
    reset_region_scaled_color_plan_cache_for_test();
    reset_region_scaled_color_plan_builds_for_test();

    for _ in 0..2 {
        let surfaces = decode_repeated_region_scaled_color_batch_direct_to_device(
            input.as_ref(),
            roi,
            Downscale::Quarter,
            PixelFormat::Rgb8,
            4,
        )
        .expect("repeated RGB region-scaled batch");
        assert_eq!(surfaces.len(), 4);
    }

    assert_eq!(
        region_scaled_color_plan_builds_for_test(),
        1,
        "same RGB ROI+scaled batch should reuse the prepared direct color plan across calls"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn known_distinct_region_scaled_color_batch_reuses_cached_plans_across_calls() {
    if !should_run_metal_runtime() {
        return;
    }

    if Device::system_default().is_none() {
        j2k_test_support::metal_device_unavailable_is_skip(module_path!());
        return;
    }

    let _guard = region_scaled_color_plan_test_lock_for_test();
    let first = encoded_rgb8_tile_for_region_scaled_plan_cache(29);
    let second = encoded_rgb8_tile_for_region_scaled_plan_cache(43);
    let roi = region_scaled_plan_cache_roi();
    let requests = vec![
        (first, roi, Downscale::Quarter),
        (second, roi, Downscale::Quarter),
    ];
    reset_region_scaled_color_plan_cache_for_test();
    reset_region_scaled_color_plan_builds_for_test();

    for _ in 0..2 {
        let surfaces =
            decode_region_scaled_color_batch_direct_to_device(&requests, PixelFormat::Rgb8)
                .expect("distinct RGB region-scaled batch");
        assert_eq!(surfaces.len(), requests.len());
    }

    assert_eq!(
        region_scaled_color_plan_builds_for_test(),
        2,
        "same distinct RGB ROI+scaled inputs should reuse prepared direct color plans across calls"
    );
}
