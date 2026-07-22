// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    checked_metal_surface_len, decode_scaled_to_surface, j2k_pack_kernel_name_for,
    j2k_pack_scale_arrays, output_shape_for, reset_shared_buffer_pool_misses_for_test,
    runtime_initialization_error, shared_buffer_pool_misses_for_test, with_runtime_for_device,
    MetalRuntime, MetalSupportError,
};
use j2k_core::PixelFormat;
use j2k_native::{
    encode_htj2k, ColorSpace as NativeColorSpace, DecodeSettings, EncodeOptions, Image,
};
use metal::{foreign_types::ForeignType, Device};

pub(super) fn should_run_metal_runtime() -> bool {
    j2k_test_support::metal_runtime_gate(module_path!())
}

#[test]
fn rgb16_with_alpha_is_rejected() {
    if !should_run_metal_runtime() {
        return;
    }

    let runtime = MetalRuntime::new().expect("Metal runtime");
    let result = output_shape_for(
        &NativeColorSpace::RGB,
        true,
        4,
        PixelFormat::Rgb16,
        &runtime,
    );
    assert!(result.is_err(), "RGBA input must not silently map to Rgb16");
}

#[test]
fn runtime_initialization_error_classifies_null_queue_as_unavailable() {
    assert!(matches!(
        runtime_initialization_error(&MetalSupportError::CommandQueueUnavailable),
        crate::Error::MetalUnavailable
    ));
}

#[test]
fn checked_metal_surface_len_accepts_valid_surface() {
    assert_eq!(
        checked_metal_surface_len((13, 7), PixelFormat::Rgb8.bytes_per_pixel(), "test surface")
            .unwrap(),
        (39, 273)
    );
}

#[test]
fn checked_metal_surface_len_reports_overflow_as_metal_error() {
    let error = checked_metal_surface_len((u32::MAX, 1), usize::MAX, "test surface").unwrap_err();

    assert!(
        matches!(error, crate::Error::MetalKernel { message } if message.contains("surface row byte count"))
    );
}

#[test]
fn two_d_threads_per_group_clamps_empty_pipeline_limits() {
    let threads = j2k_metal_support::two_d_threads_per_group(0, 0);

    assert_eq!((threads.width, threads.height, threads.depth), (1, 1, 1));
}

#[test]
fn one_d_threads_per_group_clamps_empty_pipeline_width() {
    let threads = j2k_metal_support::one_d_threads_per_group(0);

    assert_eq!((threads.width, threads.height, threads.depth), (1, 1, 1));
}

#[test]
fn two_d_threads_per_group_preserves_simd_width_and_derives_height() {
    let threads = j2k_metal_support::two_d_threads_per_group(32, 1024);

    assert_eq!((threads.width, threads.height, threads.depth), (32, 32, 1));
}

#[test]
fn with_runtime_for_device_scopes_runtime_to_requested_device() {
    if !should_run_metal_runtime() {
        return;
    }

    let Some(device) = Device::system_default() else {
        j2k_test_support::metal_device_unavailable_is_skip(module_path!());
        return;
    };

    let runtime_device =
        with_runtime_for_device(&device, |runtime| Ok(runtime.device.as_ptr() as usize))
            .expect("Metal runtime");

    assert_eq!(runtime_device, device.as_ptr() as usize);
}

#[test]
fn runtime_reuses_recycled_shared_buffers() -> Result<(), crate::Error> {
    if !should_run_metal_runtime() {
        return Ok(());
    }

    let Some(device) = Device::system_default() else {
        j2k_test_support::metal_device_unavailable_is_skip(module_path!());
        return Ok(());
    };
    let runtime = MetalRuntime::new_with_device(&device).expect("Metal runtime");

    reset_shared_buffer_pool_misses_for_test();
    let first = runtime.take_shared_buffer(64)?;
    runtime.recycle_shared_buffer(first)?;
    let _second = runtime.take_shared_buffer(64)?;

    assert_eq!(
        shared_buffer_pool_misses_for_test(),
        1,
        "recycled shared metadata buffers should be reused instead of allocating again"
    );
    Ok(())
}

#[test]
fn j2k_pack_selects_specialized_kernels_for_wsi_formats() {
    assert_eq!(
        j2k_pack_kernel_name_for(&NativeColorSpace::Gray, false, 1, PixelFormat::Gray8),
        Some("j2k_pack_gray8")
    );
    assert_eq!(
        j2k_pack_kernel_name_for(&NativeColorSpace::RGB, false, 3, PixelFormat::Rgb8),
        Some("j2k_pack_rgb8")
    );
    assert_eq!(
        j2k_pack_kernel_name_for(&NativeColorSpace::RGB, false, 3, PixelFormat::Rgba8),
        Some("j2k_pack_rgb_opaque_rgba8")
    );
    assert_eq!(
        j2k_pack_kernel_name_for(&NativeColorSpace::RGB, true, 4, PixelFormat::Rgba8),
        Some("j2k_pack_rgba8")
    );
    assert_eq!(
        j2k_pack_kernel_name_for(&NativeColorSpace::Gray, false, 1, PixelFormat::Gray16),
        Some("j2k_pack_gray16")
    );
    assert_eq!(
        j2k_pack_kernel_name_for(&NativeColorSpace::RGB, false, 3, PixelFormat::Rgb16),
        Some("j2k_pack_rgb16")
    );
    assert_eq!(
        j2k_pack_kernel_name_for(&NativeColorSpace::RGB, true, 4, PixelFormat::Rgb16),
        None,
        "RGBA input must not silently drop alpha when packing RGB16"
    );
}

#[test]
fn j2k_pack_precomputes_scale_factors_on_cpu() {
    let (max_values, u8_scales, u16_scales) = j2k_pack_scale_arrays([8, 12, 16, 0]);

    assert_f32_near(max_values[0], 255.0);
    assert_f32_near(max_values[1], 4095.0);
    assert_f32_near(max_values[2], 65_535.0);
    assert_f32_near(max_values[3], 1.0);
    assert_f32_near(u8_scales[0], 1.0);
    assert_f32_near(u8_scales[1], 255.0 / 4095.0);
    assert_f32_near(u16_scales[0], 257.0);
    assert_f32_near(u16_scales[1], 1.0);
    assert_f32_near(u16_scales[2], 1.0);
    assert_f32_near(u16_scales[3], 65_535.0);
}

fn assert_f32_near(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() <= f32::EPSILON,
        "expected {actual} to be within f32 epsilon of {expected}"
    );
}

#[test]
fn scaled_htj2k_decode_runs_through_metal_compute_path() {
    if !should_run_metal_runtime() {
        return;
    }

    let pixels: Vec<u8> = (0..16).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let bytes = encode_htj2k(&pixels, 4, 4, 1, 8, false, &options).expect("encode ht gray8");

    let image = Image::new(
        &bytes,
        &DecodeSettings {
            target_resolution: Some((2, 2)),
            ..DecodeSettings::default()
        },
    )
    .expect("image");
    let host = image.decode().expect("host scaled decode");

    let surface = decode_scaled_to_surface(
        &bytes,
        (4, 4),
        PixelFormat::Gray8,
        j2k_core::Downscale::Half,
    )
    .expect("metal scaled decode");
    assert_eq!(
        surface.as_bytes().expect("surface byte access"),
        host.as_slice()
    );
}
