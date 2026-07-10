// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use std::sync::Arc;

use j2k_core::{
    BackendKind, BackendRequest, CodecError, DeviceSurface, Downscale, PixelFormat, Rect,
};
#[cfg(target_os = "macos")]
use metal::Device;

#[cfg(target_os = "macos")]
use super::is_direct_runtime_fallback_error;
use super::surface::upload_surface;
use super::{DecodeOperation, MetalDecodeRequest};
use crate::{batch, Error, Storage, Surface, SurfaceResidency};
#[cfg(target_os = "macos")]
use crate::{hybrid, MetalBackendSession, MetalDirectFallbackReason};

#[cfg(target_os = "macos")]
fn should_run_metal_runtime() -> bool {
    j2k_test_support::metal_runtime_gate(module_path!())
}

#[cfg(target_os = "macos")]
#[test]
fn direct_runtime_fallback_classification_uses_structured_variant() {
    let fallback = Error::MetalDirectFallback {
        message: "arbitrary fallback text".to_string(),
        reason: MetalDirectFallbackReason::UnsupportedRuntimeInput,
    };
    assert!(is_direct_runtime_fallback_error(&fallback));

    let message_only = Error::MetalKernel {
        message: "unsupported classic kernel input in direct component plan".to_string(),
    };
    assert!(!is_direct_runtime_fallback_error(&message_only));
}

#[test]
fn metal_decode_request_maps_geometry_to_report_and_batch_ops() {
    let roi = Rect {
        x: 1,
        y: 2,
        w: 3,
        h: 4,
    };
    let requests = [
        (
            MetalDecodeRequest::full(PixelFormat::Gray8, BackendRequest::Auto),
            DecodeOperation::Full,
            batch::BatchOp::Full,
        ),
        (
            MetalDecodeRequest::region(PixelFormat::Gray8, roi, BackendRequest::Auto),
            DecodeOperation::Region,
            batch::BatchOp::Region(roi),
        ),
        (
            MetalDecodeRequest::scaled(PixelFormat::Gray8, Downscale::Half, BackendRequest::Auto),
            DecodeOperation::Scaled,
            batch::BatchOp::Scaled(Downscale::Half),
        ),
        (
            MetalDecodeRequest::region_scaled(
                PixelFormat::Gray8,
                roi,
                Downscale::Quarter,
                BackendRequest::Auto,
            ),
            DecodeOperation::RegionScaled,
            batch::BatchOp::RegionScaled {
                roi,
                scale: Downscale::Quarter,
            },
        ),
    ];

    for (request, report_operation, batch_op) in requests {
        assert_eq!(request.op.report_operation(), report_operation);
        assert_eq!(request.op.batch_op(), batch_op);
    }
}

#[test]
fn metal_runtime_failures_are_not_unsupported_errors() {
    for err in [
        Error::MetalRuntime {
            message: "runtime".to_string(),
        },
        Error::MetalKernel {
            message: "kernel".to_string(),
        },
        Error::MetalStatePoisoned {
            state: "J2K Metal session",
        },
    ] {
        assert!(!err.is_unsupported(), "{err:?}");
    }
}

#[test]
fn cpu_uploaded_surface_reports_host_residency() {
    let surface = upload_surface(
        vec![1, 2, 3],
        (1, 1),
        PixelFormat::Rgb8,
        BackendRequest::Cpu,
    )
    .expect("create CPU surface");

    assert_eq!(surface.backend_kind(), BackendKind::Cpu);
    assert_eq!(surface.residency(), SurfaceResidency::Host);
    #[cfg(target_os = "macos")]
    assert!(surface.metal_buffer_trusted().is_none());
}

#[test]
fn download_into_reports_inconsistent_surface_storage_range() {
    let surface = Surface {
        backend: BackendKind::Cpu,
        residency: SurfaceResidency::Host,
        dimensions: (2, 1),
        fmt: PixelFormat::Gray8,
        pitch_bytes: 2,
        byte_offset: 0,
        storage: Storage::Host(vec![7]),
    };
    let mut out = [0_u8; 2];

    let err = surface
        .download_into(&mut out, 2)
        .expect_err("inconsistent surface storage should be reported");

    assert!(matches!(
        err,
        Error::MetalKernel { message }
            if message == "J2K Metal surface byte range 0..2 exceeds storage length 1"
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn metal_backend_sessions_own_distinct_direct_plan_caches() {
    if !should_run_metal_runtime() {
        return;
    }

    let Some(device) = metal::Device::system_default() else {
        j2k_test_support::metal_device_unavailable_is_skip(module_path!());
        return;
    };

    let first = MetalBackendSession::new(device.clone());
    let second = MetalBackendSession::new(device);

    assert_ne!(
        first.direct_cache_ids_for_test(),
        second.direct_cache_ids_for_test()
    );
}

#[cfg(target_os = "macos")]
#[test]
fn explicit_metal_request_does_not_stage_cpu_pixels() {
    if !should_run_metal_runtime() {
        return;
    }

    if Device::system_default().is_none() {
        j2k_test_support::metal_device_unavailable_is_skip(module_path!());
        return;
    }

    let result = upload_surface(
        vec![1, 2, 3],
        (1, 1),
        PixelFormat::Rgb8,
        BackendRequest::Metal,
    );

    assert!(matches!(
        result,
        Err(Error::UnsupportedMetalRequest { reason })
            if reason.contains("CPU-staged")
                && reason.contains("explicit")
                && reason.contains("Metal")
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn repeated_region_scaled_color_batch_reuses_prepared_plan() {
    if !should_run_metal_runtime() {
        return;
    }

    if Device::system_default().is_none() {
        j2k_test_support::metal_device_unavailable_is_skip(module_path!());
        return;
    }

    let pixels = j2k_test_support::gradient_u8(64, 64, 3);
    let options = j2k_native::EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        ..j2k_native::EncodeOptions::default()
    };
    let input = Arc::<[u8]>::from(
        j2k_native::encode(&pixels, 64, 64, 3, 8, false, &options).expect("encode rgb8"),
    );
    let roi = Rect {
        x: 8,
        y: 8,
        w: 32,
        h: 32,
    };
    let scale = Downscale::Quarter;
    let requests = vec![(input.clone(), roi, scale); 4];
    let _guard = hybrid::region_scaled_color_plan_test_lock_for_test();
    hybrid::reset_region_scaled_color_plan_builds_for_test();

    let surfaces =
        hybrid::decode_region_scaled_color_batch_direct_to_device(&requests, PixelFormat::Rgb8)
            .expect("repeated RGB region-scaled batch");

    assert_eq!(surfaces.len(), requests.len());
    assert_eq!(
        hybrid::region_scaled_color_plan_builds_for_test(),
        1,
        "repeated RGB ROI+scaled batches should build and crop one prepared direct color plan"
    );
}

#[test]
fn decoder_modules_remain_focused_without_suppression_shortcuts() {
    const MODULES: [(&str, &str, usize); 6] = [
        ("adapters.rs", include_str!("adapters.rs"), 300),
        ("core.rs", include_str!("core.rs"), 400),
        ("direct_paths.rs", include_str!("direct_paths.rs"), 600),
        ("request.rs", include_str!("request.rs"), 220),
        ("routes.rs", include_str!("routes.rs"), 380),
        ("surface.rs", include_str!("surface.rs"), 120),
    ];

    let root = include_str!("../decoder.rs");
    assert!(
        root.lines().count() <= 40,
        "decoder.rs should remain a small explicit module facade"
    );

    for (name, source, cap) in MODULES {
        assert!(
            source.lines().count() <= cap,
            "{name} has {} lines, exceeding its {cap}-line focus cap",
            source.lines().count()
        );
        assert!(
            !source.contains("include!("),
            "{name} must be a real module"
        );
        assert!(
            !source.contains("#![allow"),
            "{name} must not use module-wide lint suppression"
        );
        assert!(
            !source.contains("allow(unused"),
            "{name} must not suppress unused-code findings"
        );
        assert!(
            !source.contains("use super::*") && !source.contains("use crate::*"),
            "{name} must use explicit imports"
        );
    }
}
