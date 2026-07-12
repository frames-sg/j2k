// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn default_gpu_encode_memory_budget_uses_forty_percent_capped_at_ten_gib() {
    const GIB: usize = 1024 * 1024 * 1024;

    assert_eq!(
        super::super::default_gpu_encode_memory_budget_bytes_for_hw_mem(8 * GIB),
        8 * GIB * 40 / 100
    );
    assert_eq!(
        super::super::default_gpu_encode_memory_budget_bytes_for_hw_mem(16 * GIB),
        16 * GIB * 40 / 100
    );
    assert_eq!(
        super::super::default_gpu_encode_memory_budget_bytes_for_hw_mem(24 * GIB),
        24 * GIB * 40 / 100
    );
    assert_eq!(
        super::super::default_gpu_encode_memory_budget_bytes_for_hw_mem(64 * GIB),
        10 * GIB
    );
}

#[test]
fn gpu_encode_inflight_resolution_clamps_requested_tiles_by_memory_budget() {
    let stats = super::super::resolve_lossless_encode_config_for_test(
        100,
        1_000,
        super::super::MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: Some(32),
            gpu_encode_memory_budget_bytes: Some(4_500),
        },
    )
    .expect("resolved config");

    assert_eq!(stats.configured_inflight_tiles, Some(32));
    assert_eq!(stats.effective_inflight_tiles, 4);
    assert_eq!(stats.configured_memory_budget_bytes, Some(4_500));
    assert_eq!(stats.effective_memory_budget_bytes, 4_500);
    assert_eq!(stats.estimated_peak_bytes_per_tile, 1_000);
}

#[test]
fn gpu_encode_default_inflight_uses_large_wsi_batch_when_memory_allows() {
    let stats = super::super::resolve_lossless_encode_config_for_test(
        600,
        1_000,
        super::super::MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: None,
            gpu_encode_memory_budget_bytes: Some(1_000_000),
        },
    )
    .expect("resolved config");

    assert_eq!(stats.configured_inflight_tiles, None);
    assert_eq!(stats.effective_inflight_tiles, 512);
}

#[test]
fn resident_classic_encode_default_inflight_uses_profiled_cap() {
    let config = super::super::resident_lossless_encode_config_for_mode(
        super::super::MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: None,
            gpu_encode_memory_budget_bytes: Some(1_000_000),
        },
        true,
        16,
    );

    assert_eq!(config.gpu_encode_inflight_tiles, Some(16));
    assert_eq!(config.gpu_encode_memory_budget_bytes, Some(1_000_000));
}

#[test]
fn resident_classic_encode_default_inflight_uses_large_batch_cap() {
    let config = super::super::resident_lossless_encode_config_for_mode(
        super::super::MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: None,
            gpu_encode_memory_budget_bytes: Some(1_000_000),
        },
        true,
        64,
    );

    assert_eq!(config.gpu_encode_inflight_tiles, Some(64));
    assert_eq!(config.gpu_encode_memory_budget_bytes, Some(1_000_000));
}

#[test]
fn resident_classic_encode_default_inflight_uses_very_large_batch_cap() {
    let config = super::super::resident_lossless_encode_config_for_mode(
        super::super::MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: None,
            gpu_encode_memory_budget_bytes: Some(1_000_000),
        },
        true,
        128,
    );

    assert_eq!(config.gpu_encode_inflight_tiles, Some(128));
    assert_eq!(config.gpu_encode_memory_budget_bytes, Some(1_000_000));
}

#[test]
fn resident_htj2k_encode_medium_batch_default_inflight_uses_profiled_cap() {
    let config = super::super::resident_lossless_encode_config_for_mode(
        super::super::MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: None,
            gpu_encode_memory_budget_bytes: Some(1_000_000),
        },
        false,
        64,
    );

    assert_eq!(config.gpu_encode_inflight_tiles, Some(32));
    assert_eq!(config.gpu_encode_memory_budget_bytes, Some(1_000_000));
}

#[test]
fn resident_htj2k_encode_large_batch_default_inflight_uses_profiled_cap() {
    let config = super::super::resident_lossless_encode_config_for_mode(
        super::super::MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: None,
            gpu_encode_memory_budget_bytes: Some(1_000_000),
        },
        false,
        128,
    );

    assert_eq!(config.gpu_encode_inflight_tiles, Some(64));
    assert_eq!(config.gpu_encode_memory_budget_bytes, Some(1_000_000));
}

#[test]
fn gpu_encode_inflight_resolution_rejects_zero_overrides() {
    let err = super::super::resolve_lossless_encode_config_for_test(
        4,
        1_000,
        super::super::MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: Some(0),
            gpu_encode_memory_budget_bytes: Some(4_000),
        },
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("in-flight"),
        "unexpected error: {err}"
    );

    let err = super::super::resolve_lossless_encode_config_for_test(
        4,
        1_000,
        super::super::MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: Some(2),
            gpu_encode_memory_budget_bytes: Some(0),
        },
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("memory budget"),
        "unexpected error: {err}"
    );
}
#[cfg(target_os = "macos")]
#[test]
#[expect(
    clippy::too_many_lines,
    clippy::cast_sign_loss,
    reason = "end-to-end batch regression keeps fixture and ordered comparisons together"
)]
fn metal_ht_batch_encode_preserves_order_and_matches_inflight_one() {
    if !should_run_metal_runtime() {
        return;
    }

    let inputs = [
        (0..8 * 8)
            .map(|i| ((i * 11 + 3) & 0xFF) as u8)
            .collect::<Vec<_>>(),
        (0..8 * 8)
            .map(|i| ((i * 13 + 5) & 0xFF) as u8)
            .collect::<Vec<_>>(),
        (0..8 * 8)
            .map(|i| ((i * 17 + 7) & 0xFF) as u8)
            .collect::<Vec<_>>(),
        (0..8 * 8)
            .map(|i| ((i * 19 + 9) & 0xFF) as u8)
            .collect::<Vec<_>>(),
    ];
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffers = inputs
        .iter()
        .map(|bytes| {
            crate::benchmark_private_buffer_with_bytes(&session, bytes)
                .expect("private benchmark input buffer")
        })
        .collect::<Vec<_>>();
    let tiles = buffers
        .iter()
        .map(|buffer| super::super::MetalLosslessEncodeTile {
            buffer,
            byte_offset: 0,
            width: 8,
            height: 8,
            pitch_bytes: 8,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Gray8,
        })
        .collect::<Vec<_>>();
    let options = lossless_options! {
        backend: EncodeBackendPreference::RequireDevice,
        block_coding_mode: J2kBlockCodingMode::HighThroughput,
        validation: J2kEncodeValidation::External,
    };

    compute::reset_resident_codestream_command_buffer_waits_for_test();
    let serial = super::super::encode_lossless_from_padded_metal_buffers_to_metal_batch(
        &tiles,
        &options,
        &session,
        super::super::MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: Some(1),
            gpu_encode_memory_budget_bytes: Some(1024 * 1024 * 1024),
        },
    )
    .expect("serial Metal HTJ2K batch");
    assert_eq!(
        compute::resident_codestream_command_buffer_waits_for_test(),
        1,
        "multi-chunk HT batch should wait once before harvesting completed chunks"
    );

    let cpu_validated_options = lossless_options! {
        backend: EncodeBackendPreference::RequireDevice,
        block_coding_mode: J2kBlockCodingMode::HighThroughput,
        validation: J2kEncodeValidation::CpuRoundTrip,
    };
    compute::reset_resident_codestream_command_buffer_waits_for_test();
    let cpu_validated = super::super::encode_lossless_from_padded_metal_buffers_to_metal_batch(
        &tiles,
        &cpu_validated_options,
        &session,
        super::super::MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: Some(1),
            gpu_encode_memory_budget_bytes: Some(1024 * 1024 * 1024),
        },
    )
    .expect("CPU-validated Metal HTJ2K batch");
    assert_eq!(cpu_validated.outcomes.len(), inputs.len());
    assert_eq!(
        compute::resident_codestream_command_buffer_waits_for_test(),
        inputs.len(),
        "CPU roundtrip validation should keep per-chunk waits to preserve overlap"
    );

    let parallel = super::super::encode_lossless_from_padded_metal_buffers_to_metal_batch(
        &tiles,
        &options,
        &session,
        super::super::MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: Some(2),
            gpu_encode_memory_budget_bytes: Some(1024 * 1024 * 1024),
        },
    )
    .expect("parallel Metal HTJ2K batch");
    let repeated_parallel = super::super::encode_lossless_from_padded_metal_buffers_to_metal_batch(
        &tiles,
        &options,
        &session,
        super::super::MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: Some(2),
            gpu_encode_memory_budget_bytes: Some(1024 * 1024 * 1024),
        },
    )
    .expect("repeated parallel Metal HTJ2K batch");

    assert_eq!(serial.outcomes.len(), inputs.len());
    assert_eq!(parallel.outcomes.len(), inputs.len());
    assert_eq!(parallel.stats.effective_inflight_tiles, 2);
    assert!(parallel.stats.max_observed_inflight_tiles <= 2);
    assert!(parallel.stats.max_observed_inflight_tiles > 0);
    for (((serial_outcome, parallel_outcome), repeated_outcome), expected) in serial
        .outcomes
        .iter()
        .zip(parallel.outcomes.iter())
        .zip(repeated_parallel.outcomes.iter())
        .zip(inputs.iter())
    {
        let serial_bytes = serial_outcome
            .encoded
            .codestream_bytes()
            .expect("serial codestream");
        let parallel_bytes = parallel_outcome
            .encoded
            .codestream_bytes()
            .expect("parallel codestream");
        let repeated_bytes = repeated_outcome
            .encoded
            .codestream_bytes()
            .expect("repeated parallel codestream");
        assert_eq!(parallel_bytes, serial_bytes);
        assert_eq!(repeated_bytes, serial_bytes);

        let decoded = Image::new(&parallel_bytes, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_eq!(&decoded.data, expected);
    }
}
#[cfg(target_os = "macos")]
#[test]
#[expect(
    clippy::cast_sign_loss,
    reason = "bounded synthetic batch pixels are nonnegative"
)]
fn metal_parallel_batch_returns_indexed_injected_failure() {
    if !should_run_metal_runtime() {
        return;
    }

    let first: Vec<u8> = (0..8 * 8).map(|i| ((i * 3) & 0xFF) as u8).collect();
    let second: Vec<u8> = (0..8 * 8).map(|i| ((i * 5) & 0xFF) as u8).collect();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let first_buffer = crate::benchmark_private_buffer_with_bytes(&session, &first)
        .expect("private benchmark input buffer");
    let second_buffer = crate::benchmark_private_buffer_with_bytes(&session, &second)
        .expect("private benchmark input buffer");
    let tiles = [
        super::super::MetalLosslessEncodeTile {
            buffer: &first_buffer,
            byte_offset: 0,
            width: 8,
            height: 8,
            pitch_bytes: 8,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Gray8,
        },
        super::super::MetalLosslessEncodeTile {
            buffer: &second_buffer,
            byte_offset: 0,
            width: 8,
            height: 8,
            pitch_bytes: 8,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Gray8,
        },
    ];
    let options = lossless_options! {
        backend: EncodeBackendPreference::RequireDevice,
        block_coding_mode: J2kBlockCodingMode::HighThroughput,
        validation: J2kEncodeValidation::External,
    };

    super::super::set_test_resident_encode_failure_index(Some(1));
    let Err(err) = super::super::encode_lossless_from_padded_metal_buffers_to_metal_batch(
        &tiles,
        &options,
        &session,
        super::super::MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: Some(2),
            gpu_encode_memory_budget_bytes: Some(1024 * 1024 * 1024),
        },
    ) else {
        panic!("injected failure should fail the batch");
    };
    super::super::set_test_resident_encode_failure_index(None);

    assert!(matches!(
        err,
        crate::Error::MetalKernel { message }
            if message == "injected J2K Metal resident encode failure at tile 1"
    ));
}
