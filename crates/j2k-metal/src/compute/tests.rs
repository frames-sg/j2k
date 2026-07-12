// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    checked_metal_surface_len, classic_batch_uses_plain_fast_path,
    classic_repeated_uses_plain_fast_path, crop_prepared_direct_grayscale_plan_to_output_region,
    decode_prepared_classic_sub_band_on_cpu, decode_scaled_to_surface,
    direct_tier1_input_buffer_prepares_for_test,
    execute_flattened_hybrid_cpu_tier1_direct_color_plan_batch_for_test,
    execute_hybrid_cpu_tier1_direct_color_plan_batch, flattened_hybrid_cpu_decode_batches_for_test,
    hybrid_cpu_decode_inputs_for_test, hybrid_cpu_decode_worker_count,
    hybrid_cpu_decode_worker_inits_for_test, hybrid_repeated_output_blits_for_test,
    hybrid_stacked_component_batches_for_test, j2k_pack_kernel_name_for, j2k_pack_scale_arrays,
    output_shape_for, prepare_direct_color_plan, prepare_direct_color_plan_for_cpu_upload,
    prepare_direct_grayscale_plan, prepare_sub_band_groups,
    prepared_direct_color_tier1_input_count, prepared_direct_grayscale_plan_compute_encoder_count,
    prepared_idwt_output_len, prepared_repeated_direct_ht_cleanup_dispatch_count,
    repeated_gray_store_is_contiguous_full_surface,
    reset_direct_tier1_input_buffer_prepares_for_test,
    reset_flattened_hybrid_cpu_decode_batches_for_test, reset_hybrid_cpu_decode_inputs_for_test,
    reset_hybrid_cpu_decode_worker_inits_for_test, reset_hybrid_repeated_output_blits_for_test,
    reset_hybrid_stacked_component_batches_for_test, reset_shared_buffer_pool_misses_for_test,
    reset_thread_hybrid_cpu_decode_inputs_for_test, retain_ht_jobs_for_required_region,
    runtime_initialization_error, shared_buffer_pool_misses_for_test,
    should_flatten_hybrid_cpu_tier1_color_batch, supports_stacked_direct_component_plane_batch,
    thread_hybrid_cpu_decode_inputs_for_test, with_runtime_for_device, DirectTier1Mode,
    J2kClassicCleanupBatchJob, J2kClassicSegment, J2kHtCleanupBatchJob, J2kRepeatedGrayStoreParams,
    MetalRuntime, MetalSupportError, PreparedClassicSubBand, PreparedDirectColorPlan,
    PreparedDirectGrayscaleStep, PreparedHtSubBand,
};
use j2k_core::PixelFormat;
use j2k_native::{
    decode_j2k_sub_band_scalar, encode, encode_htj2k, ColorSpace as NativeColorSpace,
    DecodeSettings, DecoderContext, EncodeOptions, Image, J2kCodeBlockBatchJob,
    J2kCodeBlockDecodeJob, J2kDirectGrayscaleStep as NativeDirectGrayscaleStep,
    J2kOwnedCodeBlockBatchJob, J2kOwnedSubBandPlan, J2kSubBandDecodeJob, J2kWaveletTransform,
};
use metal::foreign_types::ForeignType;
use metal::Device;
use std::sync::{Arc, Mutex};

static HYBRID_COUNTER_TEST_LOCK: Mutex<()> = Mutex::new(());

fn should_run_metal_runtime() -> bool {
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
fn classic_encode_output_capacity_keeps_conservative_default() {
    let capacity =
        super::classic_encode_output_capacity(64, 64, 11).expect("classic output capacity");

    assert_eq!(capacity, 64 * 64 * 11 * 8 + 4097);
}

#[test]
fn classic_encode_segment_capacity_uses_coding_style_bound() {
    assert_eq!(super::classic_encode_segment_capacity(0, 16), 1);
    assert_eq!(
        super::classic_encode_segment_capacity(
            super::J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS,
            9,
        ),
        11
    );
    assert_eq!(
        super::classic_encode_segment_capacity(
            super::J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS,
            16,
        ),
        25
    );
    assert_eq!(
        super::classic_encode_segment_capacity(
            super::J2K_CLASSIC_STYLE_TERMINATION_ON_EACH_PASS,
            16,
        ),
        46
    );
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
fn classic_tier1_pass_class_counts_split_bypass_pass_types() {
    let counts = super::classic_tier1_pass_class_counts(
        23,
        super::J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS,
    );

    assert_eq!(counts.arithmetic, 14);
    assert_eq!(counts.raw, 9);
    assert_eq!(counts.cleanup, 8);
    assert_eq!(counts.sigprop, 8);
    assert_eq!(counts.magref, 7);
    assert_eq!(counts.arithmetic_cleanup, 8);
    assert_eq!(counts.arithmetic_sigprop, 3);
    assert_eq!(counts.arithmetic_magref, 3);
    assert_eq!(counts.raw_sigprop, 5);
    assert_eq!(counts.raw_magref, 4);
}

#[test]
fn classic_tier1_pass_class_counts_style0_stays_arithmetic() {
    let counts = super::classic_tier1_pass_class_counts(5, 0);

    assert_eq!(counts.arithmetic, 5);
    assert_eq!(counts.raw, 0);
    assert_eq!(counts.cleanup, 2);
    assert_eq!(counts.sigprop, 2);
    assert_eq!(counts.magref, 1);
    assert_eq!(counts.arithmetic_cleanup, 2);
    assert_eq!(counts.arithmetic_sigprop, 2);
    assert_eq!(counts.arithmetic_magref, 1);
    assert_eq!(counts.raw_sigprop, 0);
    assert_eq!(counts.raw_magref, 0);
}

#[test]
fn classic_tier1_scan_estimates_multiply_passes_by_block_area() {
    let pass_counts = super::classic_tier1_pass_class_counts(
        23,
        super::J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS,
    );
    let mut stats = super::J2kResidentEncodeStageStats::default();

    super::accumulate_classic_tier1_scan_estimates(&mut stats, pass_counts, 32 * 32);

    assert_eq!(stats.tier1_full_scan_coeff_visit_count_total, 23 * 1024);
    assert_eq!(
        stats.tier1_arithmetic_scan_coeff_visit_count_total,
        14 * 1024
    );
    assert_eq!(stats.tier1_raw_scan_coeff_visit_count_total, 9 * 1024);
    assert_eq!(stats.tier1_cleanup_scan_coeff_visit_count_total, 8 * 1024);
    assert_eq!(stats.tier1_sigprop_scan_coeff_visit_count_total, 8 * 1024);
    assert_eq!(stats.tier1_magref_scan_coeff_visit_count_total, 7 * 1024);
    assert_eq!(stats.max_tier1_full_scan_coeff_visits_per_block, 23 * 1024);
}

#[test]
fn classic_packet_output_capacity_uses_raw_sample_bound_when_smaller() {
    let codestream = super::J2kLosslessCodestreamAssemblyJob {
        width: 512,
        height: 512,
        component_count: 3,
        bit_depth: 8,
        signed: false,
        num_decomposition_levels: 3,
        use_mct: true,
        guard_bits: 2,
        code_block_width_exp: 4,
        code_block_height_exp: 4,
        progression_order: j2k_native::EncodeProgressionOrder::Lrcp,
        write_tlm: false,
        block_coding_mode: super::J2kLosslessCodestreamBlockCodingMode::Classic,
    };
    let header_capacity = 1024 * 256 + 4096;
    let conservative_capacity = 12 * 1024 * 1024;
    let packet_descriptor_count = 3;

    let capacity = super::classic_packet_output_capacity(
        conservative_capacity,
        header_capacity,
        packet_descriptor_count,
        codestream,
    )
    .expect("classic packet capacity");

    let raw_bytes = 512 * 512 * 3;
    let descriptor_slack = packet_descriptor_count * 256;
    assert_eq!(
        capacity,
        raw_bytes + header_capacity + descriptor_slack + 64 * 1024
    );

    let tiny_tier1_capacity = 4096;
    let clamped = super::classic_packet_output_capacity(
        tiny_tier1_capacity,
        header_capacity,
        packet_descriptor_count,
        codestream,
    )
    .expect("classic packet capacity");
    let conservative_packet_capacity =
        tiny_tier1_capacity + header_capacity * packet_descriptor_count + 1024;
    assert_eq!(clamped, conservative_packet_capacity);
}

#[test]
fn ht_encode_output_capacity_scales_with_code_block_area() {
    let max_block = super::ht_encode_output_capacity(128, 128).expect("max HT output capacity");
    assert_eq!(max_block, super::J2K_HT_ENCODE_BASE_OUTPUT_SIZE);

    let smaller_block =
        super::ht_encode_output_capacity(32, 32).expect("scaled HT output capacity");
    assert!(smaller_block < max_block / 2);
    assert!(smaller_block >= 8192);
}

#[test]
fn classic_encode_pipeline_kind_prefers_style0_32_for_resident_jobs() {
    let jobs = [super::J2kClassicEncodeBatchJob {
        width: 32,
        height: 32,
        style_flags: 0,
        ..super::J2kClassicEncodeBatchJob::default()
    }];

    assert_eq!(
        super::classic_encode_code_blocks_pipeline_kind(&jobs),
        super::J2kClassicEncodePipelineKind::Style0_32
    );
}

#[test]
fn classic_encode_pipeline_kind_prefers_bypass_32_for_resident_jobs() {
    let jobs = [super::J2kClassicEncodeBatchJob {
        width: 32,
        height: 32,
        style_flags: super::J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS,
        total_bitplanes: 31,
        ..super::J2kClassicEncodeBatchJob::default()
    }];

    assert_eq!(
        super::classic_encode_code_blocks_pipeline_kind(&jobs),
        super::J2kClassicEncodePipelineKind::Bypass32
    );
}

#[test]
fn classic_encode_pipeline_kind_prefers_bypass_u16_32_for_low_bitplane_resident_jobs() {
    let jobs = [super::J2kClassicEncodeBatchJob {
        width: 32,
        height: 32,
        style_flags: super::J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS,
        total_bitplanes: 16,
        ..super::J2kClassicEncodeBatchJob::default()
    }];

    assert_eq!(
        super::classic_encode_code_blocks_pipeline_kind(&jobs),
        super::J2kClassicEncodePipelineKind::BypassU16_32
    );
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

fn test_ht_job(output_x: u32, output_y: u32, width: u32, height: u32) -> J2kHtCleanupBatchJob {
    J2kHtCleanupBatchJob {
        coded_offset: output_y
            .checked_mul(32)
            .and_then(|base| base.checked_add(output_x))
            .expect("test coded offset"),
        width,
        height,
        coded_len: 1,
        cleanup_length: 1,
        refinement_length: 0,
        missing_msbs: 0,
        num_bitplanes: 8,
        roi_shift: 0,
        number_of_coding_passes: 1,
        output_stride: 8,
        output_offset: output_y
            .checked_mul(8)
            .and_then(|base| base.checked_add(output_x))
            .expect("test output offset"),
        dequantization_step: 1.0,
        stripe_causal: 0,
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "test helper receives small synthetic band identifiers"
)]
fn test_ht_sub_band(band_id: u32) -> PreparedHtSubBand {
    PreparedHtSubBand {
        band_id,
        width: 8,
        height: 8,
        coded_data: vec![band_id as u8],
        coded_buffer: None,
        jobs: vec![test_ht_job(0, 0, 2, 2)],
        jobs_buffer: None,
    }
}

fn separator_store_step() -> PreparedDirectGrayscaleStep {
    PreparedDirectGrayscaleStep::Store(j2k_native::J2kDirectStoreStep {
        input_band_id: 99,
        input_rect: j2k_native::J2kRect {
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 1,
        },
        source_x: 0,
        source_y: 0,
        copy_width: 1,
        copy_height: 1,
        output_width: 1,
        output_height: 1,
        output_x: 0,
        output_y: 0,
        addend: 0.0,
    })
}

#[test]
fn direct_sub_band_grouping_groups_adjacent_ht_runs_without_runtime() {
    let steps = vec![
        PreparedDirectGrayscaleStep::HtSubBand(test_ht_sub_band(1)),
        PreparedDirectGrayscaleStep::HtSubBand(test_ht_sub_band(2)),
        separator_store_step(),
        PreparedDirectGrayscaleStep::HtSubBand(test_ht_sub_band(3)),
        PreparedDirectGrayscaleStep::HtSubBand(test_ht_sub_band(4)),
    ];

    let groups = prepare_sub_band_groups(
        &steps,
        DirectTier1Mode::CpuUpload,
        |step| match step {
            PreparedDirectGrayscaleStep::HtSubBand(sub_band) => Some(sub_band),
            _ => None,
        },
        |start, end, sub_bands, _| {
            Ok((
                start,
                end,
                sub_bands
                    .iter()
                    .map(|sub_band| sub_band.band_id)
                    .collect::<Vec<_>>(),
            ))
        },
    )
    .expect("group adjacent HT sub-bands");

    assert_eq!(groups, vec![(0, 2, vec![1, 2]), (3, 5, vec![3, 4])]);
}

#[test]
fn direct_roi_prunes_ht_jobs_without_runtime() {
    let mut jobs = vec![
        test_ht_job(0, 0, 2, 2),
        test_ht_job(6, 0, 2, 2),
        test_ht_job(2, 2, 3, 3),
    ];
    let required =
        j2k_native::J2kRequiredBandRegion::new(0, 0, 4, 4).expect("required test region");

    retain_ht_jobs_for_required_region(&mut jobs, Some(required));

    let retained_offsets = jobs.iter().map(|job| job.output_offset).collect::<Vec<_>>();
    assert_eq!(retained_offsets, vec![0, 18]);

    retain_ht_jobs_for_required_region(&mut jobs, None);
    assert!(jobs.is_empty());
}

#[test]
#[ignore = "requires Metal runtime; exercised by the fail-closed Metal release lane"]
fn prepared_ht_direct_plan_groups_cleanup_subbands_before_idwt() {
    let pixels: Vec<u8> = (0..64).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let bytes = encode_htj2k(&pixels, 8, 8, 1, 8, false, &options).expect("encode ht gray8");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();
    let plan = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("direct grayscale plan");
    let ht_subband_steps = plan
        .steps
        .iter()
        .filter(|step| matches!(step, j2k_native::J2kDirectGrayscaleStep::HtSubBand(_)))
        .count();
    assert!(
        ht_subband_steps > 1,
        "fixture must exercise multiple HT sub-band cleanup steps"
    );

    let prepared = prepare_direct_grayscale_plan(&plan).expect("prepared direct plan");
    assert_eq!(
        prepared.ht_groups.len(),
        1,
        "single-tile HTJ2K direct decode should group adjacent HT sub-bands into one cleanup dispatch"
    );
    assert_eq!(prepared.ht_groups[0].members.len(), ht_subband_steps);
    assert!(matches!(
        prepared.steps[prepared.ht_groups[0].start_step],
        PreparedDirectGrayscaleStep::HtSubBand(_)
    ));
}

#[test]
#[ignore = "requires Metal runtime; exercised by the fail-closed Metal release lane"]
fn grouped_ht_direct_plan_uses_one_group_coded_arena() {
    let pixels: Vec<u8> = (0..64).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let bytes = encode_htj2k(&pixels, 8, 8, 1, 8, false, &options).expect("encode ht gray8");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();
    let plan = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("direct grayscale plan");

    reset_direct_tier1_input_buffer_prepares_for_test();
    let prepared = prepare_direct_grayscale_plan(&plan).expect("prepared direct plan");
    assert_eq!(
        prepared.ht_groups.len(),
        1,
        "fixture must exercise one grouped HT dispatch"
    );
    let group = &prepared.ht_groups[0];
    assert!(!group.coded_arena.data.is_empty());
    assert_eq!(
        direct_tier1_input_buffer_prepares_for_test(),
        2,
        "grouped HT dispatch should prepare one coded arena buffer and one job buffer"
    );

    for step in &prepared.steps[group.start_step..group.end_step] {
        let PreparedDirectGrayscaleStep::HtSubBand(sub_band) = step else {
            panic!("HT group should only span HT sub-band steps");
        };
        assert!(sub_band.coded_buffer.is_none());
        assert!(sub_band.jobs_buffer.is_none());
    }
}

#[test]
#[ignore = "requires Metal runtime; exercised by the fail-closed Metal release lane"]
fn prepared_classic_sub_band_decodes_on_cpu_for_hybrid_upload() {
    let pixels: Vec<u8> = (0..64).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let bytes = encode(&pixels, 8, 8, 1, 8, false, &options).expect("encode classic gray8");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();
    let plan = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("direct grayscale plan");
    let prepared = prepare_direct_grayscale_plan(&plan).expect("prepared direct plan");
    let native_sub_band = first_native_classic_sub_band(&plan);
    let prepared_sub_band = first_prepared_classic_sub_band(&prepared);

    let expected = decode_native_classic_sub_band(native_sub_band);
    let actual =
        decode_prepared_classic_sub_band_on_cpu(prepared_sub_band).expect("prepared CPU decode");

    assert_eq!(actual, expected);
}

#[test]
fn cpu_upload_color_prepare_skips_tier1_metal_input_buffers() {
    if !should_run_metal_runtime() {
        return;
    }

    if Device::system_default().is_none() {
        j2k_test_support::metal_device_unavailable_is_skip(module_path!());
        return;
    }

    let pixels = j2k_test_support::gradient_u8(32, 32, 3);
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        ..EncodeOptions::default()
    };
    let bytes = encode(&pixels, 32, 32, 3, 8, false, &options).expect("encode rgb8");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();
    let plan = image
        .build_direct_color_plan_with_context(&mut context)
        .expect("direct color plan");

    reset_direct_tier1_input_buffer_prepares_for_test();
    let metal_prepared = prepare_direct_color_plan(&plan).expect("Metal prepared color plan");
    assert_eq!(metal_prepared.component_plans.len(), 3);
    assert!(
        direct_tier1_input_buffer_prepares_for_test() > 0,
        "normal Metal preparation should build Tier-1 input buffers"
    );

    reset_direct_tier1_input_buffer_prepares_for_test();
    let cpu_upload_prepared =
        prepare_direct_color_plan_for_cpu_upload(&plan).expect("CPUUpload prepared color plan");
    assert_eq!(cpu_upload_prepared.component_plans.len(), 3);
    assert_eq!(
        direct_tier1_input_buffer_prepares_for_test(),
        0,
        "CPUUpload preparation should keep coded Tier-1 payloads on CPU and skip Metal input buffers"
    );
}

fn first_native_classic_sub_band(
    plan: &j2k_native::J2kDirectGrayscalePlan,
) -> &J2kOwnedSubBandPlan {
    plan.steps
        .iter()
        .find_map(|step| match step {
            NativeDirectGrayscaleStep::ClassicSubBand(sub_band) => Some(sub_band),
            _ => None,
        })
        .expect("classic sub-band step")
}

fn first_prepared_classic_sub_band(
    plan: &super::PreparedDirectGrayscalePlan,
) -> &PreparedClassicSubBand {
    plan.steps
        .iter()
        .find_map(|step| match step {
            PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => Some(sub_band),
            _ => None,
        })
        .expect("prepared classic sub-band step")
}

fn cached_direct_color_tier1_input_count(plan: &PreparedDirectColorPlan) -> usize {
    plan.component_plans
        .iter()
        .map(cached_direct_component_tier1_input_count)
        .sum()
}

fn cached_direct_component_tier1_input_count(plan: &super::PreparedDirectGrayscalePlan) -> usize {
    let mut count = 0;
    let mut step_idx = 0;
    while step_idx < plan.steps.len() {
        if let Some(group) = plan.classic_group_starting_at(step_idx) {
            if has_cached_cpu_tier1_coefficients(plan, step_idx, group.total_coefficients) {
                count += 1;
            }
            step_idx = group.end_step;
            continue;
        }
        if let Some(group) = plan.ht_group_starting_at(step_idx) {
            if has_cached_cpu_tier1_coefficients(plan, step_idx, group.total_coefficients) {
                count += 1;
            }
            step_idx = group.end_step;
            continue;
        }
        match &plan.steps[step_idx] {
            PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                let output_len = sub_band.width as usize * sub_band.height as usize;
                if has_cached_cpu_tier1_coefficients(plan, step_idx, output_len) {
                    count += 1;
                }
            }
            PreparedDirectGrayscaleStep::HtSubBand(sub_band) => {
                let output_len = sub_band.width as usize * sub_band.height as usize;
                if has_cached_cpu_tier1_coefficients(plan, step_idx, output_len) {
                    count += 1;
                }
            }
            PreparedDirectGrayscaleStep::Idwt(_) | PreparedDirectGrayscaleStep::Store(_) => {}
        }
        step_idx += 1;
    }
    count
}

fn has_cached_cpu_tier1_coefficients(
    plan: &super::PreparedDirectGrayscalePlan,
    step_idx: usize,
    output_len: usize,
) -> bool {
    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("J2K Metal test CPU Tier-1 cache lookup");
    plan.cached_cpu_tier1_coefficients(&mut budget, step_idx, output_len)
        .expect("CPU Tier-1 cache lookup")
        .is_some()
}

fn decode_native_classic_sub_band(plan: &J2kOwnedSubBandPlan) -> Vec<f32> {
    let mut output = vec![0.0_f32; plan.width as usize * plan.height as usize];
    let jobs = plan
        .jobs
        .iter()
        .map(|job| J2kCodeBlockBatchJob {
            output_x: job.output_x,
            output_y: job.output_y,
            code_block: native_classic_job(job),
        })
        .collect::<Vec<_>>();
    decode_j2k_sub_band_scalar(
        J2kSubBandDecodeJob {
            width: plan.width,
            height: plan.height,
            jobs: &jobs,
        },
        &mut output,
    )
    .expect("native scalar classic sub-band decode");
    output
}

fn native_classic_job(job: &J2kOwnedCodeBlockBatchJob) -> J2kCodeBlockDecodeJob<'_> {
    J2kCodeBlockDecodeJob {
        data: &job.data,
        segments: &job.segments,
        width: job.width,
        height: job.height,
        output_stride: job.output_stride,
        missing_bit_planes: job.missing_bit_planes,
        number_of_coding_passes: job.number_of_coding_passes,
        total_bitplanes: job.total_bitplanes,
        roi_shift: job.roi_shift,
        sub_band_type: job.sub_band_type,
        style: job.style,
        strict: job.strict,
        dequantization_step: job.dequantization_step,
    }
}

#[test]
#[ignore = "requires Metal runtime; exercised by the fail-closed Metal release lane"]
fn prepared_ht_direct_plan_encodes_full_decode_in_one_compute_encoder() {
    let pixels: Vec<u8> = (0..64).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let bytes = encode_htj2k(&pixels, 8, 8, 1, 8, false, &options).expect("encode ht gray8");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();
    let plan = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("direct grayscale plan");
    let prepared = prepare_direct_grayscale_plan(&plan).expect("prepared direct plan");

    assert_eq!(
        prepared_direct_grayscale_plan_compute_encoder_count(&prepared, PixelFormat::Gray8),
        1,
        "prepared single-tile direct decode should keep cleanup, IDWT, and grayscale store in one compute encoder"
    );
}

#[test]
#[ignore = "requires Metal runtime; exercised by the fail-closed Metal release lane"]
fn repeated_prepared_ht_direct_plan_groups_cleanup_subbands_before_idwt() {
    let pixels: Vec<u8> = (0..64).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let bytes = encode_htj2k(&pixels, 8, 8, 1, 8, false, &options).expect("encode ht gray8");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();
    let plan = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("direct grayscale plan");
    let ht_subband_steps = plan
        .steps
        .iter()
        .filter(|step| matches!(step, j2k_native::J2kDirectGrayscaleStep::HtSubBand(_)))
        .count();
    assert!(
        ht_subband_steps > 1,
        "fixture must exercise multiple HT sub-band cleanup steps"
    );

    let prepared = prepare_direct_grayscale_plan(&plan).expect("prepared direct plan");
    assert_eq!(
        prepared_repeated_direct_ht_cleanup_dispatch_count(&prepared),
        1,
        "repeated HTJ2K WSI tile batches should group adjacent sub-band cleanups like the single-tile path"
    );
}

#[test]
#[ignore = "requires Metal runtime; exercised by the fail-closed Metal release lane"]
fn distinct_prepared_ht_direct_plans_support_stacked_component_batch() {
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let bytes_a = encode_htj2k(&(0..64).collect::<Vec<u8>>(), 8, 8, 1, 8, false, &options)
        .expect("encode first ht gray8");
    let bytes_b = encode_htj2k(
        &(0..64).rev().collect::<Vec<u8>>(),
        8,
        8,
        1,
        8,
        false,
        &options,
    )
    .expect("encode second ht gray8");
    let image_a = Image::new(&bytes_a, &DecodeSettings::default()).expect("first image");
    let image_b = Image::new(&bytes_b, &DecodeSettings::default()).expect("second image");
    let mut context_a = DecoderContext::default();
    let mut context_b = DecoderContext::default();
    let plan_a = image_a
        .build_direct_grayscale_plan_with_context(&mut context_a)
        .expect("first direct plan");
    let plan_b = image_b
        .build_direct_grayscale_plan_with_context(&mut context_b)
        .expect("second direct plan");
    let prepared_a = prepare_direct_grayscale_plan(&plan_a).expect("first prepared plan");
    let prepared_b = prepare_direct_grayscale_plan(&plan_b).expect("second prepared plan");

    assert!(
        supports_stacked_direct_component_plane_batch(&[&prepared_a, &prepared_b]),
        "distinct same-shape HTJ2K grayscale plans should be eligible for one stacked batch graph"
    );
}

#[test]
fn hybrid_rgb8_batch_uses_stacked_component_graph() {
    if !should_run_metal_runtime() {
        return;
    }

    let pixels = j2k_test_support::gradient_u8(32, 32, 3);
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        ..EncodeOptions::default()
    };
    let bytes = encode(&pixels, 32, 32, 3, 8, false, &options).expect("encode rgb8");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();
    let plan = image
        .build_direct_color_plan_with_context(&mut context)
        .expect("direct color plan");
    let prepared = Arc::new(prepare_direct_color_plan(&plan).expect("prepared color plan"));
    let _guard = HYBRID_COUNTER_TEST_LOCK
        .lock()
        .expect("hybrid counter lock");
    reset_hybrid_stacked_component_batches_for_test();
    reset_hybrid_cpu_decode_worker_inits_for_test();

    let surfaces = execute_hybrid_cpu_tier1_direct_color_plan_batch(
        &[prepared.clone(), prepared],
        PixelFormat::Rgb8,
    )
    .expect("hybrid RGB8 batch");

    assert_eq!(surfaces.len(), 2);
    assert!(
        hybrid_stacked_component_batches_for_test() >= 3,
        "hybrid RGB batch should stack each component plane instead of encoding each tile/component serially"
    );
    assert!(
        hybrid_cpu_decode_worker_inits_for_test() > 0,
        "hybrid RGB batch should use worker-local CPU decode scratch instead of per-input decode/flatten"
    );
}

#[test]
fn hybrid_rgb8_repeated_batch_decodes_shared_tier1_inputs_once() {
    if !should_run_metal_runtime() {
        return;
    }

    let pixels = j2k_test_support::gradient_u8(32, 32, 3);
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        ..EncodeOptions::default()
    };
    let bytes = encode(&pixels, 32, 32, 3, 8, false, &options).expect("encode rgb8");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();
    let plan = image
        .build_direct_color_plan_with_context(&mut context)
        .expect("direct color plan");
    let prepared = Arc::new(prepare_direct_color_plan(&plan).expect("prepared color plan"));
    let unique_tier1_inputs = prepared_direct_color_tier1_input_count(&prepared);
    assert!(
        unique_tier1_inputs > 0,
        "fixture should have Tier-1 inputs to decode"
    );
    let _guard = HYBRID_COUNTER_TEST_LOCK
        .lock()
        .expect("hybrid counter lock");
    reset_hybrid_cpu_decode_inputs_for_test();
    reset_thread_hybrid_cpu_decode_inputs_for_test();

    let surfaces = execute_hybrid_cpu_tier1_direct_color_plan_batch(
        &[prepared.clone(), prepared.clone(), prepared],
        PixelFormat::Rgb8,
    )
    .expect("hybrid repeated RGB8 batch");

    assert_eq!(surfaces.len(), 3);
    assert!(
        hybrid_cpu_decode_inputs_for_test() >= unique_tier1_inputs,
        "repeated RGB hybrid batches should decode the shared coefficient inputs"
    );
}

#[test]
fn hybrid_rgb8_reused_plan_caches_cpu_tier1_inputs_across_calls() {
    if !should_run_metal_runtime() {
        return;
    }

    let pixels = j2k_test_support::gradient_u8(32, 32, 3);
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        ..EncodeOptions::default()
    };
    let bytes = encode(&pixels, 32, 32, 3, 8, false, &options).expect("encode rgb8");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();
    let plan = image
        .build_direct_color_plan_with_context(&mut context)
        .expect("direct color plan");
    let prepared = Arc::new(prepare_direct_color_plan(&plan).expect("prepared color plan"));
    let unique_tier1_inputs = prepared_direct_color_tier1_input_count(&prepared);
    assert!(
        unique_tier1_inputs > 0,
        "fixture should have Tier-1 inputs to decode"
    );
    let _guard = HYBRID_COUNTER_TEST_LOCK
        .lock()
        .expect("hybrid counter lock");
    reset_hybrid_cpu_decode_inputs_for_test();
    reset_thread_hybrid_cpu_decode_inputs_for_test();

    let surfaces = execute_hybrid_cpu_tier1_direct_color_plan_batch(
        &[prepared.clone(), prepared.clone()],
        PixelFormat::Rgb8,
    )
    .expect("first hybrid repeated RGB8 batch");
    assert_eq!(surfaces.len(), 2);
    assert_eq!(
        cached_direct_color_tier1_input_count(&prepared),
        unique_tier1_inputs,
        "first RGB hybrid call should cache every decoded CPU Tier-1 input"
    );
    assert_eq!(
        thread_hybrid_cpu_decode_inputs_for_test(),
        unique_tier1_inputs,
        "first RGB hybrid call should decode each shared Tier-1 input once"
    );

    let surfaces = execute_hybrid_cpu_tier1_direct_color_plan_batch(
        &[prepared.clone(), prepared.clone()],
        PixelFormat::Rgb8,
    )
    .expect("second hybrid repeated RGB8 batch");
    assert_eq!(surfaces.len(), 2);
    assert_eq!(
        cached_direct_color_tier1_input_count(&prepared),
        unique_tier1_inputs,
        "second RGB hybrid call should keep every decoded CPU Tier-1 input cached"
    );
    assert_eq!(
        thread_hybrid_cpu_decode_inputs_for_test(),
        unique_tier1_inputs,
        "second RGB hybrid call must reuse cached CPU Tier-1 coefficients without re-decoding"
    );
}

#[test]
fn hybrid_rgb8_repeated_batch_decodes_once_and_blits_distinct_outputs() {
    if !should_run_metal_runtime() {
        return;
    }

    let pixels = j2k_test_support::gradient_u8(32, 32, 3);
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        ..EncodeOptions::default()
    };
    let bytes = encode(&pixels, 32, 32, 3, 8, false, &options).expect("encode rgb8");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();
    let plan = image
        .build_direct_color_plan_with_context(&mut context)
        .expect("direct color plan");
    let prepared = Arc::new(prepare_direct_color_plan(&plan).expect("prepared color plan"));
    let _guard = HYBRID_COUNTER_TEST_LOCK
        .lock()
        .expect("hybrid counter lock");
    reset_hybrid_repeated_output_blits_for_test();

    let surfaces = execute_hybrid_cpu_tier1_direct_color_plan_batch(
        &[
            prepared.clone(),
            prepared.clone(),
            prepared.clone(),
            prepared,
        ],
        PixelFormat::Rgb8,
    )
    .expect("hybrid repeated RGB8 batch");

    assert_eq!(surfaces.len(), 4);
    let surface_bytes = surfaces[0].as_bytes().expect("surface byte access").len();
    let offsets = surfaces
        .iter()
        .map(|surface| {
            surface
                .metal_buffer_trusted()
                .expect("resident Metal surface")
                .1
        })
        .collect::<Vec<_>>();
    assert_eq!(
        offsets,
        (0..surfaces.len())
            .map(|index| index * surface_bytes)
            .collect::<Vec<_>>(),
        "repeated outputs must retain distinct Metal buffer offsets"
    );
    for surface in &surfaces[1..] {
        assert_eq!(
            surface.as_bytes().expect("surface byte access"),
            surfaces[0].as_bytes().expect("surface byte access"),
            "repeated outputs should remain byte-identical"
        );
    }
    assert_eq!(
        hybrid_repeated_output_blits_for_test(),
        2,
        "repeated RGB hybrid batches should duplicate packed output surfaces with logarithmic Metal blit ranges"
    );
}

#[test]
fn hybrid_rgb8_distinct_batch_keeps_tier1_inputs_separate() {
    if !should_run_metal_runtime() {
        return;
    }

    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        ..EncodeOptions::default()
    };
    let bytes_a = encode(
        &j2k_test_support::gradient_variant_u8(32, 32, 3, 0),
        32,
        32,
        3,
        8,
        false,
        &options,
    )
    .expect("encode first rgb8");
    let bytes_b = encode(
        &j2k_test_support::gradient_variant_u8(32, 32, 3, 7),
        32,
        32,
        3,
        8,
        false,
        &options,
    )
    .expect("encode second rgb8");
    let image_a = Image::new(&bytes_a, &DecodeSettings::default()).expect("first image");
    let image_b = Image::new(&bytes_b, &DecodeSettings::default()).expect("second image");
    let mut context_a = DecoderContext::default();
    let mut context_b = DecoderContext::default();
    let plan_a = image_a
        .build_direct_color_plan_with_context(&mut context_a)
        .expect("first direct color plan");
    let plan_b = image_b
        .build_direct_color_plan_with_context(&mut context_b)
        .expect("second direct color plan");
    let prepared_a = Arc::new(prepare_direct_color_plan(&plan_a).expect("first prepared"));
    let prepared_b = Arc::new(prepare_direct_color_plan(&plan_b).expect("second prepared"));
    let expected_inputs = prepared_direct_color_tier1_input_count(&prepared_a)
        + prepared_direct_color_tier1_input_count(&prepared_b);
    let _guard = HYBRID_COUNTER_TEST_LOCK
        .lock()
        .expect("hybrid counter lock");
    reset_hybrid_cpu_decode_inputs_for_test();

    let surfaces = execute_hybrid_cpu_tier1_direct_color_plan_batch(
        &[prepared_a, prepared_b],
        PixelFormat::Rgb8,
    )
    .expect("hybrid distinct RGB8 batch");

    assert_eq!(surfaces.len(), 2);
    assert_ne!(
        surfaces[0].as_bytes().expect("surface byte access"),
        surfaces[1].as_bytes().expect("surface byte access"),
        "distinct RGB inputs must not reuse the first tile's decoded coefficients"
    );
    assert_eq!(
        thread_hybrid_cpu_decode_inputs_for_test(),
        expected_inputs,
        "distinct RGB hybrid batches should decode each tile's own Tier-1 inputs"
    );
}

#[test]
fn hybrid_rgb8_flattened_cpu_tier1_batch_uses_one_decode_queue() {
    if !should_run_metal_runtime() {
        return;
    }

    let pixels_a = j2k_test_support::gradient_variant_u8(32, 32, 3, 0);
    let pixels_b = j2k_test_support::gradient_variant_u8(32, 32, 3, 11);
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        ..EncodeOptions::default()
    };
    let bytes_a = encode(&pixels_a, 32, 32, 3, 8, false, &options).expect("encode first rgb8");
    let bytes_b = encode(&pixels_b, 32, 32, 3, 8, false, &options).expect("encode second rgb8");
    let image_a = Image::new(&bytes_a, &DecodeSettings::default()).expect("first image");
    let image_b = Image::new(&bytes_b, &DecodeSettings::default()).expect("second image");
    let mut context_a = DecoderContext::default();
    let mut context_b = DecoderContext::default();
    let plan_a = image_a
        .build_direct_color_plan_with_context(&mut context_a)
        .expect("first direct color plan");
    let plan_b = image_b
        .build_direct_color_plan_with_context(&mut context_b)
        .expect("second direct color plan");
    let prepared_a = Arc::new(prepare_direct_color_plan(&plan_a).expect("first prepared"));
    let prepared_b = Arc::new(prepare_direct_color_plan(&plan_b).expect("second prepared"));
    let expected_inputs = prepared_direct_color_tier1_input_count(&prepared_a)
        + prepared_direct_color_tier1_input_count(&prepared_b);
    let _guard = HYBRID_COUNTER_TEST_LOCK
        .lock()
        .expect("hybrid counter lock");
    reset_hybrid_cpu_decode_inputs_for_test();
    reset_flattened_hybrid_cpu_decode_batches_for_test();

    let surfaces = execute_flattened_hybrid_cpu_tier1_direct_color_plan_batch_for_test(
        &[prepared_a, prepared_b],
        PixelFormat::Rgb8,
    )
    .expect("flattened hybrid distinct RGB8 batch");

    assert_eq!(surfaces.len(), 2);
    assert_ne!(
        surfaces[0].as_bytes().expect("surface byte access"),
        surfaces[1].as_bytes().expect("surface byte access"),
        "flattened distinct RGB hybrid batches must keep each tile's coefficients separate"
    );
    assert!(
        hybrid_cpu_decode_inputs_for_test() >= expected_inputs,
        "flattened RGB hybrid batches should still decode every distinct Tier-1 input"
    );
    assert!(
        flattened_hybrid_cpu_decode_batches_for_test() >= 1,
        "flattened RGB hybrid should collect Tier-1 work through the flattened CPU decode queue"
    );
}

#[test]
fn flattened_cpu_tier1_default_gate_targets_large_distinct_batches_only() {
    fn color_plan(width: u32, height: u32) -> Arc<PreparedDirectColorPlan> {
        Arc::new(PreparedDirectColorPlan {
            dimensions: (width, height),
            bit_depths: [8, 8, 8],
            mct: true,
            transform: J2kWaveletTransform::Reversible53,
            component_plans: Vec::new(),
        })
    }

    let repeated = vec![color_plan(1024, 1024); 16];
    assert!(
        !should_flatten_hybrid_cpu_tier1_color_batch(&repeated),
        "repeated RGB batches already win through shared Tier-1 decode and should not use the flattened distinct scheduler"
    );

    let small_distinct = (0..16).map(|_| color_plan(256, 256)).collect::<Vec<_>>();
    assert!(
        !should_flatten_hybrid_cpu_tier1_color_batch(&small_distinct),
        "small RGB batches measured slower with flattened Tier-1 and should stay on the grouped path"
    );

    let large_distinct = (0..16).map(|_| color_plan(1024, 1024)).collect::<Vec<_>>();
    assert!(
        should_flatten_hybrid_cpu_tier1_color_batch(&large_distinct),
        "large distinct RGB explicit hybrid batches measured faster with flattened Tier-1"
    );
}

#[test]
fn hybrid_cpu_decode_worker_count_allows_two_way_small_batch_parallelism() {
    let available = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    if available < 2 {
        return;
    }

    assert_eq!(
        hybrid_cpu_decode_worker_count(2),
        2,
        "two independent hybrid CPU Tier-1 inputs should be able to use two workers"
    );
}

#[test]
#[ignore = "requires Metal runtime; exercised by the fail-closed Metal release lane"]
fn cropped_region_scaled_ht_direct_plan_prunes_codeblocks_outside_output_roi() {
    let mut pixels = Vec::with_capacity(256 * 256);
    for y in 0..256u32 {
        for x in 0..256u32 {
            pixels.push(((x * 3 + y * 5) & 0xff) as u8);
        }
    }
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 3,
        code_block_width_exp: 0,
        code_block_height_exp: 0,
        ..EncodeOptions::default()
    };
    let bytes = encode_htj2k(&pixels, 256, 256, 1, 8, false, &options).expect("encode ht gray8");
    let image = Image::new(
        &bytes,
        &DecodeSettings {
            target_resolution: Some((64, 64)),
            ..DecodeSettings::default()
        },
    )
    .expect("scaled image");
    let mut context = DecoderContext::default();
    let plan = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("direct grayscale plan");
    let mut prepared = prepare_direct_grayscale_plan(&plan).expect("prepared direct plan");
    let full_jobs = prepared_direct_grayscale_ht_job_count(&prepared);
    assert!(
        full_jobs > 8,
        "fixture should have multiple HT code-block jobs"
    );

    crop_prepared_direct_grayscale_plan_to_output_region(
        &mut prepared,
        j2k_core::Rect {
            x: 24,
            y: 24,
            w: 8,
            h: 8,
        },
    )
    .expect("crop direct plan");
    let cropped_jobs = prepared_direct_grayscale_ht_job_count(&prepared);

    assert!(
        cropped_jobs > 0 && cropped_jobs < full_jobs,
        "cropped ROI should prune HT code-block jobs; full={full_jobs}, cropped={cropped_jobs}"
    );
}

#[test]
#[ignore = "requires Metal runtime; exercised by the fail-closed Metal release lane"]
fn cropped_region_scaled_ht_direct_plan_compacts_coded_payloads() {
    let mut pixels = Vec::with_capacity(256 * 256);
    for y in 0..256u32 {
        for x in 0..256u32 {
            pixels.push(((x * 3 + y * 5) & 0xff) as u8);
        }
    }
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 3,
        code_block_width_exp: 0,
        code_block_height_exp: 0,
        ..EncodeOptions::default()
    };
    let bytes = encode_htj2k(&pixels, 256, 256, 1, 8, false, &options).expect("encode ht gray8");
    let image = Image::new(
        &bytes,
        &DecodeSettings {
            target_resolution: Some((64, 64)),
            ..DecodeSettings::default()
        },
    )
    .expect("scaled image");
    let mut context = DecoderContext::default();
    let plan = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("direct grayscale plan");
    let mut prepared = prepare_direct_grayscale_plan(&plan).expect("prepared direct plan");
    let full_bytes = prepared_direct_grayscale_ht_coded_byte_count(&prepared);
    assert!(full_bytes > 0, "fixture should carry HT coded payloads");

    crop_prepared_direct_grayscale_plan_to_output_region(
        &mut prepared,
        j2k_core::Rect {
            x: 24,
            y: 24,
            w: 8,
            h: 8,
        },
    )
    .expect("crop direct plan");
    let cropped_bytes = prepared_direct_grayscale_ht_coded_byte_count(&prepared);

    assert!(
        cropped_bytes > 0 && cropped_bytes < full_bytes,
        "cropped ROI should compact HT coded bytes; full={full_bytes}, cropped={cropped_bytes}"
    );
}

#[test]
#[ignore = "requires Metal runtime; exercised by the fail-closed Metal release lane"]
fn cropped_region_scaled_ht_direct_plan_reduces_idwt_output_work() {
    let mut pixels = Vec::with_capacity(128 * 128);
    for y in 0..128u32 {
        for x in 0..128u32 {
            pixels.push(((x * 3 + y * 5) & 0xff) as u8);
        }
    }
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 3,
        code_block_width_exp: 0,
        code_block_height_exp: 0,
        ..EncodeOptions::default()
    };
    let bytes = encode_htj2k(&pixels, 128, 128, 1, 8, false, &options).expect("encode ht gray8");
    let image = Image::new(
        &bytes,
        &DecodeSettings {
            target_resolution: Some((32, 32)),
            ..DecodeSettings::default()
        },
    )
    .expect("scaled image");
    let mut context = DecoderContext::default();
    let plan = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("direct grayscale plan");
    let mut prepared = prepare_direct_grayscale_plan(&plan).expect("prepared direct plan");
    let full_samples = prepared_direct_grayscale_idwt_output_sample_count(&prepared);

    crop_prepared_direct_grayscale_plan_to_output_region(
        &mut prepared,
        j2k_core::Rect {
            x: 10,
            y: 10,
            w: 4,
            h: 4,
        },
    )
    .expect("crop direct plan");
    let cropped_samples = prepared_direct_grayscale_idwt_output_sample_count(&prepared);

    assert!(
        cropped_samples > 0 && cropped_samples < full_samples,
        "cropped ROI should reduce IDWT output work; full={full_samples}, cropped={cropped_samples}"
    );
}

#[test]
#[ignore = "requires Metal runtime; exercised by the fail-closed Metal release lane"]
fn cropped_region_ht_direct_plan_keeps_idwt_windows_bounded() {
    let mut pixels = Vec::with_capacity(256 * 256);
    for y in 0..256u32 {
        for x in 0..256u32 {
            pixels.push(((x * 3 + y * 5) & 0xff) as u8);
        }
    }
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 3,
        code_block_width_exp: 0,
        code_block_height_exp: 0,
        ..EncodeOptions::default()
    };
    let bytes = encode_htj2k(&pixels, 256, 256, 1, 8, false, &options).expect("encode ht gray8");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();
    let plan = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("direct grayscale plan");
    let mut prepared = prepare_direct_grayscale_plan(&plan).expect("prepared direct plan");
    let idwt_levels = prepared_direct_grayscale_idwt_full_and_prepared_lens(&prepared);
    assert!(
        idwt_levels.len() >= 3,
        "fixture should exercise a multi-level IDWT plan"
    );

    crop_prepared_direct_grayscale_plan_to_output_region(
        &mut prepared,
        j2k_core::Rect {
            x: 112,
            y: 112,
            w: 32,
            h: 32,
        },
    )
    .expect("crop direct plan");
    let cropped_idwt_levels = prepared_direct_grayscale_idwt_full_and_prepared_lens(&prepared);

    assert_eq!(cropped_idwt_levels.len(), idwt_levels.len());
    for (level_idx, (full_len, cropped_len)) in cropped_idwt_levels.iter().copied().enumerate() {
        assert!(
            cropped_len > 0 && cropped_len <= full_len,
            "cropped ROI should keep IDWT level {level_idx} bounded; full={full_len}, cropped={cropped_len}"
        );
    }
    assert!(
        cropped_idwt_levels
            .iter()
            .any(|(full_len, cropped_len)| cropped_len < full_len),
        "cropped ROI should reduce at least one IDWT level"
    );
}

fn prepared_direct_grayscale_ht_job_count(plan: &super::PreparedDirectGrayscalePlan) -> usize {
    plan.steps
        .iter()
        .map(|step| match step {
            PreparedDirectGrayscaleStep::HtSubBand(sub_band) => sub_band.jobs.len(),
            _ => 0,
        })
        .sum()
}

fn prepared_direct_grayscale_ht_coded_byte_count(
    plan: &super::PreparedDirectGrayscalePlan,
) -> usize {
    plan.steps
        .iter()
        .map(|step| match step {
            PreparedDirectGrayscaleStep::HtSubBand(sub_band) => sub_band.coded_data.len(),
            _ => 0,
        })
        .sum()
}

fn prepared_direct_grayscale_idwt_output_sample_count(
    plan: &super::PreparedDirectGrayscalePlan,
) -> usize {
    plan.steps
        .iter()
        .map(|step| match step {
            PreparedDirectGrayscaleStep::Idwt(idwt) => prepared_idwt_output_len(idwt),
            _ => 0,
        })
        .sum()
}

fn prepared_direct_grayscale_idwt_full_and_prepared_lens(
    plan: &super::PreparedDirectGrayscalePlan,
) -> Vec<(usize, usize)> {
    plan.steps
        .iter()
        .filter_map(|step| match step {
            PreparedDirectGrayscaleStep::Idwt(idwt) => Some((
                idwt.step.rect.width() as usize * idwt.step.rect.height() as usize,
                prepared_idwt_output_len(idwt),
            )),
            _ => None,
        })
        .collect()
}

#[test]
#[ignore = "requires Metal runtime; exercised by the fail-closed Metal release lane"]
fn prepared_classic_direct_plan_groups_cleanup_subbands_before_idwt() {
    let pixels: Vec<u8> = (0..64).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let bytes = encode(&pixels, 8, 8, 1, 8, false, &options).expect("encode j2k gray8");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();
    let plan = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("direct grayscale plan");
    let classic_subband_steps = plan
        .steps
        .iter()
        .filter(|step| matches!(step, j2k_native::J2kDirectGrayscaleStep::ClassicSubBand(_)))
        .count();
    assert!(
        classic_subband_steps > 1,
        "fixture must exercise multiple classic sub-band cleanup steps"
    );

    let prepared = prepare_direct_grayscale_plan(&plan).expect("prepared direct plan");
    assert_eq!(
        prepared.classic_groups.len(),
        1,
        "classic J2K direct decode should group adjacent sub-band cleanups before IDWT"
    );
    assert_eq!(
        prepared.classic_groups[0].members.len(),
        classic_subband_steps
    );
    assert!(matches!(
        prepared.steps[prepared.classic_groups[0].start_step],
        PreparedDirectGrayscaleStep::ClassicSubBand(_)
    ));
}

#[test]
fn classic_plain_fast_path_accepts_style_zero_arithmetic_jobs() {
    let jobs = [J2kClassicCleanupBatchJob {
        coded_offset: 0,
        coded_len: 1,
        segment_offset: 0,
        segment_count: 1,
        width: 64,
        height: 64,
        output_stride: 64,
        output_offset: 0,
        missing_msbs: 0,
        total_bitplanes: 8,
        roi_shift: 0,
        number_of_coding_passes: 1,
        sub_band_type: 0,
        style_flags: 0,
        strict: 1,
        dequantization_step: 1.0,
    }];
    let segments = [J2kClassicSegment {
        data_offset: 0,
        data_length: 1,
        start_coding_pass: 0,
        end_coding_pass: 1,
        use_arithmetic: 1,
    }];

    assert!(
        classic_batch_uses_plain_fast_path(&jobs, &segments),
        "style-0 arithmetic-only classic J2K jobs should use the fused plain cleanup/store kernel"
    );
}

#[test]
fn classic_repeated_plain_fast_path_stays_off_for_wsi_batch_size() {
    let jobs = [J2kClassicCleanupBatchJob {
        coded_offset: 0,
        coded_len: 1,
        segment_offset: 0,
        segment_count: 1,
        width: 64,
        height: 64,
        output_stride: 64,
        output_offset: 0,
        missing_msbs: 0,
        total_bitplanes: 8,
        roi_shift: 0,
        number_of_coding_passes: 1,
        sub_band_type: 0,
        style_flags: 0,
        strict: 1,
        dequantization_step: 1.0,
    }];
    let segments = [J2kClassicSegment {
        data_offset: 0,
        data_length: 1,
        start_coding_pass: 0,
        end_coding_pass: 1,
        use_arithmetic: 1,
    }];

    assert!(
        !classic_repeated_uses_plain_fast_path(16, &jobs, &segments),
        "batch-16 WSI classic J2K should keep the device-state cleanup plus separate store path"
    );
}

#[test]
fn repeated_gray_store_detects_contiguous_full_wsi_tiles() {
    let full_tile = J2kRepeatedGrayStoreParams {
        input_width: 1024,
        input_height: 1024,
        source_x: 0,
        source_y: 0,
        copy_width: 1024,
        copy_height: 1024,
        output_width: 1024,
        output_height: 1024,
        output_x: 0,
        output_y: 0,
        addend: 0.0,
        batch_count: 16,
        max_value: 255.0,
        u8_scale: 1.0,
        u16_scale: 257.0,
    };
    assert!(
        repeated_gray_store_is_contiguous_full_surface(full_tile),
        "full repeated grayscale WSI stores should use the contiguous store kernel"
    );

    let windowed = J2kRepeatedGrayStoreParams {
        source_x: 1,
        copy_width: 1023,
        ..full_tile
    };
    assert!(
        !repeated_gray_store_is_contiguous_full_surface(windowed),
        "ROI/windowed repeated grayscale stores must stay on the generic store kernel"
    );
}
