// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use super::super::abi::{J2kClassicCleanupBatchJob, J2kClassicSegment, J2kRepeatedGrayStoreParams};
use super::super::decode_dispatch::store::repeated_gray_store_is_contiguous_full_surface;
use super::super::decode_dispatch::{
    classic_batch_uses_plain_fast_path, classic_repeated_uses_plain_fast_path,
};
use super::super::{
    decode_prepared_classic_sub_band_on_cpu, direct_tier1_input_buffer_prepares_for_test,
    execute_hybrid_cpu_tier1_direct_color_plan, execute_prepared_direct_grayscale_plan,
    prepare_direct_color_plan, prepare_direct_color_plan_for_cpu_upload,
    prepare_direct_grayscale_plan, reset_direct_tier1_input_buffer_prepares_for_test,
    ClassicTier1Buffers, MetalRuntime, PreparedClassicSubBand, PreparedDirectColorPlan,
    PreparedDirectGrayscalePlan, PreparedDirectGrayscaleStep,
};
use super::runtime::should_run_metal_runtime;
use crate::metal_types::prelude::*;
use j2k_native::{
    decode_j2k_sub_band_scalar, encode, DecodeSettings, DecoderContext, EncodeOptions, Image,
    J2kCodeBlockBatchJob, J2kCodeBlockDecodeJob,
    J2kDirectGrayscaleStep as NativeDirectGrayscaleStep, J2kOwnedCodeBlockBatchJob,
    J2kOwnedSubBandPlan, J2kSubBandDecodeJob,
};
use std::sync::Arc;

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
#[ignore = "requires Metal runtime; exercised by the fail-closed Metal release lane"]
fn prepared_irreversible_classic_sub_band_records_midpoint_reconstruction() {
    let pixels: Vec<u8> = (0..64).collect();
    let options = EncodeOptions {
        reversible: false,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let bytes = encode(&pixels, 8, 8, 1, 8, false, &options).expect("encode 9/7 gray8");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();
    let plan = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("direct grayscale plan");
    let prepared = prepare_direct_grayscale_plan(&plan).expect("prepared direct plan");
    let prepared_sub_band = first_prepared_classic_sub_band(&prepared);

    assert!(
        prepared_sub_band
            .jobs
            .iter()
            .all(|job| job.irreversible_midpoint != 0),
        "every prepared job in a 9/7 sub-band must retain midpoint reconstruction"
    );
}

#[test]
#[ignore = "requires Metal runtime; exercised by the fail-closed Metal release lane"]
fn irreversible_hybrid_cpu_tier1_matches_native_decode_exactly() {
    let pixels = j2k_test_support::gradient_u8(16, 16, 3);
    let bytes = encode(
        &pixels,
        16,
        16,
        3,
        8,
        false,
        &EncodeOptions {
            reversible: false,
            num_decomposition_levels: 2,
            ..EncodeOptions::default()
        },
    )
    .expect("encode irreversible RGB8");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut expected_context = DecoderContext::default();
    let expected = image
        .decode_with_context(&mut expected_context)
        .expect("native decode");
    let mut plan_context = DecoderContext::default();
    let plan = image
        .build_direct_color_plan_with_context(&mut plan_context)
        .expect("direct color plan");
    let prepared = prepare_direct_color_plan_for_cpu_upload(&plan).expect("prepared color plan");

    let surface =
        execute_hybrid_cpu_tier1_direct_color_plan(Arc::new(prepared), j2k_core::PixelFormat::Rgb8)
            .expect("hybrid decode");

    assert_eq!(
        surface.as_bytes().expect("surface bytes").as_ref(),
        expected.data
    );
}

#[test]
fn cpu_upload_color_prepare_skips_tier1_metal_input_buffers() {
    if !should_run_metal_runtime() {
        return;
    }

    if j2k_metal_support::system_default_device().is_err() {
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
    assert_eq!(
        direct_tier1_input_buffer_prepares_for_test(),
        0,
        "Metal preparation should leave Tier-1 input buffers to their first reader"
    );
    let runtime = MetalRuntime::new().expect("Metal runtime");
    let read = read_color_plan_tier1_buffers(&runtime, &metal_prepared).len();
    assert!(read > 0, "fixture must have classic sub-bands");
    assert_eq!(
        direct_tier1_input_buffer_prepares_for_test(),
        3 * read,
        "the first read should build each sub-band's and group's three Tier-1 input buffers"
    );

    reset_direct_tier1_input_buffer_prepares_for_test();
    let cpu_upload_prepared =
        prepare_direct_color_plan_for_cpu_upload(&plan).expect("CPUUpload prepared color plan");
    assert_eq!(cpu_upload_prepared.component_plans.len(), 3);
    read_color_plan_tier1_buffers(&runtime, &cpu_upload_prepared);
    assert_eq!(
        direct_tier1_input_buffer_prepares_for_test(),
        0,
        "CPUUpload plans should keep coded Tier-1 payloads on CPU and skip Metal input buffers"
    );
}

#[test]
fn classic_tier1_buffers_are_built_once_and_charged_from_preparation() {
    if !should_run_metal_runtime() {
        return;
    }
    let bytes = encoded_classic(32, 32, 1);
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let plan = image
        .build_direct_grayscale_plan_with_context(&mut DecoderContext::default())
        .expect("direct grayscale plan");
    let prepared = prepare_direct_grayscale_plan(&plan).expect("prepared grayscale plan");
    assert!(
        !prepared.classic_groups.is_empty(),
        "fixture must exercise grouped classic sub-bands"
    );
    let charged = prepared.retained_cache_bytes().expect("retained bytes");
    let runtime = MetalRuntime::new().expect("Metal runtime");

    reset_direct_tier1_input_buffer_prepares_for_test();
    let first = read_grayscale_plan_tier1_buffers(&runtime, &prepared);
    let built = direct_tier1_input_buffer_prepares_for_test();
    assert_eq!(built, 3 * first.len());
    let second = read_grayscale_plan_tier1_buffers(&runtime, &prepared);
    assert_eq!(
        direct_tier1_input_buffer_prepares_for_test(),
        built,
        "later reads should reuse the buffers built by the first"
    );
    for (first, second) in first.iter().zip(&second) {
        assert!(std::ptr::eq(*first, *second));
    }

    assert_eq!(
        prepared.retained_cache_bytes().expect("retained bytes"),
        charged,
        "building the buffers must not change the weight the cache charged at insertion"
    );
    let allocated = first
        .iter()
        .map(|buffers| buffers.coded.length() + buffers.jobs.length() + buffers.segments.length())
        .sum::<usize>();
    assert_eq!(
        charged.device, allocated,
        "the cache should charge exactly the device bytes the buffers occupy"
    );
}

#[test]
fn single_image_route_builds_tier1_buffers_once_per_plan() {
    if !should_run_metal_runtime() {
        return;
    }
    let bytes = encoded_classic(32, 32, 1);
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let plan = prepare_direct_grayscale_plan(
        &image
            .build_direct_grayscale_plan_with_context(&mut DecoderContext::default())
            .expect("direct grayscale plan"),
    )
    .expect("prepared grayscale plan");
    let expected = cpu_decode(&bytes, 32, j2k_core::PixelFormat::Gray8);

    reset_direct_tier1_input_buffer_prepares_for_test();
    for pass in ["first", "second"] {
        let surface = execute_prepared_direct_grayscale_plan(&plan, j2k_core::PixelFormat::Gray8)
            .expect("single-image decode");
        assert_eq!(
            surface.as_bytes().expect("surface bytes").as_ref(),
            expected,
            "{pass} pass"
        );
        let built = built_grayscale_plan_tier1_buffers(&plan);
        assert!(built > 0, "the single-image route reads the plan's buffers");
        assert_eq!(
            direct_tier1_input_buffer_prepares_for_test(),
            3 * built,
            "{pass} pass"
        );
    }
}

pub(super) fn encoded_classic(width: u32, height: u32, components: u16) -> Vec<u8> {
    let pixels = j2k_test_support::gradient_u8(width, height, usize::from(components));
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        ..EncodeOptions::default()
    };
    encode(&pixels, width, height, components, 8, false, &options).expect("encode classic")
}

pub(super) fn cpu_decode(bytes: &[u8], width: usize, format: j2k_core::PixelFormat) -> Vec<u8> {
    let mut decoder = j2k::J2kDecoder::new(bytes).expect("CPU decoder");
    let row_bytes = width * format.channels();
    let mut output = vec![0; row_bytes * decoder.info().dimensions.1 as usize];
    decoder
        .decode_into(&mut output, row_bytes, format)
        .expect("CPU decode");
    output
}

fn read_grayscale_plan_tier1_buffers<'a>(
    runtime: &MetalRuntime,
    plan: &'a PreparedDirectGrayscalePlan,
) -> Vec<&'a ClassicTier1Buffers> {
    let sub_bands = plan.steps.iter().filter_map(|step| match step {
        PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
            Some(sub_band.tier1_buffers(runtime))
        }
        _ => None,
    });
    let groups = plan
        .classic_groups
        .iter()
        .map(|group| group.tier1_buffers(runtime));
    sub_bands
        .chain(groups)
        .collect::<Result<_, _>>()
        .expect("classic Tier-1 buffers")
}

fn read_color_plan_tier1_buffers<'a>(
    runtime: &MetalRuntime,
    plan: &'a PreparedDirectColorPlan,
) -> Vec<&'a ClassicTier1Buffers> {
    plan.component_plans
        .iter()
        .flat_map(|component| read_grayscale_plan_tier1_buffers(runtime, component))
        .collect()
}

pub(super) fn built_grayscale_plan_tier1_buffers(plan: &PreparedDirectGrayscalePlan) -> usize {
    let sub_bands = plan.steps.iter().filter(|step| {
        matches!(step, PreparedDirectGrayscaleStep::ClassicSubBand(sub_band)
            if sub_band.tier1_inputs.buffers.get().is_some())
    });
    let groups = plan
        .classic_groups
        .iter()
        .filter(|group| group.tier1_inputs.buffers.get().is_some());
    sub_bands.count() + groups.count()
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

fn first_prepared_classic_sub_band(plan: &PreparedDirectGrayscalePlan) -> &PreparedClassicSubBand {
    plan.steps
        .iter()
        .find_map(|step| match step {
            PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => Some(sub_band),
            _ => None,
        })
        .expect("prepared classic sub-band step")
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
        irreversible_midpoint: 0,
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
        irreversible_midpoint: 0,
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
