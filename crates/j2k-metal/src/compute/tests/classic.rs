// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::abi::{J2kClassicCleanupBatchJob, J2kClassicSegment, J2kRepeatedGrayStoreParams};
use super::super::decode_dispatch::store::repeated_gray_store_is_contiguous_full_surface;
use super::super::decode_dispatch::{
    classic_batch_uses_plain_fast_path, classic_repeated_uses_plain_fast_path,
};
use super::super::{
    decode_prepared_classic_sub_band_on_cpu, direct_tier1_input_buffer_prepares_for_test,
    prepare_direct_color_plan, prepare_direct_color_plan_for_cpu_upload,
    prepare_direct_grayscale_plan, reset_direct_tier1_input_buffer_prepares_for_test,
    PreparedClassicSubBand, PreparedDirectGrayscalePlan, PreparedDirectGrayscaleStep,
};
use super::runtime::should_run_metal_runtime;
use j2k_native::{
    decode_j2k_sub_band_scalar, encode, DecodeSettings, DecoderContext, EncodeOptions, Image,
    J2kCodeBlockBatchJob, J2kCodeBlockDecodeJob,
    J2kDirectGrayscaleStep as NativeDirectGrayscaleStep, J2kOwnedCodeBlockBatchJob,
    J2kOwnedSubBandPlan, J2kSubBandDecodeJob,
};
use metal::Device;

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
