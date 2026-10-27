// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::abi::J2kClassicCleanupBatchJob;
use super::super::direct_roi::prepared_idwt_output_len;
use super::super::{
    crop_prepared_direct_grayscale_plan_to_output_region, prepare_direct_grayscale_plan,
    MetalRuntime, PreparedDirectGrayscalePlan, PreparedDirectGrayscaleStep,
    PreparedHtPayloadSource,
};
use super::runtime::should_run_metal_runtime;
use crate::metal_types::prelude::*;
use j2k_native::{encode, encode_htj2k, DecodeSettings, DecoderContext, EncodeOptions, Image};

// Empty packets do not depend on the code-block grid. Enlarge a tiny neutral
// image's SIZ geometry to reproduce the large-image allocation without storing
// or encoding hundreds of megabytes of pixels.
fn empty_large_rgb_codestream(ht: bool) -> Vec<u8> {
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 5,
        ..EncodeOptions::default()
    };
    let encoder = if ht { encode_htj2k } else { encode };
    let mut bytes =
        encoder(&[128; 32 * 32 * 3], 32, 32, 3, 8, false, &options).expect("encode neutral RGB8");
    let siz = bytes
        .windows(2)
        .position(|b| b == [0xff, 0x51])
        .expect("SIZ");
    for (offset, value) in [(6, 18_000_u32), (10, 12_000), (22, 18_000), (26, 12_000)] {
        bytes[siz + offset..siz + offset + 4].copy_from_slice(&value.to_be_bytes());
    }
    bytes
}

#[test]
fn large_region_coefficients_stay_within_the_requested_window() {
    use super::super::{
        crop_prepared_direct_color_plan_to_output_region, prepare_direct_color_plan,
        prepare_direct_color_plan_for_cpu_upload,
    };

    let roi = j2k_core::Rect {
        x: 8_501,
        y: 5_503,
        w: 16,
        h: 16,
    };
    for ht in [false, true] {
        let bytes = empty_large_rgb_codestream(ht);
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("large image");
        let plan = image
            .build_direct_color_plan_region_with_context(
                &mut DecoderContext::default(),
                (roi.x, roi.y, roi.w, roi.h),
            )
            .expect("large region plan");
        for prepare in [
            prepare_direct_color_plan,
            prepare_direct_color_plan_for_cpu_upload,
        ] {
            let mut prepared = prepare(&plan).expect("prepare region");
            crop_prepared_direct_color_plan_to_output_region(&mut prepared, roi)
                .expect("crop region");
            for component in &prepared.component_plans {
                let coefficients: usize = component
                    .classic_groups
                    .iter()
                    .map(|g| g.total_coefficients)
                    .chain(component.ht_groups.iter().map(|g| g.total_coefficients))
                    .sum();
                assert!(
                    coefficients < 1024 * 1024,
                    "a 16x16 ROI must not retain whole-image coefficients: {coefficients}"
                );
            }
        }
    }
}

#[test]
fn large_region_decode_matches_pixels_on_both_tier1_routes() {
    use super::super::{
        crop_prepared_direct_color_plan_to_output_region,
        execute_hybrid_cpu_tier1_direct_color_plan, execute_prepared_direct_color_plan,
        prepare_direct_color_plan, prepare_direct_color_plan_for_cpu_upload,
    };
    use j2k_core::{DeviceSurface, Downscale, PixelFormat, Rect};
    use std::sync::Arc;

    if !should_run_metal_runtime() {
        return;
    }
    for ht in [false, true] {
        let bytes = empty_large_rgb_codestream(ht);
        for scale in [Downscale::None, Downscale::Half] {
            let denominator = scale.denominator();
            let image = Image::new(
                &bytes,
                &DecodeSettings {
                    target_resolution: Some((18_000 / denominator, 12_000 / denominator)),
                    ..DecodeSettings::default()
                },
            )
            .expect("large scaled image");
            for (x, y) in [(0, 0), (8_501, 5_503), (17_983, 11_983)] {
                let roi = Rect { x, y, w: 17, h: 17 }.scaled_covering(scale);
                let plan = image
                    .build_direct_color_plan_region_with_context(
                        &mut DecoderContext::default(),
                        (roi.x, roi.y, roi.w, roi.h),
                    )
                    .expect("large region plan");
                for cpu in [false, true] {
                    let mut prepared = if cpu {
                        prepare_direct_color_plan_for_cpu_upload(&plan)
                    } else {
                        prepare_direct_color_plan(&plan)
                    }
                    .expect("prepare region");
                    crop_prepared_direct_color_plan_to_output_region(&mut prepared, roi)
                        .expect("crop region");
                    let execute = if cpu {
                        execute_hybrid_cpu_tier1_direct_color_plan
                    } else {
                        execute_prepared_direct_color_plan
                    };
                    let surface = execute(Arc::new(prepared), PixelFormat::Rgb8)
                        .expect("large region decode");
                    assert_eq!(surface.dimensions(), (roi.w, roi.h));
                    assert_eq!(
                        surface.as_bytes().expect("pixels").as_ref(),
                        vec![128; roi.w as usize * roi.h as usize * 3]
                    );
                }
            }
        }
    }
}

#[test]
fn cropped_plans_without_idwt_match_host_when_reused_concurrently() {
    use super::super::{
        crop_prepared_direct_color_plan_to_output_region,
        execute_hybrid_cpu_tier1_direct_color_plan, execute_prepared_direct_color_plan,
        prepare_direct_color_plan, prepare_direct_color_plan_for_cpu_upload,
        with_runtime_for_session,
    };
    use j2k_core::{PixelFormat, Rect};
    use std::sync::Arc;

    if !should_run_metal_runtime() {
        return;
    }
    let device = j2k_metal_support::system_default_device().expect("Metal device");
    let session = crate::MetalBackendSession::new(device);
    let roi = Rect {
        x: 67,
        y: 35,
        w: 17,
        h: 13,
    };
    for ht in [false, true] {
        let encoder = if ht { encode_htj2k } else { encode };
        let bytes = encoder(
            &j2k_test_support::patterned_rgb8(128, 96),
            128,
            96,
            3,
            8,
            false,
            &EncodeOptions {
                reversible: true,
                num_decomposition_levels: 0,
                ..EncodeOptions::default()
            },
        )
        .expect("encode without IDWT");
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let expected = image
            .decode_region((roi.x, roi.y, roi.w, roi.h))
            .expect("host region")
            .data;
        let plan = image
            .build_direct_color_plan_region_with_context(
                &mut DecoderContext::default(),
                (roi.x, roi.y, roi.w, roi.h),
            )
            .expect("region plan");
        for cpu in [false, true] {
            let prepared = if cpu {
                prepare_direct_color_plan_for_cpu_upload(&plan)
            } else {
                prepare_direct_color_plan(&plan)
            }
            .expect("prepare region");
            let execute = if cpu {
                execute_hybrid_cpu_tier1_direct_color_plan
            } else {
                execute_prepared_direct_color_plan
            };
            let prepared = Arc::new(prepared);
            // Populate the runtime's Tier-1 caches before rebasing the jobs.
            // The crop must invalidate cached inputs even when no job is removed.
            with_runtime_for_session(&session, |_| execute(prepared.clone(), PixelFormat::Rgb8))
                .expect("decode before crop");
            let mut prepared = Arc::try_unwrap(prepared).ok().expect("exclusive plan");
            crop_prepared_direct_color_plan_to_output_region(&mut prepared, roi)
                .expect("crop region");
            let prepared = Arc::new(prepared);
            std::thread::scope(|scope| {
                for _ in 0..2 {
                    let prepared = &prepared;
                    let expected = &expected;
                    let session = &session;
                    scope.spawn(move || {
                        with_runtime_for_session(session, |_| {
                            for _ in 0..2 {
                                let surface = execute(prepared.clone(), PixelFormat::Rgb8)?;
                                assert_eq!(surface.as_bytes().expect("pixels").as_ref(), expected);
                            }
                            Ok(())
                        })
                        .expect("shared-session region decode");
                    });
                }
            });
        }
    }
}

#[test]
fn cropping_a_classic_plan_drops_tier1_buffers_built_from_pruned_jobs() {
    if !should_run_metal_runtime() {
        return;
    }
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        code_block_width_exp: 0,
        code_block_height_exp: 0,
        ..EncodeOptions::default()
    };
    let pixels = j2k_test_support::gradient_u8(128, 128, 1);
    let bytes = encode(&pixels, 128, 128, 1, 8, false, &options).expect("encode classic gray8");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let plan = image
        .build_direct_grayscale_plan_with_context(&mut DecoderContext::default())
        .expect("direct grayscale plan");
    let mut prepared = prepare_direct_grayscale_plan(&plan).expect("prepared direct plan");
    let runtime = MetalRuntime::new().expect("Metal runtime");
    let outputs = |jobs: &[J2kClassicCleanupBatchJob]| -> Vec<(u32, u32)> {
        jobs.iter()
            .map(|job| (job.output_offset, job.output_stride))
            .collect()
    };
    let mut full_jobs = Vec::new();
    for step in &prepared.steps {
        if let PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) = step {
            sub_band.tier1_buffers(&runtime).expect("Tier-1 buffers");
            full_jobs.push(outputs(&sub_band.jobs));
        }
    }

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

    let sub_bands = prepared.steps.iter().filter_map(|step| match step {
        PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => Some(sub_band),
        _ => None,
    });
    let mut pruned = 0;
    // A crop drops jobs and stores the rest over their bounds, which moves
    // their outputs.
    for (sub_band, full_jobs) in sub_bands.zip(full_jobs) {
        if outputs(&sub_band.jobs) == full_jobs {
            assert!(sub_band.tier1_inputs.buffers.get().is_some());
            continue;
        }
        pruned += 1;
        assert!(
            sub_band.tier1_inputs.buffers.get().is_none(),
            "buffers built from the uncropped jobs must not survive the crop"
        );
        let rebuilt = sub_band
            .tier1_buffers(&runtime)
            .expect("rebuilt Tier-1 buffers");
        assert_eq!(
            rebuilt.jobs.length(),
            size_of_val(sub_band.jobs.as_slice()).max(1)
        );
    }
    assert!(pruned > 0, "fixture must prune classic code-block jobs");
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

fn prepared_direct_grayscale_ht_job_count(plan: &PreparedDirectGrayscalePlan) -> usize {
    plan.steps
        .iter()
        .map(|step| match step {
            PreparedDirectGrayscaleStep::HtSubBand(sub_band) => sub_band.jobs.len(),
            _ => 0,
        })
        .sum()
}

fn prepared_direct_grayscale_ht_coded_byte_count(plan: &PreparedDirectGrayscalePlan) -> usize {
    plan.steps
        .iter()
        .map(|step| match step {
            PreparedDirectGrayscaleStep::HtSubBand(sub_band) => match &sub_band.payload_source {
                PreparedHtPayloadSource::Contiguous(data) => data.len(),
                PreparedHtPayloadSource::Referenced { .. } => 0,
            },
            _ => 0,
        })
        .sum()
}

fn prepared_direct_grayscale_idwt_output_sample_count(plan: &PreparedDirectGrayscalePlan) -> usize {
    plan.steps
        .iter()
        .map(|step| match step {
            PreparedDirectGrayscaleStep::Idwt(idwt) => prepared_idwt_output_len(idwt)
                .expect("prepared IDWT output dimensions must fit Metal f32 storage"),
            _ => 0,
        })
        .sum()
}

fn prepared_direct_grayscale_idwt_full_and_prepared_lens(
    plan: &PreparedDirectGrayscalePlan,
) -> Vec<(usize, usize)> {
    plan.steps
        .iter()
        .filter_map(|step| match step {
            PreparedDirectGrayscaleStep::Idwt(idwt) => Some((
                idwt.step.rect.width() as usize * idwt.step.rect.height() as usize,
                prepared_idwt_output_len(idwt)
                    .expect("prepared IDWT output dimensions must fit Metal f32 storage"),
            )),
            _ => None,
        })
        .collect()
}
