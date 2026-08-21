// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::direct_roi::prepared_idwt_output_len;
use super::super::{
    crop_prepared_direct_grayscale_plan_to_output_region, prepare_direct_grayscale_plan,
    PreparedDirectGrayscalePlan, PreparedDirectGrayscaleStep, PreparedHtPayloadSource,
};
use j2k_native::{encode_htj2k, DecodeSettings, DecoderContext, EncodeOptions, Image};

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
