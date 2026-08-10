// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

fn fixture_openhtj2k_ht_refinement() -> &'static [u8] {
    include_bytes!("../../fixtures/htj2k/openhtj2k_ds0_ht_12_b11.j2k")
}

fn fixture_openhtj2k_ht_refinement_pixels() -> &'static [u8] {
    include_bytes!("../../fixtures/htj2k/openhtj2k_ds0_ht_12_b11.gray")
}

fn fixture_openhtj2k_ht_refinement_odd() -> &'static [u8] {
    include_bytes!("../../fixtures/htj2k/openhtj2k_ds0_ht_09_b11.j2k")
}

fn fixture_openhtj2k_ht_refinement_odd_pixels() -> &'static [u8] {
    include_bytes!("../../fixtures/htj2k/openhtj2k_ds0_ht_09_b11.gray")
}

#[derive(Default)]
struct CapturingHtDecoder {
    called: bool,
    blocks: usize,
    refinement_jobs: usize,
    max_coding_passes: u8,
}

impl HtCodeBlockDecoder for CapturingHtDecoder {
    fn decode_code_block(
        &mut self,
        job: HtCodeBlockDecodeJob<'_>,
        output: &mut [f32],
    ) -> Result<()> {
        self.decode_code_block_with_midpoint(job, output, false)
    }

    fn decode_code_block_with_midpoint(
        &mut self,
        job: HtCodeBlockDecodeJob<'_>,
        output: &mut [f32],
        irreversible_midpoint: bool,
    ) -> Result<()> {
        self.called = true;
        self.blocks += 1;
        self.max_coding_passes = self.max_coding_passes.max(job.number_of_coding_passes);
        if job.refinement_length > 0 {
            self.refinement_jobs += 1;
            assert!(
                job.number_of_coding_passes > 1,
                "refinement bytes must correspond to refinement coding passes"
            );
        }

        if irreversible_midpoint {
            decode_ht_code_block_scalar_with_workspace_midpoint(
                job,
                output,
                &mut HtCodeBlockDecodeWorkspace::default(),
            )
        } else {
            decode_ht_code_block_scalar(job, output)
        }
    }
}

#[derive(Clone)]
struct CapturedHtDecodeJob {
    data: Vec<u8>,
    cleanup_length: u32,
    refinement_length: u32,
    width: u32,
    height: u32,
    output_stride: usize,
    missing_bit_planes: u8,
    number_of_coding_passes: u8,
    num_bitplanes: u8,
    roi_shift: u8,
    stripe_causal: bool,
    strict: bool,
    dequantization_step: f32,
}

impl CapturedHtDecodeJob {
    fn from_job(job: HtCodeBlockDecodeJob<'_>) -> Self {
        Self {
            data: job.data.to_vec(),
            cleanup_length: job.cleanup_length,
            refinement_length: job.refinement_length,
            width: job.width,
            height: job.height,
            output_stride: job.output_stride,
            missing_bit_planes: job.missing_bit_planes,
            number_of_coding_passes: job.number_of_coding_passes,
            num_bitplanes: job.num_bitplanes,
            roi_shift: job.roi_shift,
            stripe_causal: job.stripe_causal,
            strict: job.strict,
            dequantization_step: job.dequantization_step,
        }
    }

    fn borrowed(&self) -> HtCodeBlockDecodeJob<'_> {
        HtCodeBlockDecodeJob {
            data: &self.data,
            cleanup_length: self.cleanup_length,
            refinement_length: self.refinement_length,
            width: self.width,
            height: self.height,
            output_stride: self.output_stride,
            missing_bit_planes: self.missing_bit_planes,
            number_of_coding_passes: self.number_of_coding_passes,
            num_bitplanes: self.num_bitplanes,
            roi_shift: self.roi_shift,
            stripe_causal: self.stripe_causal,
            strict: self.strict,
            dequantization_step: self.dequantization_step,
        }
    }
}

#[derive(Default)]
struct FirstHtJobDecoder {
    job: Option<CapturedHtDecodeJob>,
}

impl HtCodeBlockDecoder for FirstHtJobDecoder {
    fn decode_code_block_with_midpoint(
        &mut self,
        job: HtCodeBlockDecodeJob<'_>,
        output: &mut [f32],
        irreversible_midpoint: bool,
    ) -> Result<()> {
        if self.job.is_none() {
            self.job = Some(CapturedHtDecodeJob::from_job(job));
        }
        if irreversible_midpoint {
            decode_ht_code_block_scalar_with_workspace_midpoint(
                job,
                output,
                &mut HtCodeBlockDecodeWorkspace::default(),
            )
        } else {
            decode_ht_code_block_scalar(job, output)
        }
    }
}

struct ZeroRefinementHtDecoder;

impl HtCodeBlockDecoder for ZeroRefinementHtDecoder {
    fn decode_code_block(
        &mut self,
        job: HtCodeBlockDecodeJob<'_>,
        output: &mut [f32],
    ) -> Result<()> {
        self.decode_code_block_with_midpoint(job, output, false)
    }

    fn decode_code_block_with_midpoint(
        &mut self,
        job: HtCodeBlockDecodeJob<'_>,
        output: &mut [f32],
        irreversible_midpoint: bool,
    ) -> Result<()> {
        let mut data = job.data.to_vec();
        let cleanup_len = job.cleanup_length as usize;
        let refinement_len = job.refinement_length as usize;
        data[cleanup_len..cleanup_len + refinement_len].fill(0);
        let zeroed = HtCodeBlockDecodeJob { data: &data, ..job };

        if irreversible_midpoint {
            decode_ht_code_block_scalar_with_workspace_midpoint(
                zeroed,
                output,
                &mut HtCodeBlockDecodeWorkspace::default(),
            )
        } else {
            decode_ht_code_block_scalar(zeroed, output)
        }
    }
}

#[derive(Default)]
struct CleanupLimitedHtDecoder {
    blocks: usize,
    refinement_blocks: usize,
    cleanup_bytes: usize,
    refinement_bytes: usize,
}

impl HtCodeBlockDecoder for CleanupLimitedHtDecoder {
    fn decode_code_block_with_midpoint(
        &mut self,
        job: HtCodeBlockDecodeJob<'_>,
        output: &mut [f32],
        irreversible_midpoint: bool,
    ) -> Result<()> {
        self.blocks += 1;
        self.cleanup_bytes += job.cleanup_length as usize;
        if job.refinement_length > 0 {
            self.refinement_blocks += 1;
            self.refinement_bytes += job.refinement_length as usize;
        }

        if irreversible_midpoint {
            let cleanup_only = HtCodeBlockDecodeJob {
                refinement_length: 0,
                number_of_coding_passes: 1,
                ..job
            };
            decode_ht_code_block_scalar_with_workspace_midpoint(
                cleanup_only,
                output,
                &mut HtCodeBlockDecodeWorkspace::default(),
            )
        } else {
            decode_ht_code_block_scalar_until_phase(
                job,
                output,
                HtCodeBlockDecodePhaseLimit::Cleanup,
            )
        }
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "fixture samples are rounded and clamped to the full u8 range before conversion"
)]
fn rounded_u8(sample: f32) -> u8 {
    sample.round().clamp(0.0, 255.0) as u8
}

#[test]
fn irreversible_htj2k_direct_plan_retains_fixed_point_reconstruction() {
    let pixels = gradient_pixels(32, 32, 1);
    let bytes = encode_htj2k(
        &pixels,
        32,
        32,
        1,
        8,
        false,
        &EncodeOptions {
            reversible: false,
            num_decomposition_levels: 3,
            ..EncodeOptions::default()
        },
    )
    .expect("encode irreversible HTJ2K");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();

    let plan = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("build irreversible direct plan");
    let ht_sub_bands = plan.steps.iter().filter_map(|step| match step {
        J2kDirectGrayscaleStep::HtSubBand(plan) if !plan.jobs.is_empty() => Some(plan),
        _ => None,
    });
    let mut count = 0;
    for sub_band in ht_sub_bands {
        count += 1;
        assert!(sub_band.irreversible_midpoint);
    }
    assert!(count > 0, "fixture must retain HT sub-band jobs");
}

#[test]
fn openhtj2k_conformance_fixture_exercises_refinement_passes() {
    for fixture in [
        (
            "ds0_ht_12_b11",
            fixture_openhtj2k_ht_refinement(),
            fixture_openhtj2k_ht_refinement_pixels(),
            (3, 5),
            8,
            2,
            4,
        ),
        (
            "ds0_ht_09_b11",
            fixture_openhtj2k_ht_refinement_odd(),
            fixture_openhtj2k_ht_refinement_odd_pixels(),
            (17, 37),
            14,
            14,
            629,
        ),
    ] {
        let (name, codestream, expected_pixels, dimensions, blocks, refinement_jobs, zero_diffs) =
            fixture;
        let image = Image::new(codestream, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let mut hook = CapturingHtDecoder::default();

        let components = image
            .decode_components_with_ht_decoder(&mut context, &mut hook)
            .expect("decode OpenHTJ2K HTJ2K fixture");

        assert!(
            hook.called,
            "{name}: HTJ2K fixture must use HT code-block decode"
        );
        assert!(
            hook.refinement_jobs > 0,
            "{name}: OpenHTJ2K fixture must contain non-empty refinement segments"
        );
        assert!(
            hook.max_coding_passes > 1,
            "{name}: OpenHTJ2K fixture must exercise more than the cleanup pass"
        );
        assert_eq!(hook.blocks, blocks, "{name}: HT code-block count");
        assert_eq!(
            hook.refinement_jobs, refinement_jobs,
            "{name}: refinement job count"
        );
        assert_eq!(hook.max_coding_passes, 3, "{name}: max HT coding passes");
        assert_eq!(components.dimensions(), dimensions, "{name}: dimensions");
        assert_eq!(components.planes().len(), 1, "{name}: component planes");

        let decoded: Vec<u8> = components.planes()[0]
            .samples()
            .iter()
            .copied()
            .map(rounded_u8)
            .collect();
        assert_eq!(decoded, expected_pixels, "{name}: decoded pixels");

        let mut zero_context = DecoderContext::default();
        let mut zero_hook = ZeroRefinementHtDecoder;
        let zeroed_components = image
            .decode_components_with_ht_decoder(&mut zero_context, &mut zero_hook)
            .expect("decode OpenHTJ2K fixture with zeroed refinement bytes");
        let actual_zero_diffs = components.planes()[0]
            .samples()
            .iter()
            .zip(zeroed_components.planes()[0].samples())
            .filter(|(actual, zeroed)| (*actual - *zeroed).abs() > f32::EPSILON)
            .count();
        assert_eq!(
            actual_zero_diffs, zero_diffs,
            "{name}: zeroing refinement bytes must change decoded samples"
        );
    }
}

#[test]
fn openhtj2k_refinement_phase_limited_decode_differs_and_records_ht_stats() {
    let image = Image::new(
        fixture_openhtj2k_ht_refinement_odd(),
        &DecodeSettings::default(),
    )
    .expect("image");
    let mut full_context = DecoderContext::default();

    let (full_samples, full_decoded) = {
        let full_components = image
            .decode_components_with_context(&mut full_context)
            .expect("full native decode of OpenHTJ2K refinement fixture");
        let full_samples = full_components.planes()[0].samples().to_vec();
        let full_decoded: Vec<u8> = full_samples.iter().copied().map(rounded_u8).collect();
        (full_samples, full_decoded)
    };
    assert_eq!(
        full_decoded,
        fixture_openhtj2k_ht_refinement_odd_pixels(),
        "full decode must match the checked-in OpenHTJ2K oracle"
    );

    let stats = full_context
        .tile_decode_context
        .debug_counters
        .ht_phase_stats;
    assert_eq!(stats.blocks, 14, "HT block count");
    assert_eq!(stats.refinement_blocks, 14, "HT refinement block count");
    assert!(stats.cleanup_bytes > 0, "cleanup byte total");
    assert!(stats.refinement_bytes > 0, "refinement byte total");

    let mut cleanup_context = DecoderContext::default();
    let mut cleanup_hook = CleanupLimitedHtDecoder::default();
    let cleanup_components = image
        .decode_components_with_ht_decoder(&mut cleanup_context, &mut cleanup_hook)
        .expect("cleanup-limited decode of OpenHTJ2K refinement fixture");
    let cleanup_decoded: Vec<u8> = cleanup_components.planes()[0]
        .samples()
        .iter()
        .copied()
        .map(rounded_u8)
        .collect();
    let cleanup_sample_diffs = full_samples
        .iter()
        .zip(cleanup_components.planes()[0].samples())
        .filter(|(full, cleanup)| (*full - *cleanup).abs() > f32::EPSILON)
        .count();

    assert!(
        cleanup_sample_diffs > 0,
        "cleanup-limited decode must omit refinement effects"
    );
    assert_eq!(
        cleanup_decoded, full_decoded,
        "fixture refinement differences are below final u8 clamping"
    );
    assert_eq!(cleanup_hook.blocks, 14, "hook HT block count");
    assert_eq!(
        cleanup_hook.refinement_blocks, 14,
        "hook HT refinement block count"
    );
    assert!(cleanup_hook.cleanup_bytes > 0, "hook cleanup byte total");
    assert!(
        cleanup_hook.refinement_bytes > 0,
        "hook refinement byte total"
    );
}

#[test]
fn scalar_htj2k_decode_workspace_matches_fresh_decode_and_reuses_capacity() {
    let image = Image::new(
        fixture_openhtj2k_ht_refinement_odd(),
        &DecodeSettings::default(),
    )
    .expect("image");
    let mut context = DecoderContext::default();
    let mut hook = FirstHtJobDecoder::default();
    image
        .decode_components_with_ht_decoder(&mut context, &mut hook)
        .expect("decode fixture while collecting HT jobs");
    let job = hook
        .job
        .as_ref()
        .expect("fixture must expose an HT decode job")
        .borrowed();
    let mut fresh = vec![0.0_f32; job.width as usize * job.height as usize];
    let mut reused = vec![0.0_f32; fresh.len()];
    let mut profiled = vec![0.0_f32; fresh.len()];
    let mut workspace = HtCodeBlockDecodeWorkspace::default();
    let mut profile = HtCodeBlockDecodeProfile::default();

    decode_ht_code_block_scalar(job, &mut fresh).expect("fresh HT decode");
    decode_ht_code_block_scalar_with_workspace(job, &mut reused, &mut workspace)
        .expect("workspace HT decode");
    let first_capacity = workspace.coefficient_capacity();
    decode_ht_code_block_scalar_with_workspace(job, &mut reused, &mut workspace)
        .expect("second workspace HT decode");
    decode_ht_code_block_scalar_with_workspace_profiled(
        job,
        &mut profiled,
        &mut workspace,
        &mut profile,
    )
    .expect("profiled workspace HT decode");

    assert_eq!(reused, fresh);
    assert_eq!(profiled, fresh);
    assert!(first_capacity >= fresh.len());
    assert_eq!(workspace.coefficient_capacity(), first_capacity);
    assert_eq!(profile.blocks, 1);
    assert!(profile.cleanup_bytes > 0);
}
