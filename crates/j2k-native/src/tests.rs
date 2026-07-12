// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::j2c::ComponentData;

#[test]
fn direct_grayscale_plan_rejects_rgb_image_with_typed_reason() {
    let pixels = vec![0, 16, 32, 64, 80, 96, 128, 144, 160, 192, 208, 224];
    let bytes =
        encode(&pixels, 2, 2, 3, 8, false, &EncodeOptions::default()).expect("encode rgb j2k");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();

    let error = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect_err("rgb image should not build a grayscale direct plan");

    assert_eq!(
        error,
        DecodeError::Decoding(DecodingError::DirectPlanUnsupported(
            DirectPlanUnsupportedReason::GrayscaleImageWithoutAlpha
        ))
    );
}

#[test]
fn ht_uvlc_encode_table_bytes_match_entry_packing_order() {
    let entries = ht_uvlc_encode_table();
    let bytes = ht_uvlc_encode_table_bytes();

    assert_eq!(bytes.len(), entries.len() * 6);
    for (index, entry) in entries.iter().enumerate() {
        let offset = index * 6;
        assert_eq!(
            &bytes[offset..offset + 6],
            &[
                entry.pre,
                entry.pre_len,
                entry.suf,
                entry.suf_len,
                entry.ext,
                entry.ext_len
            ],
        );
    }
}

#[test]
fn roi_maxshift_inverse_preserves_background_and_unshifts_roi_coefficients() {
    assert_eq!(apply_roi_maxshift_inverse_i32(127, 7), 127);
    assert_eq!(apply_roi_maxshift_inverse_i32(-127, 7), -127);
    assert_eq!(apply_roi_maxshift_inverse_i32(128, 7), 1);
    assert_eq!(apply_roi_maxshift_inverse_i32(-128, 7), -1);
    assert_eq!(apply_roi_maxshift_inverse_i32(255, 7), 1);
    assert_eq!(apply_roi_maxshift_inverse_i32(-255, 7), -1);
    assert_eq!(apply_roi_maxshift_inverse_i32(256, 7), 2);
    assert_eq!(apply_roi_maxshift_inverse_i32(-256, 7), -2);
    assert_eq!(apply_roi_maxshift_inverse_i32(42, 0), 42);
    assert_eq!(apply_roi_maxshift_inverse_i64(1_i64 << 38, 7), 1_i64 << 31);
    assert_eq!(
        apply_roi_maxshift_inverse_i64(-(1_i64 << 38), 7),
        -(1_i64 << 31)
    );
}

#[test]
fn cielab_conversion_uses_b_range_independently_from_a_range() {
    let mut components = vec![
        lab_component(vec![0.0]),
        lab_component(vec![0.0]),
        lab_component(vec![255.0]),
    ];
    let lab = CieLab {
        rl: Some(100),
        ol: Some(0),
        ra: Some(100),
        oa: Some(0),
        rb: Some(220),
        ob: Some(0),
    };

    dispatch!(Level::new(), simd => {
        cielab_to_rgb(simd, &mut components, 8, &lab)
    })
    .expect("CIELab conversion succeeds");

    let b = components[2].container.truncated()[0];
    assert!(
        (b - 348.0).abs() < 0.001,
        "b channel must use rb=220, not ra=100; got {b}"
    );
}

#[test]
fn cielab_conversion_honors_b_range_when_a_range_is_missing() {
    let mut components = vec![
        lab_component(vec![0.0]),
        lab_component(vec![0.0]),
        lab_component(vec![255.0]),
    ];
    let lab = CieLab {
        rl: Some(100),
        ol: Some(0),
        ra: None,
        oa: Some(0),
        rb: Some(220),
        ob: Some(0),
    };

    dispatch!(Level::new(), simd => {
        cielab_to_rgb(simd, &mut components, 8, &lab)
    })
    .expect("CIELab conversion succeeds");

    let b = components[2].container.truncated()[0];
    assert!(
        (b - 348.0).abs() < 0.001,
        "b channel must use explicit rb even when ra is absent; got {b}"
    );
}

fn lab_component(samples: Vec<f32>) -> ComponentData {
    ComponentData {
        container: math::SimdBuffer::new(samples),
        integer_container: None,
        bit_depth: 8,
        signed: false,
    }
}

#[test]
fn classic_decode_adapter_accepts_legal_38_bit_roi_bitplane_count() {
    assert_eq!(
        add_roi_shift_to_bitplanes(38, 0, MAX_CLASSIC_DECODE_BITPLANES).unwrap(),
        38
    );
    assert_eq!(
        add_roi_shift_to_bitplanes(37, 1, MAX_CLASSIC_DECODE_BITPLANES).unwrap(),
        38
    );
}

#[test]
#[expect(
    clippy::float_cmp,
    reason = "reversible ROI decoding must preserve exact integer-valued f32 coefficients"
)]
fn classic_scalar_decode_applies_nonzero_roi_maxshift() {
    let roi_shift = 3;
    let total_bitplanes = 3;
    let style = J2kCodeBlockStyle {
        selective_arithmetic_coding_bypass: false,
        reset_context_probabilities: false,
        termination_on_each_pass: false,
        vertically_causal_context: false,
        segmentation_symbols: false,
    };
    let coded_coefficients = [0, 5, 1 << roi_shift, -(2 << roi_shift)];
    let encoded = encode_j2k_code_block_scalar_with_style(
        &coded_coefficients,
        2,
        2,
        J2kSubBandType::LowLow,
        total_bitplanes + roi_shift,
        style,
    )
    .expect("encode ROI-shifted code block");
    let job = J2kCodeBlockDecodeJob {
        data: &encoded.data,
        segments: &encoded.segments,
        width: 2,
        height: 2,
        output_stride: 2,
        missing_bit_planes: encoded.missing_bit_planes,
        number_of_coding_passes: encoded.number_of_coding_passes,
        total_bitplanes,
        roi_shift,
        sub_band_type: J2kSubBandType::LowLow,
        style,
        strict: true,
        dequantization_step: 1.0,
    };
    let mut output = [0.0; 4];

    decode_j2k_code_block_scalar(job, &mut output).expect("decode ROI-shifted code block");

    assert_eq!(output, [0.0, 5.0, 1.0, -2.0]);
}

#[test]
fn classic_scalar_token_pack_matches_scalar_single_cleanup_block() {
    let style = J2kCodeBlockStyle {
        selective_arithmetic_coding_bypass: true,
        reset_context_probabilities: false,
        termination_on_each_pass: false,
        vertically_causal_context: false,
        segmentation_symbols: false,
    };
    let scalar =
        encode_j2k_code_block_scalar_with_style(&[1], 1, 1, J2kSubBandType::LowLow, 1, style)
            .expect("encode scalar");
    let token_bytes = pack_mq_test_tokens(&[(0, 1), (9, 0)]);
    let packed = pack_j2k_code_block_scalar_from_tier1_tokens(
        &token_bytes,
        &[J2kTier1TokenSegment {
            token_bit_offset: 0,
            token_bit_count: 12,
            start_coding_pass: 0,
            end_coding_pass: 1,
            use_arithmetic: true,
        }],
        scalar.number_of_coding_passes,
        scalar.missing_bit_planes,
    )
    .expect("pack tokens");

    assert_eq!(packed.data, scalar.data);
    assert_eq!(packed.segments, scalar.segments);
    assert_eq!(
        packed.number_of_coding_passes,
        scalar.number_of_coding_passes
    );
    assert_eq!(packed.missing_bit_planes, scalar.missing_bit_planes);
}

#[test]
fn scalar_encode_adapters_preserve_typed_input_and_cap_categories() {
    let style = J2kCodeBlockStyle {
        selective_arithmetic_coding_bypass: false,
        reset_context_probabilities: false,
        termination_on_each_pass: false,
        vertically_causal_context: false,
        segmentation_symbols: false,
    };
    assert!(matches!(
        encode_j2k_code_block_scalar_with_style(&[1], 2, 2, J2kSubBandType::LowLow, 1, style,),
        Err(EncodeError::InvalidInput {
            what: "contiguous coefficient block length mismatch",
        })
    ));
    assert!(matches!(
        encode_ht_code_block_scalar(&[0], 1, 1, 0),
        Err(EncodeError::InvalidInput {
            what: "HTJ2K scalar encoder currently supports 1..=31 bitplanes",
        })
    ));
    assert!(matches!(
        encode_ht_code_block_scalar_with_passes(&[0], 1, 1, 1, 4),
        Err(EncodeError::InvalidInput {
            what: "HTJ2K scalar encoder currently supports cleanup, sigprop, and one magref refinement pass",
        })
    ));

    let error = pack_j2k_code_block_scalar_from_tier1_tokens(
        &[],
        &[J2kTier1TokenSegment {
            token_bit_offset: 0,
            token_bit_count: u32::MAX - 3,
            start_coding_pass: 0,
            end_coding_pass: 1,
            use_arithmetic: true,
        }],
        1,
        0,
    )
    .expect_err("oversized token payload must fail before reading token bytes");
    assert!(matches!(
        error,
        EncodeError::AllocationTooLarge {
            what: "classic Tier-1 token worker allocation",
            requested,
            cap: DEFAULT_MAX_CODEC_BYTES,
        } if requested > DEFAULT_MAX_CODEC_BYTES
    ));
}

fn pack_mq_test_tokens(tokens: &[(u8, u8)]) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut current = 0u8;
    let mut bits = 0u8;
    for &(ctx, bit) in tokens {
        let value = (ctx & 0x1F) | ((bit & 1) << 5);
        for shift in (0..6).rev() {
            current = (current << 1) | ((value >> shift) & 1);
            bits += 1;
            if bits == 8 {
                bytes.push(current);
                current = 0;
                bits = 0;
            }
        }
    }
    if bits != 0 {
        bytes.push(current << (8 - bits));
    }
    bytes
}

#[test]
fn classic_scalar_profiled_decode_matches_unprofiled_decode() {
    let width = 64u32;
    let height = 64u32;
    let sample_count = width as usize * height as usize;
    let total_bitplanes = 12;
    let style = default_classic_test_style();
    let coefficients = (0..sample_count)
        .map(|idx| {
            let value = i32::try_from((idx * 37) % 4095).expect("sample value fits i32") - 2048;
            if idx % 17 == 0 {
                0
            } else {
                value
            }
        })
        .collect::<Vec<_>>();
    let encoded = encode_j2k_code_block_scalar_with_style(
        &coefficients,
        width,
        height,
        J2kSubBandType::LowLow,
        total_bitplanes,
        style,
    )
    .expect("encode classic block");
    let job = classic_low_low_decode_job(&encoded, width, height, total_bitplanes, style);
    let mut expected = vec![0.0_f32; sample_count];
    let mut actual = vec![0.0_f32; sample_count];
    let mut profile = J2kCodeBlockDecodeProfile::default();

    decode_j2k_code_block_scalar(job, &mut expected).expect("unprofiled classic decode");
    decode_j2k_code_block_scalar_profiled(job, &mut actual, &mut profile)
        .expect("profiled classic decode");

    assert_eq!(actual, expected);
    #[cfg(feature = "std")]
    assert!(profile.cleanup_us > 0);
    #[cfg(not(feature = "std"))]
    assert_eq!(profile.cleanup_us, 0);
}

#[test]
fn classic_scalar_workspace_reuse_matches_fresh_decode() {
    let total_bitplanes = 6;
    let style = default_classic_test_style();
    let mut workspace = J2kCodeBlockDecodeWorkspace::default();

    for (width, height, seed) in [(8, 8, 0x31), (4, 16, 0x47)] {
        let coefficients = (0..width * height)
            .map(|idx| {
                let idx = i32::try_from(idx).expect("small test index fits i32");
                let value = ((idx * seed) % 23) - 11;
                if idx % 7 == 0 {
                    0
                } else {
                    value
                }
            })
            .collect::<Vec<_>>();
        let encoded = encode_j2k_code_block_scalar_with_style(
            &coefficients,
            width,
            height,
            J2kSubBandType::LowLow,
            total_bitplanes,
            style,
        )
        .expect("encode classic block");
        let job = classic_low_low_decode_job(&encoded, width, height, total_bitplanes, style);
        let mut fresh = vec![0.0_f32; width as usize * height as usize];
        let mut reused = vec![0.0_f32; width as usize * height as usize];

        decode_j2k_code_block_scalar(job, &mut fresh).expect("fresh classic decode");
        decode_j2k_code_block_scalar_with_workspace(job, &mut reused, &mut workspace)
            .expect("workspace classic decode");

        assert_eq!(reused, fresh);
    }
}

fn default_classic_test_style() -> J2kCodeBlockStyle {
    J2kCodeBlockStyle {
        selective_arithmetic_coding_bypass: false,
        reset_context_probabilities: false,
        termination_on_each_pass: false,
        vertically_causal_context: false,
        segmentation_symbols: false,
    }
}

fn classic_low_low_decode_job(
    encoded: &EncodedJ2kCodeBlock,
    width: u32,
    height: u32,
    total_bitplanes: u8,
    style: J2kCodeBlockStyle,
) -> J2kCodeBlockDecodeJob<'_> {
    J2kCodeBlockDecodeJob {
        data: &encoded.data,
        segments: &encoded.segments,
        width,
        height,
        output_stride: width as usize,
        missing_bit_planes: encoded.missing_bit_planes,
        number_of_coding_passes: encoded.number_of_coding_passes,
        total_bitplanes,
        roi_shift: 0,
        sub_band_type: J2kSubBandType::LowLow,
        style,
        strict: true,
        dequantization_step: 1.0,
    }
}

#[test]
fn scalar_packetization_rejects_overflowing_ht_refinement_lengths_without_panic() {
    let payload = [0x12];
    let block = J2kPacketizationCodeBlock {
        data: &payload,
        ht_cleanup_length: u32::MAX,
        ht_refinement_length: 1,
        num_coding_passes: 3,
        num_zero_bitplanes: 2,
        previously_included: false,
        l_block: 3,
        block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
    };
    let subband = J2kPacketizationSubband {
        code_blocks: vec![block],
        num_cbs_x: 1,
        num_cbs_y: 1,
    };
    let resolution = J2kPacketizationResolution {
        subbands: vec![subband],
    };
    let resolutions = [resolution];
    let job = J2kPacketizationEncodeJob {
        resolution_count: 1,
        num_layers: 1,
        num_components: 1,
        code_block_count: 1,
        progression_order: J2kPacketizationProgressionOrder::Lrcp,
        packet_descriptors: &[],
        resolutions: &resolutions,
    };

    let err = encode_j2k_packetization_scalar(job)
        .expect_err("overflowing HT packetization segment lengths rejected");

    assert_eq!(
        err,
        EncodeError::ArithmeticOverflow {
            what: "multi-pass HTJ2K packet contribution length overflow",
        }
    );
}

#[derive(Default)]
struct DecodeWorkCounter {
    classic_code_blocks: usize,
    ht_code_blocks: usize,
    idwt_output_samples: usize,
}

impl DecodeWorkCounter {
    fn code_blocks(&self) -> usize {
        self.classic_code_blocks + self.ht_code_blocks
    }
}

struct FailingHtDecoder {
    called: bool,
}

impl HtCodeBlockDecoder for FailingHtDecoder {
    fn decode_code_block(
        &mut self,
        _job: HtCodeBlockDecodeJob<'_>,
        _output: &mut [f32],
    ) -> Result<()> {
        self.called = true;
        Err(DecodingError::CodeBlockDecodeFailure.into())
    }
}

struct FailingClassicDecoder {
    called: bool,
}

impl HtCodeBlockDecoder for FailingClassicDecoder {
    fn decode_code_block(
        &mut self,
        _job: HtCodeBlockDecodeJob<'_>,
        _output: &mut [f32],
    ) -> Result<()> {
        panic!("HT hook must not be used for classic J2K test")
    }

    fn decode_j2k_code_block(
        &mut self,
        _job: J2kCodeBlockDecodeJob<'_>,
        _output: &mut [f32],
    ) -> Result<bool> {
        self.called = true;
        Err(DecodingError::CodeBlockDecodeFailure.into())
    }
}

struct FailingClassicBatchDecoder {
    called: bool,
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

        decode_ht_code_block_scalar(job, output)
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
    fn decode_code_block(
        &mut self,
        job: HtCodeBlockDecodeJob<'_>,
        output: &mut [f32],
    ) -> Result<()> {
        if self.job.is_none() {
            self.job = Some(CapturedHtDecodeJob::from_job(job));
        }
        decode_ht_code_block_scalar(job, output)
    }
}

struct ZeroRefinementHtDecoder;

impl HtCodeBlockDecoder for ZeroRefinementHtDecoder {
    fn decode_code_block(
        &mut self,
        job: HtCodeBlockDecodeJob<'_>,
        output: &mut [f32],
    ) -> Result<()> {
        let mut data = job.data.to_vec();
        let cleanup_len = job.cleanup_length as usize;
        let refinement_len = job.refinement_length as usize;
        data[cleanup_len..cleanup_len + refinement_len].fill(0);
        let zeroed = HtCodeBlockDecodeJob { data: &data, ..job };

        decode_ht_code_block_scalar(zeroed, output)
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
    fn decode_code_block(
        &mut self,
        job: HtCodeBlockDecodeJob<'_>,
        output: &mut [f32],
    ) -> Result<()> {
        self.blocks += 1;
        self.cleanup_bytes += job.cleanup_length as usize;
        if job.refinement_length > 0 {
            self.refinement_blocks += 1;
            self.refinement_bytes += job.refinement_length as usize;
        }

        decode_ht_code_block_scalar_until_phase(job, output, HtCodeBlockDecodePhaseLimit::Cleanup)
    }
}

impl HtCodeBlockDecoder for FailingClassicBatchDecoder {
    fn decode_code_block(
        &mut self,
        _job: HtCodeBlockDecodeJob<'_>,
        _output: &mut [f32],
    ) -> Result<()> {
        panic!("HT hook must not be used for classic J2K batch test")
    }

    fn decode_j2k_code_block(
        &mut self,
        _job: J2kCodeBlockDecodeJob<'_>,
        _output: &mut [f32],
    ) -> Result<bool> {
        panic!("per-block classic hook must not be used when the batch hook handles the sub-band")
    }

    fn decode_j2k_sub_band(
        &mut self,
        _job: J2kSubBandDecodeJob<'_>,
        _output: &mut [f32],
    ) -> Result<bool> {
        self.called = true;
        Err(DecodingError::CodeBlockDecodeFailure.into())
    }
}

fn fixture() -> Vec<u8> {
    let pixels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(&pixels, 2, 2, 3, 8, false, &options).expect("encode")
}

#[test]
fn decode_into_rejects_short_output_buffer() {
    let bytes = fixture();
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();
    let mut output = vec![0; 11];

    let err = image
        .decode_into(&mut output, &mut context)
        .expect_err("short output buffer must be rejected");

    assert_eq!(
        err,
        DecodeError::Decoding(DecodingError::OutputBufferTooSmall)
    );
}

fn fixture_multi_block() -> Vec<u8> {
    let pixels: Vec<u8> = (0..64).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 0,
        code_block_width_exp: 0,
        code_block_height_exp: 0,
        ..EncodeOptions::default()
    };
    encode(&pixels, 8, 8, 1, 8, false, &options).expect("encode multi-block classic")
}

fn fixture_gray() -> Vec<u8> {
    let pixels: Vec<u8> = (0..16).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(&pixels, 4, 4, 1, 8, false, &options).expect("encode classic gray8")
}

#[test]
fn reversible_coefficient_handoff_is_complete_and_releases_decode_storage() {
    let bytes = fixture_gray();
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();

    let coefficients = image
        .decode_reversible_53_coefficients_with_context(&mut context)
        .expect("extract reversible coefficients");

    assert_eq!(coefficients.image.components.len(), 1);
    let component = &coefficients.image.components[0].dwt;
    let coefficient_count = component.ll.len()
        + component
            .levels
            .iter()
            .map(|level| level.hl.len() + level.lh.len() + level.hh.len())
            .sum::<usize>();
    assert_eq!(coefficient_count, 16);
    assert_eq!(context.storage.retained_capacity_bytes().unwrap(), 0);
    assert_eq!(
        context.tile_decode_context.tier1_capacity_bytes().unwrap(),
        0
    );
}

#[test]
fn repeated_decode_and_recode_reuse_immutable_tile_part_metadata() {
    let bytes = fixture_gray();
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();
    let ht_options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        use_ht_block_coding: true,
        ..EncodeOptions::default()
    };

    let initial_pixels = image
        .decode_native_with_context(&mut context)
        .expect("first decode");
    let initial_coefficients = image
        .decode_reversible_53_coefficients_with_context(&mut context)
        .expect("first coefficient extraction");
    let initial_codestream = initial_coefficients
        .encode_htj2k(&ht_options)
        .expect("first coefficient recode");

    let repeated_pixels = image
        .decode_native_with_context(&mut context)
        .expect("second decode");
    let repeated_coefficients = image
        .decode_reversible_53_coefficients_with_context(&mut context)
        .expect("second coefficient extraction");
    let repeated_codestream = repeated_coefficients
        .encode_htj2k(&ht_options)
        .expect("second coefficient recode");

    assert_eq!(repeated_pixels.data, initial_pixels.data);
    assert_eq!(repeated_codestream, initial_codestream);
}

#[test]
fn repeated_direct_plan_build_reuses_immutable_tile_part_metadata() {
    let bytes = fixture_gray();
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();

    let first = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("first direct plan");
    let second = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("second direct plan");

    assert_eq!(second.dimensions, first.dimensions);
    assert_eq!(second.bit_depth, first.bit_depth);
    assert_eq!(second.steps.len(), first.steps.len());
    for (second_step, first_step) in second.steps.iter().zip(&first.steps) {
        assert_eq!(
            core::mem::discriminant(second_step),
            core::mem::discriminant(first_step)
        );
    }
}

#[test]
fn native_bytes_per_sample_tracks_high_bit_depths() {
    for (bit_depth, expected) in [
        (1_u8, 1_usize),
        (8, 1),
        (9, 2),
        (16, 2),
        (17, 3),
        (24, 3),
        (32, 4),
        (38, 5),
    ] {
        assert_eq!(native_bytes_per_sample(bit_depth).unwrap(), expected);
    }
}

#[test]
fn native_sample_packing_writes_high_bit_unsigned_little_endian_bytes() {
    let mut out = Vec::new();
    Image::push_native_sample_bytes(&mut out, 1_193_046.0, 24, false);
    assert_eq!(out, [0x56, 0x34, 0x12]);

    out.clear();
    Image::push_native_sample_bytes(&mut out, f32::MAX, 24, false);
    assert_eq!(out, [0xff, 0xff, 0xff]);
}

#[test]
fn native_sample_packing_writes_high_bit_signed_little_endian_bytes() {
    let mut out = Vec::new();
    Image::push_native_sample_bytes(&mut out, -1.0, 38, true);
    assert_eq!(out, [0xff, 0xff, 0xff, 0xff, 0xff]);

    out.clear();
    Image::push_native_sample_bytes(&mut out, -137_438_953_472.0, 38, true);
    assert_eq!(out, [0x00, 0x00, 0x00, 0x00, 0xe0]);
}

fn rewrite_siz_to_single_large_tile(codestream: &mut [u8], dimensions: u32) {
    let siz = codestream
        .windows(2)
        .position(|w| w == [0xFF, 0x51])
        .expect("SIZ marker");
    codestream[siz + 6..siz + 10].copy_from_slice(&dimensions.to_be_bytes());
    codestream[siz + 10..siz + 14].copy_from_slice(&dimensions.to_be_bytes());
    codestream[siz + 22..siz + 26].copy_from_slice(&dimensions.to_be_bytes());
    codestream[siz + 26..siz + 30].copy_from_slice(&dimensions.to_be_bytes());
}

fn rewrite_siz_tile_grid(codestream: &mut [u8], dimensions: (u32, u32), tile_size: (u32, u32)) {
    let siz = codestream
        .windows(2)
        .position(|w| w == [0xFF, 0x51])
        .expect("SIZ marker");
    codestream[siz + 6..siz + 10].copy_from_slice(&dimensions.0.to_be_bytes());
    codestream[siz + 10..siz + 14].copy_from_slice(&dimensions.1.to_be_bytes());
    codestream[siz + 22..siz + 26].copy_from_slice(&tile_size.0.to_be_bytes());
    codestream[siz + 26..siz + 30].copy_from_slice(&tile_size.1.to_be_bytes());
}

fn rewrite_siz_component_count(codestream: &mut Vec<u8>, component_count: u16) {
    let siz = codestream
        .windows(2)
        .position(|w| w == [0xFF, 0x51])
        .expect("SIZ marker");
    let old_component_count =
        u16::from_be_bytes([codestream[siz + 38], codestream[siz + 39]]) as usize;
    let component_start = siz + 40;
    let component_end = component_start + old_component_count * 3;
    let descriptor = codestream[component_start..component_start + 3].to_vec();
    let mut descriptors = Vec::with_capacity(usize::from(component_count) * 3);
    for _ in 0..component_count {
        descriptors.extend_from_slice(&descriptor);
    }

    let siz_len = 38_u16
        .checked_add(
            component_count
                .checked_mul(3)
                .expect("SIZ component bytes fit"),
        )
        .expect("SIZ length fits");
    codestream[siz + 2..siz + 4].copy_from_slice(&siz_len.to_be_bytes());
    codestream[siz + 38..siz + 40].copy_from_slice(&component_count.to_be_bytes());
    codestream.splice(component_start..component_end, descriptors);
}

#[test]
fn inspect_rejects_component_count_above_j2k_spec_cap() {
    let mut bytes = fixture_gray();
    rewrite_siz_component_count(&mut bytes, MAX_J2K_SPEC_COMPONENTS + 1);

    let err = inspect_j2k_codestream_header(&bytes)
        .expect_err("SIZ component count above spec cap must be rejected");

    assert_eq!(
        err,
        J2kCodestreamHeaderError::InvalidSiz {
            what: "component count exceeds JPEG 2000 limit"
        }
    );
}

#[test]
fn inspect_fallibly_materializes_the_maximum_component_metadata() {
    let mut bytes = fixture_gray();
    rewrite_siz_component_count(&mut bytes, MAX_J2K_SPEC_COMPONENTS);

    let metadata = inspect_j2k_codestream_header(&bytes)
        .expect("maximum JPEG 2000 component metadata remains inspectable");

    assert_eq!(metadata.components, MAX_J2K_SPEC_COMPONENTS);
    assert_eq!(
        metadata.component_info.len(),
        usize::from(MAX_J2K_SPEC_COMPONENTS)
    );
}

#[test]
fn native_parse_accepts_spec_component_count_above_u8() {
    let mut bytes = fixture_gray();
    rewrite_siz_component_count(&mut bytes, MAX_J2K_SPEC_COMPONENTS + 1);

    let Err(err) = Image::new(&bytes, &DecodeSettings::default()) else {
        panic!("component count above the JPEG 2000 spec cap must still reject");
    };
    assert_eq!(
        err,
        DecodeError::Validation(ValidationError::TooManyChannels)
    );

    let mut bytes = fixture_gray();
    rewrite_siz_component_count(&mut bytes, 256);
    Image::new(&bytes, &DecodeSettings::default())
        .expect("component count above u8 should parse within the JPEG 2000 spec cap");
}

#[test]
fn tile_parse_rejects_component_tile_structural_bomb_before_allocation() {
    let mut bytes = fixture_gray();
    rewrite_siz_component_count(&mut bytes, MAX_J2K_SPEC_COMPONENTS);
    rewrite_siz_tile_grid(&mut bytes, (256, 256), (1, 1));
    let parsed = j2c::parse_raw(&bytes, &DecodeSettings::default()).expect("raw header parses");
    let mut context = j2c::DecoderContext::default();
    let mut ht_decoder: Option<&mut dyn HtCodeBlockDecoder> = None;

    let retained_header_bytes = j2c::codestream::allocation::retained_header_bytes(&parsed.header)
        .expect("parsed header capacity");
    let err = j2c::decode(
        parsed.data,
        &parsed.header,
        retained_header_bytes,
        &mut context,
        &mut ht_decoder,
    )
    .expect_err("tile structural budget must reject before tile allocation");

    assert_eq!(err, DecodeError::Validation(ValidationError::ImageTooLarge));
}

#[test]
fn retained_container_metadata_rejects_header_parse_before_decode_growth() {
    let bytes = fixture_gray();
    let Err(error) = j2c::parse_raw_with_retained_baseline(
        &bytes,
        &DecodeSettings::default(),
        DEFAULT_MAX_DECODE_BYTES,
    ) else {
        panic!("a full retained-container baseline must leave no room for the header");
    };
    assert!(matches!(
        error,
        DecodeError::AllocationTooLarge {
            what: "native codestream header metadata",
            requested,
            cap: DEFAULT_MAX_DECODE_BYTES,
        } if requested > DEFAULT_MAX_DECODE_BYTES
    ));
}

#[test]
fn owned_decode_rejects_large_siz_before_allocating_output() {
    let mut bytes = fixture_gray();
    rewrite_siz_to_single_large_tile(&mut bytes, 60_000);
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("large SIZ parses");

    let Err(err) = image.decode() else {
        panic!("large owned decode must be capped");
    };

    assert_large_siz_component_budget(&err);
}

#[test]
fn decode_into_rejects_large_siz_before_allocating_component_storage() {
    let mut bytes = fixture_gray();
    rewrite_siz_to_single_large_tile(&mut bytes, 60_000);
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("large SIZ parses");
    let mut context = DecoderContext::default();
    let mut out = [];

    let Err(err) = image.decode_into(&mut out, &mut context) else {
        panic!("component storage must be capped before allocation");
    };

    assert_large_siz_component_budget(&err);
    assert!(
        context
            .tile_decode_context
            .channel_data
            .iter()
            .all(|component| component.container.capacity() == 0
                && component.integer_container.is_none()),
        "the cap must reject before component sample owners are allocated"
    );
}

fn assert_large_siz_component_budget(error: &DecodeError) {
    match error {
        DecodeError::AllocationTooLarge {
            what,
            requested,
            cap,
        } => {
            assert_eq!(*what, "native decoder context retained components");
            assert!(*requested > *cap);
            assert_eq!(*cap, DEFAULT_MAX_DECODE_BYTES);
        }
        other => panic!("unexpected large-SIZ rejection category: {other:?}"),
    }
}

#[test]
fn decode_region_rejects_large_full_tile_workspace_before_allocation() {
    let mut codestream = fixture_gray();
    rewrite_siz_to_single_large_tile(&mut codestream, 60_000);
    let image = Image::new(&codestream, &DecodeSettings::default()).expect("large SIZ parses");

    let Err(err) = image.decode_region((0, 0, 1, 1)) else {
        panic!("tiny ROI must not bypass the full-tile coefficient budget");
    };

    assert_eq!(err, DecodeError::Validation(ValidationError::ImageTooLarge));
}

#[test]
fn decode_region_rejects_code_block_layer_metadata_amplification() {
    let pixels: Vec<u8> = (0..64).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 0,
        code_block_width_exp: 0,
        code_block_height_exp: 0,
        num_layers: 32,
        ..EncodeOptions::default()
    };
    let mut bytes =
        encode(&pixels, 8, 8, 1, 8, false, &options).expect("encode many-layer classic fixture");
    rewrite_siz_to_single_large_tile(&mut bytes, 4_096);
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("large SIZ parses");

    let Err(err) = image.decode_region((0, 0, 1, 1)) else {
        panic!("tiny ROI must not bypass the structural metadata budget");
    };

    assert_eq!(err, DecodeError::Validation(ValidationError::ImageTooLarge));
}

fn fixture_ht_gray() -> Vec<u8> {
    let pixels: Vec<u8> = (0..16).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode_htj2k(&pixels, 4, 4, 1, 8, false, &options).expect("encode ht gray8")
}

fn fixture_ht_multi_block() -> Vec<u8> {
    let pixels: Vec<u8> = (0..64).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 0,
        code_block_width_exp: 0,
        code_block_height_exp: 0,
        ..EncodeOptions::default()
    };
    encode_htj2k(&pixels, 8, 8, 1, 8, false, &options).expect("encode multi-block HT gray8")
}

fn fixture_ht_rgb_multi_block() -> Vec<u8> {
    let pixels = gradient_pixels(8, 8, 3);
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 0,
        code_block_width_exp: 0,
        code_block_height_exp: 0,
        ..EncodeOptions::default()
    };
    encode_htj2k(&pixels, 8, 8, 3, 8, false, &options).expect("encode multi-block HT RGB8")
}

fn direct_ht_job_count(plan: &J2kDirectGrayscalePlan) -> usize {
    plan.steps
        .iter()
        .map(|step| match step {
            J2kDirectGrayscaleStep::HtSubBand(sub_band) => sub_band.jobs.len(),
            _ => 0,
        })
        .sum()
}

fn direct_color_ht_job_count(plan: &J2kDirectColorPlan) -> usize {
    plan.component_plans.iter().map(direct_ht_job_count).sum()
}

fn fixture_openhtj2k_ht_refinement() -> &'static [u8] {
    include_bytes!("../fixtures/htj2k/openhtj2k_ds0_ht_12_b11.j2k")
}

fn fixture_openhtj2k_ht_refinement_pixels() -> &'static [u8] {
    include_bytes!("../fixtures/htj2k/openhtj2k_ds0_ht_12_b11.gray")
}

fn fixture_openhtj2k_ht_refinement_odd() -> &'static [u8] {
    include_bytes!("../fixtures/htj2k/openhtj2k_ds0_ht_09_b11.j2k")
}

fn fixture_openhtj2k_ht_refinement_odd_pixels() -> &'static [u8] {
    include_bytes!("../fixtures/htj2k/openhtj2k_ds0_ht_09_b11.gray")
}

fn gradient_pixels(width: u32, height: u32, components: u8) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize * components as usize);
    for y in 0..height {
        for x in 0..width {
            for component in 0..components {
                pixels.push(((x * 3 + y * 5 + u32::from(component) * 41) & 0xff) as u8);
            }
        }
    }
    pixels
}

fn roi_fixture(classic: bool, components: u8) -> Vec<u8> {
    let width = 64;
    let height = 64;
    let pixels = gradient_pixels(width, height, components);
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        code_block_width_exp: 0,
        code_block_height_exp: 0,
        ..EncodeOptions::default()
    };
    if classic {
        encode(
            &pixels,
            width,
            height,
            components.into(),
            8,
            false,
            &options,
        )
        .expect("encode ROI classic fixture")
    } else {
        encode_htj2k(
            &pixels,
            width,
            height,
            components.into(),
            8,
            false,
            &options,
        )
        .expect("encode ROI HT fixture")
    }
}

fn crop_interleaved(
    full: &[u8],
    full_width: u32,
    channels: usize,
    roi: (u32, u32, u32, u32),
) -> Vec<u8> {
    let (x, y, width, height) = roi;
    let mut out = Vec::with_capacity(width as usize * height as usize * channels);
    let row_bytes = full_width as usize * channels;
    let roi_row_bytes = width as usize * channels;
    for row in y as usize..(y + height) as usize {
        let start = row * row_bytes + x as usize * channels;
        out.extend_from_slice(&full[start..start + roi_row_bytes]);
    }
    out
}

fn count_decode_work(bytes: &[u8], roi: Option<(u32, u32, u32, u32)>) -> DecodeWorkCounter {
    let image = Image::new(bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();
    match roi {
        Some(roi) => {
            image
                .decode_region_with_context(roi, &mut context)
                .expect("region decode with counter");
        }
        None => {
            image
                .decode_with_context(&mut context)
                .expect("full decode with counter");
        }
    }
    let counters = context.tile_decode_context.debug_counters;
    DecodeWorkCounter {
        classic_code_blocks: counters.decoded_code_blocks,
        ht_code_blocks: 0,
        idwt_output_samples: counters.idwt_output_samples,
    }
}

#[test]
fn roi_decode_matches_full_crop_for_classic_and_htj2k_gray_and_rgb() {
    let cases = [
        (true, 1_u8, true, false),
        (true, 3_u8, false, false),
        (false, 1_u8, true, false),
        (false, 3_u8, false, false),
    ];
    let rois = [
        (20, 18, 17, 19),
        (0, 0, 9, 11),
        (63, 63, 1, 1),
        (7, 5, 13, 9),
        (0, 0, 64, 64),
    ];

    for (classic, components, expect_gray, has_alpha) in cases {
        let bytes = roi_fixture(classic, components);
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let full = image.decode().expect("full decode");
        let channels = components as usize;
        for roi in rois {
            let region = image.decode_region(roi).expect("region decode");
            assert_eq!(matches!(region.color_space, ColorSpace::Gray), expect_gray);
            assert_eq!(region.has_alpha, has_alpha);
            assert_eq!(
                region.data,
                crop_interleaved(&full, 64, channels, roi),
                "classic={classic} components={components} roi={roi:?}"
            );
        }
    }
}

#[test]
fn roi_decode_prunes_code_blocks_and_idwt_work_for_classic_and_htj2k() {
    let roi = (48, 48, 16, 16);
    for classic in [true, false] {
        let bytes = {
            let pixels = gradient_pixels(128, 128, 1);
            let options = EncodeOptions {
                reversible: true,
                num_decomposition_levels: 3,
                code_block_width_exp: 0,
                code_block_height_exp: 0,
                ..EncodeOptions::default()
            };
            if classic {
                encode(&pixels, 128, 128, 1, 8, false, &options)
                    .expect("encode classic work fixture")
            } else {
                encode_htj2k(&pixels, 128, 128, 1, 8, false, &options)
                    .expect("encode ht work fixture")
            }
        };
        let full = count_decode_work(&bytes, None);
        let region = count_decode_work(&bytes, Some(roi));

        assert!(
            region.code_blocks() > 0 && region.code_blocks() < full.code_blocks(),
            "ROI should decode fewer code-blocks for classic={classic}; full={}, region={}",
            full.code_blocks(),
            region.code_blocks()
        );
        assert!(
                region.idwt_output_samples > 0
                    && region.idwt_output_samples < full.idwt_output_samples,
                "ROI should produce fewer IDWT output samples for classic={classic}; full={}, region={}",
                full.idwt_output_samples,
                region.idwt_output_samples
            );
    }
}

#[test]
fn region_decode_reuses_region_sized_component_storage() {
    let bytes = fixture();
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();

    let bitmap = image
        .decode_region_with_context((1, 0, 1, 2), &mut context)
        .expect("region decode");

    assert_eq!((bitmap.width, bitmap.height), (1, 2));
    assert_eq!(context.tile_decode_context.channel_data.len(), 3);
    assert!(context
        .tile_decode_context
        .channel_data
        .iter()
        .all(|component| component.container.truncated().len() == 2));
}

#[test]
fn native_region_decode_reuses_region_sized_component_storage() {
    let bytes = fixture();
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();

    let bitmap = image
        .decode_native_region_with_context((1, 0, 1, 2), &mut context)
        .expect("native region decode");

    assert_eq!((bitmap.width, bitmap.height), (1, 2));
    assert_eq!(context.tile_decode_context.channel_data.len(), 3);
    assert!(context
        .tile_decode_context
        .channel_data
        .iter()
        .all(|component| component.container.truncated().len() == 2));
}

#[derive(Debug, PartialEq, Eq)]
struct ChannelCapacitySnapshot {
    outer_ptr: *const ComponentData,
    outer_capacity: usize,
    components: Vec<ComponentCapacitySnapshot>,
}

#[derive(Debug, PartialEq, Eq)]
struct ComponentCapacitySnapshot {
    sample_ptr: *const f32,
    sample_capacity: usize,
    integer_owner: Option<(*const i64, usize)>,
}

fn channel_capacity_snapshot(context: &DecoderContext<'_>) -> ChannelCapacitySnapshot {
    let components = &context.tile_decode_context.channel_data;
    ChannelCapacitySnapshot {
        outer_ptr: components.as_ptr(),
        outer_capacity: components.capacity(),
        components: components
            .iter()
            .map(|component| ComponentCapacitySnapshot {
                sample_ptr: component.container.as_ptr(),
                sample_capacity: component.container.capacity(),
                integer_owner: component
                    .integer_container
                    .as_ref()
                    .map(|samples| (samples.as_ptr(), samples.capacity())),
            })
            .collect(),
    }
}

#[test]
fn decoder_context_reuses_component_owners_across_packed_and_component_outputs() {
    let bytes = fixture();
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();

    let first = image
        .decode_with_context(&mut context)
        .expect("first packed decode");
    assert_eq!(context.tile_decode_context.channel_data.len(), 3);
    let expected = first.data.clone();
    let owners = channel_capacity_snapshot(&context);
    context.tile_decode_context.channel_data[0].container[0] = f32::INFINITY;
    drop(first);

    let second = image
        .decode_with_context(&mut context)
        .expect("second packed decode");
    assert_eq!(second.data, expected);
    assert_eq!(channel_capacity_snapshot(&context), owners);
    drop(second);

    let component_output = image
        .decode_native_components_with_context(&mut context)
        .expect("owned component decode");
    assert_eq!(component_output.planes().len(), 3);
    assert_eq!(channel_capacity_snapshot(&context), owners);
    drop(component_output);

    let native = image
        .decode_native_with_context(&mut context)
        .expect("native packed decode");
    assert_eq!(native.data, expected);
    assert_eq!(channel_capacity_snapshot(&context), owners);
}

#[test]
fn exact_integer_decoder_context_reuses_and_resets_i64_samples() {
    let samples = [0_u32, 1, (1_u32 << 28) + 7, (1_u32 << 29) - 1];
    let pixels = samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect::<Vec<_>>();
    let planes = [EncodeTypedComponentPlane {
        data: &pixels,
        x_rsiz: 1,
        y_rsiz: 1,
        bit_depth: 29,
        signed: false,
    }];
    let bytes = encode_typed_component_planes_53(
        &planes,
        2,
        2,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            use_mct: false,
            ..EncodeOptions::default()
        },
    )
    .expect("encode exact integer fixture");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();

    let first = image
        .decode_native_with_context(&mut context)
        .expect("first exact decode");
    assert_eq!(first.data, pixels);
    let owners = channel_capacity_snapshot(&context);
    let integer_samples = context.tile_decode_context.channel_data[0]
        .integer_container
        .as_mut()
        .expect("29-bit decode uses exact integer samples");
    assert_eq!(integer_samples.len(), 4);
    integer_samples.fill(i64::MIN);
    drop(first);

    let second = image
        .decode_native_with_context(&mut context)
        .expect("second exact decode");
    assert_eq!(second.data, pixels);
    assert_eq!(channel_capacity_snapshot(&context), owners);
}

#[test]
fn decoder_context_defaults_to_auto_cpu_parallelism() {
    let context = DecoderContext::default();

    assert_eq!(context.cpu_decode_parallelism(), CpuDecodeParallelism::Auto);
}

#[test]
fn classic_j2k_auto_and_serial_cpu_parallelism_match_pixels() {
    let bytes = fixture_multi_block();
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut auto_context = DecoderContext::default();
    let mut serial_context = DecoderContext::default();
    serial_context.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);

    let auto = image
        .decode_with_context(&mut auto_context)
        .expect("auto decode");
    let serial = image
        .decode_with_context(&mut serial_context)
        .expect("serial decode");

    assert_eq!(auto.data, serial.data);
}

#[test]
fn htj2k_97_auto_and_serial_cpu_parallelism_match_pixels() {
    let width = 128_u32;
    let height = 128_u32;
    let pixels = (0..width * height)
        .map(|idx| ((idx * 17 + idx / width * 31) & 0xff) as u8)
        .collect::<Vec<_>>();
    let bytes = encode_htj2k(
        &pixels,
        width,
        height,
        1,
        8,
        false,
        &EncodeOptions {
            reversible: false,
            guard_bits: 2,
            num_decomposition_levels: 5,
            ..EncodeOptions::default()
        },
    )
    .expect("encode HTJ2K 9/7");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut auto_context = DecoderContext::default();
    let mut serial_context = DecoderContext::default();
    serial_context.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);

    let auto = image
        .decode_with_context(&mut auto_context)
        .expect("auto decode");
    let serial = image
        .decode_with_context(&mut serial_context)
        .expect("serial decode");

    assert_eq!(auto.data, serial.data);
}

#[test]
fn serial_cpu_parallelism_disables_classic_sub_band_parallel_branch() {
    assert!(!j2c::should_decode_classic_sub_band_in_parallel(
        CpuDecodeParallelism::Serial,
        16
    ));
}

#[test]
fn grayscale_direct_plan_is_built_without_materializing_channel_data() {
    let bytes = fixture_gray();
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();

    let plan = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("build direct plan");

    assert_eq!(plan.dimensions, (4, 4));
    assert_eq!(plan.bit_depth, 8);
    assert!(
        !plan.steps.is_empty(),
        "direct plan must contain executable steps"
    );
    assert!(
        plan.steps.iter().any(|step| matches!(
            step,
            J2kDirectGrayscaleStep::ClassicSubBand(plan) if !plan.jobs.is_empty()
        )),
        "classic J2K direct plan must contain at least one non-empty classic sub-band job"
    );
    assert!(
        context.tile_decode_context.channel_data.is_empty(),
        "building a direct plan must not materialize host component planes"
    );
    assert_eq!(
        context.storage.retained_capacity_bytes().unwrap(),
        0,
        "all source graph owners must be released after the owned plan handoff"
    );
}

#[test]
fn grayscale_direct_plan_honors_target_resolution() {
    let bytes = fixture_ht_gray();
    let image = Image::new(
        &bytes,
        &DecodeSettings {
            target_resolution: Some((2, 2)),
            ..DecodeSettings::default()
        },
    )
    .expect("scaled image");
    let mut context = DecoderContext::default();

    let plan = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("build scaled direct plan");

    assert_eq!(plan.dimensions, (2, 2));
    assert!(plan.steps.iter().any(|step| matches!(
        step,
        J2kDirectGrayscaleStep::HtSubBand(plan) if !plan.jobs.is_empty()
    )));
    assert!(plan.steps.iter().any(|step| matches!(
        step,
        J2kDirectGrayscaleStep::Store(store)
            if store.output_width == 2 && store.output_height == 2
    )));
    assert!(
        context.tile_decode_context.channel_data.is_empty(),
        "building a scaled direct plan must not materialize host component planes"
    );
}

#[test]
fn odd_dimensions_honor_covering_target_resolution() {
    let pixels = gradient_pixels(9, 7, 1);
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        ..EncodeOptions::default()
    };
    let bytes = encode_htj2k(&pixels, 9, 7, 1, 8, false, &options).expect("encode odd HT gray8");

    for (target, expected) in [((5, 4), (5, 4)), ((3, 2), (3, 2))] {
        let image = Image::new(
            &bytes,
            &DecodeSettings {
                target_resolution: Some(target),
                ..DecodeSettings::default()
            },
        )
        .expect("scaled odd image");

        assert_eq!(
            (image.width(), image.height()),
            expected,
            "target {target:?}"
        );
    }
}

#[test]
fn grayscale_direct_plan_region_prunes_unneeded_ht_code_blocks() {
    let bytes = fixture_ht_multi_block();
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut full_context = DecoderContext::default();
    let mut roi_context = DecoderContext::default();

    let full = image
        .build_direct_grayscale_plan_with_context(&mut full_context)
        .expect("build full direct plan");
    let roi = image
        .build_direct_grayscale_plan_region_with_context(&mut roi_context, (0, 0, 2, 2))
        .expect("build ROI direct plan");

    let full_jobs = direct_ht_job_count(&full);
    let roi_jobs = direct_ht_job_count(&roi);
    assert!(full_jobs > 1, "fixture must expose multiple HT jobs");
    assert!(
        roi_jobs < full_jobs,
        "ROI direct plan must prune HT jobs before device preparation"
    );
}

#[test]
fn color_direct_plan_region_prunes_unneeded_ht_code_blocks() {
    let bytes = fixture_ht_rgb_multi_block();
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut full_context = DecoderContext::default();
    let mut roi_context = DecoderContext::default();

    let full = image
        .build_direct_color_plan_with_context(&mut full_context)
        .expect("build full RGB direct plan");
    let roi = image
        .build_direct_color_plan_region_with_context(&mut roi_context, (0, 0, 2, 2))
        .expect("build ROI RGB direct plan");

    let full_jobs = direct_color_ht_job_count(&full);
    let roi_jobs = direct_color_ht_job_count(&roi);
    assert!(full_jobs > 3, "fixture must expose multiple RGB HT jobs");
    assert!(
        roi_jobs < full_jobs,
        "RGB ROI direct plan must prune HT jobs before device preparation"
    );
}

#[test]
fn color_direct_plan_honors_target_resolution() {
    for (name, bytes) in [
        ("classic", {
            let pixels = gradient_pixels(8, 8, 3);
            let options = EncodeOptions {
                reversible: true,
                num_decomposition_levels: 2,
                ..EncodeOptions::default()
            };
            encode(&pixels, 8, 8, 3, 8, false, &options).expect("encode classic rgb8")
        }),
        ("htj2k", {
            let pixels = gradient_pixels(8, 8, 3);
            let options = EncodeOptions {
                reversible: true,
                num_decomposition_levels: 2,
                ..EncodeOptions::default()
            };
            encode_htj2k(&pixels, 8, 8, 3, 8, false, &options).expect("encode ht rgb8")
        }),
    ] {
        let image = Image::new(
            &bytes,
            &DecodeSettings {
                target_resolution: Some((4, 4)),
                ..DecodeSettings::default()
            },
        )
        .expect("scaled RGB image");
        let mut context = DecoderContext::default();

        let plan = image
            .build_direct_color_plan_with_context(&mut context)
            .expect("build scaled direct color plan");

        assert_eq!(plan.dimensions, (4, 4), "{name}: output dimensions");
        assert_eq!(plan.component_plans.len(), 3, "{name}: component count");
        for component_plan in &plan.component_plans {
            assert_eq!(component_plan.dimensions, (4, 4), "{name}: component dims");
            assert!(component_plan.steps.iter().any(|step| matches!(
                step,
                J2kDirectGrayscaleStep::Store(store)
                    if store.output_width == 4 && store.output_height == 4
            )));
        }
        assert!(
                context.tile_decode_context.channel_data.is_empty(),
                "{name}: building a scaled color direct plan must not materialize host component planes"
            );
    }
}

#[test]
fn direct_color_cpu_rgb8_executor_matches_scaled_region_decode() {
    for (name, bytes) in [
        ("classic", {
            let pixels = gradient_pixels(16, 16, 3);
            let options = EncodeOptions {
                reversible: true,
                num_decomposition_levels: 2,
                ..EncodeOptions::default()
            };
            encode(&pixels, 16, 16, 3, 8, false, &options).expect("encode classic rgb8")
        }),
        ("htj2k", {
            let pixels = gradient_pixels(16, 16, 3);
            let options = EncodeOptions {
                reversible: true,
                num_decomposition_levels: 2,
                ..EncodeOptions::default()
            };
            encode_htj2k(&pixels, 16, 16, 3, 8, false, &options).expect("encode ht rgb8")
        }),
    ] {
        let image = Image::new(
            &bytes,
            &DecodeSettings {
                target_resolution: Some((4, 4)),
                ..DecodeSettings::default()
            },
        )
        .expect("scaled RGB image");
        let mut expected_context = DecoderContext::default();
        let expected_full = image
            .decode_with_context(&mut expected_context)
            .expect("decode scaled reference");
        let output_region = J2kRect {
            x0: 1,
            y0: 1,
            x1: 3,
            y1: 3,
        };
        let mut direct_context = DecoderContext::default();
        let plan = image
            .build_direct_color_plan_region_with_context(
                &mut direct_context,
                (
                    output_region.x0,
                    output_region.y0,
                    output_region.width(),
                    output_region.height(),
                ),
            )
            .expect("build direct RGB region plan");

        let stride = output_region.width() as usize * 3;
        let mut direct = vec![0_u8; stride * output_region.height() as usize];
        let mut scratch = J2kDirectCpuScratch::new();
        execute_direct_color_plan_rgb8_into(
            &plan,
            output_region,
            &mut scratch,
            &mut direct,
            stride,
        )
        .expect("execute direct RGB plan");

        let mut expected = Vec::with_capacity(direct.len());
        let full_stride = image.width() as usize * 3;
        for y in output_region.y0..output_region.y1 {
            let start = y as usize * full_stride + output_region.x0 as usize * 3;
            expected.extend_from_slice(&expected_full.data[start..start + stride]);
        }

        assert_eq!(direct, expected, "{name}: direct RGB output");

        let rgba_stride = output_region.width() as usize * 4;
        let mut direct_rgba = vec![0_u8; rgba_stride * output_region.height() as usize];
        execute_direct_color_plan_rgba8_into(
            &plan,
            output_region,
            &mut scratch,
            &mut direct_rgba,
            rgba_stride,
        )
        .expect("execute direct RGBA plan");

        let mut expected_rgba = Vec::with_capacity(direct_rgba.len());
        for rgb in expected.chunks_exact(3) {
            expected_rgba.extend_from_slice(rgb);
            expected_rgba.push(255);
        }
        assert_eq!(direct_rgba, expected_rgba, "{name}: direct RGBA output");
    }
}

#[test]
fn htj2k_grayscale_direct_plan_contains_ht_sub_band_steps() {
    let bytes = fixture_ht_gray();
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();

    let plan = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("build direct plan");

    assert!(
        plan.steps.iter().any(|step| matches!(
            step,
            J2kDirectGrayscaleStep::HtSubBand(plan) if !plan.jobs.is_empty()
        )),
        "HTJ2K direct plan must contain at least one non-empty HT sub-band decode step"
    );
}

#[test]
fn ht_decoder_hook_is_used_for_htj2k_codeblocks() {
    let pixels: Vec<u8> = (0..16).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let bytes = encode_htj2k(&pixels, 4, 4, 1, 8, false, &options).expect("encode ht");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut hooked_context = DecoderContext::default();
    let mut hook = FailingHtDecoder { called: false };
    let Err(error) = image.decode_components_with_ht_decoder(&mut hooked_context, &mut hook) else {
        panic!("hooked decode must use external HT decoder");
    };

    assert!(hook.called, "HT decoder hook must be invoked");
    assert_eq!(
        error,
        DecodeError::Decoding(DecodingError::CodeBlockDecodeFailure)
    );
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
fn scalar_htj2k_encoder_contract_is_cleanup_only() {
    let coefficients = (0..64)
        .map(|index| {
            let magnitude = (index % 7) + 1;
            if index % 2 == 0 {
                magnitude
            } else {
                -magnitude
            }
        })
        .collect::<Vec<_>>();

    let encoded =
        encode_ht_code_block_scalar(&coefficients, 8, 8, 8).expect("encode HT code block");

    assert_eq!(
        encoded.num_coding_passes, 1,
        "current scalar HTJ2K encoder emits only the cleanup pass"
    );
    assert_eq!(
        encoded.num_zero_bitplanes, 7,
        "current cleanup-only HTJ2K encoder includes one bitplane"
    );
    assert!(
        !encoded.data.is_empty(),
        "non-zero cleanup-only block must still produce payload bytes"
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

#[test]
fn classic_decoder_hook_is_used_for_j2k_codeblocks() {
    let bytes = fixture();
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut hooked_context = DecoderContext::default();
    let mut hook = FailingClassicDecoder { called: false };
    let Err(error) = image.decode_components_with_ht_decoder(&mut hooked_context, &mut hook) else {
        panic!("hooked decode must use external classic decoder");
    };

    assert!(hook.called, "classic decoder hook must be invoked");
    assert_eq!(
        error,
        DecodeError::Decoding(DecodingError::CodeBlockDecodeFailure)
    );
}

#[test]
fn classic_sub_band_decoder_hook_is_used_for_j2k_codeblocks() {
    let bytes = fixture_multi_block();
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut hooked_context = DecoderContext::default();
    let mut hook = FailingClassicBatchDecoder { called: false };
    let Err(error) = image.decode_components_with_ht_decoder(&mut hooked_context, &mut hook) else {
        panic!("hooked decode must use external classic batch decoder");
    };

    assert!(hook.called, "classic sub-band decoder hook must be invoked");
    assert_eq!(
        error,
        DecodeError::Decoding(DecodingError::CodeBlockDecodeFailure)
    );
}

// -----------------------------------------------------------------------
// Sanity tests for the four scalar-reference exports
// -----------------------------------------------------------------------

#[test]
fn forward_dwt53_reference_matches_internal_path() {
    // 4×4 constant-ramp input; 1 decomposition level.
    let samples: Vec<f32> = (0_u8..16).map(f32::from).collect();
    let out = forward_dwt53_reference(&samples, 4, 4, 1).expect("fallible 5/3 reference DWT");

    // Internal path
    let internal = j2c::fdwt::forward_dwt(&samples, 4, 4, 1, true);

    assert_eq!(out.ll, internal.ll, "LL subband mismatch");
    assert_eq!(out.ll_width, internal.ll_width, "LL width mismatch");
    assert_eq!(out.ll_height, internal.ll_height, "LL height mismatch");
    assert_eq!(out.levels.len(), internal.levels.len(), "level count");
    for (pub_lvl, int_lvl) in out.levels.iter().zip(internal.levels.iter()) {
        assert_eq!(pub_lvl.hl, int_lvl.hl, "HL mismatch");
        assert_eq!(pub_lvl.lh, int_lvl.lh, "LH mismatch");
        assert_eq!(pub_lvl.hh, int_lvl.hh, "HH mismatch");
    }
}

#[test]
fn scalar_forward_dwt_rejects_caller_geometry_with_typed_errors() {
    let short = forward_dwt53_reference(&[0.0; 3], 2, 2, 1)
        .expect_err("short 5/3 reference plane must fail");
    assert_eq!(
        short,
        EncodeError::InvalidInput {
            what: "packed forward DWT coefficient length mismatch",
        }
    );

    let zero = forward_dwt97_reference(&[], 0, 1, 1)
        .expect_err("zero-width 9/7 reference plane must fail");
    assert_eq!(
        zero,
        EncodeError::InvalidInput {
            what: "packed forward DWT dimensions must be non-zero",
        }
    );
}

#[test]
#[expect(
    clippy::float_cmp,
    reason = "the reversible color transform uses exactly representable integer-valued f32 outputs"
)]
fn forward_rct_reference_matches_internal_path() {
    // Single pixel: R=100, G=150, B=200
    let planes = vec![vec![100.0f32], vec![150.0f32], vec![200.0f32]];
    let result = forward_rct_reference(planes.clone());

    // Internal path
    let mut internal = planes;
    j2c::forward_mct::forward_rct(&mut internal);

    assert_eq!(result, internal, "RCT output mismatch");
    // Y = floor((100 + 300 + 200) / 4) = 150
    assert_eq!(result[0][0], 150.0, "Y component");
    assert_eq!(result[1][0], 50.0, "Cb component");
    assert_eq!(result[2][0], -50.0, "Cr component");
}

#[test]
fn forward_ict_reference_matches_internal_path() {
    let planes = vec![vec![100.0f32], vec![150.0f32], vec![200.0f32]];
    let result = forward_ict_reference(planes.clone());

    let mut internal = planes;
    j2c::forward_mct::forward_ict(&mut internal);

    assert_eq!(result, internal, "ICT output mismatch");
}

#[test]
fn forward_dwt97_reference_matches_internal_path() {
    let samples = (0..64)
        .map(|idx| {
            f32::from(u8::try_from((idx * 19 + idx / 3) & 0xff).expect("masked sample fits u8"))
                - 128.0
        })
        .collect::<Vec<_>>();
    let result = forward_dwt97_reference(&samples, 8, 8, 2).expect("fallible 9/7 reference DWT");
    let internal = j2c::fdwt::forward_dwt(&samples, 8, 8, 2, false);

    assert_eq!(result.ll, internal.ll, "DWT 9/7 LL mismatch");
    assert_eq!(result.ll_width, internal.ll_width);
    assert_eq!(result.ll_height, internal.ll_height);
    assert_eq!(result.levels.len(), internal.levels.len());
    for (actual, expected) in result.levels.iter().zip(internal.levels.iter()) {
        assert_eq!(actual.hl, expected.hl, "DWT 9/7 HL mismatch");
        assert_eq!(actual.lh, expected.lh, "DWT 9/7 LH mismatch");
        assert_eq!(actual.hh, expected.hh, "DWT 9/7 HH mismatch");
    }
}

#[test]
fn quantize_reversible_reference_matches_internal_path() {
    let coefficients = vec![3.7f32, -8.2, 0.5, -0.5, 10.0];
    let exponent = 8u16;
    let mantissa = 0u16;
    let range_bits = 8u8;

    let result = quantize_reversible_reference(&coefficients, exponent, mantissa, range_bits, true);

    // Internal path
    let step = j2c::quantize::QuantStepSize { exponent, mantissa };
    let internal = j2c::quantize::quantize_subband(&coefficients, &step, range_bits, true);

    assert_eq!(result, internal, "quantize output mismatch");
    // reversible: round to nearest
    assert_eq!(result[0], 4, "3.7 rounds to 4");
    assert_eq!(result[1], -8, "-8.2 rounds to -8");
}

#[test]
fn quantize_subband_reference_matches_irreversible_internal_path() {
    let coefficients = vec![3.7f32, -8.2, 0.5, -0.5, 10.0];
    let exponent = 8u16;
    let mantissa = 256u16;
    let range_bits = 8u8;

    let result = quantize_subband_reference(&coefficients, exponent, mantissa, range_bits, false);

    let step = j2c::quantize::QuantStepSize { exponent, mantissa };
    let internal = j2c::quantize::quantize_subband(&coefficients, &step, range_bits, false);

    assert_eq!(result, internal, "irreversible quantize output mismatch");
}

#[test]
fn deinterleave_reference_matches_internal_path() {
    // 2-pixel RGB8 unsigned: [R0,G0,B0, R1,G1,B1]
    let pixels: Vec<u8> = vec![128, 64, 200, 10, 20, 30];
    let result = try_deinterleave_reference(&pixels, 2, 3, 8, false)
        .expect("valid deinterleave reference input");

    let internal = j2c::encode::deinterleave_to_f32(&pixels, 2, 3, 8, false);

    assert_eq!(result, internal, "deinterleave output mismatch");
    assert_eq!(result.len(), 3, "three component planes");
    assert_eq!(result[0].len(), 2, "two pixels per plane");
    // unsigned 8-bit with level shift: val - 128
    assert!((result[0][0] - 0.0f32).abs() < 1e-6, "R0 level-shifted");
    assert!((result[1][0] - (-64.0f32)).abs() < 1e-6, "G0 level-shifted");
    assert!((result[2][0] - 72.0f32).abs() < 1e-6, "B0 level-shifted");
}

#[test]
fn try_deinterleave_reference_rejects_invalid_geometry() {
    let valid_pixels: Vec<u8> = vec![128, 64, 200, 10, 20, 30];

    for (label, result) in [
        (
            "zero components",
            try_deinterleave_reference(&valid_pixels, 2, 0, 8, false),
        ),
        (
            "zero bit depth",
            try_deinterleave_reference(&valid_pixels, 2, 3, 0, false),
        ),
        (
            "unsupported bit depth",
            try_deinterleave_reference(&valid_pixels, 2, 3, 39, false),
        ),
        (
            "short input",
            try_deinterleave_reference(&valid_pixels[..5], 2, 3, 8, false),
        ),
        (
            "trailing bytes",
            try_deinterleave_reference(&[valid_pixels.as_slice(), &[0]].concat(), 2, 3, 8, false),
        ),
    ] {
        assert!(
            matches!(
                result,
                Err(DecodeError::Validation(
                    ValidationError::InvalidComponentMetadata
                ))
            ),
            "{label} should be rejected, got {result:?}"
        );
    }
}

#[test]
fn decode_settings_constructors_make_strictness_explicit() {
    assert!(DecodeSettings::default().lenient_tolerance_enabled());
    assert!(DecodeSettings::lenient().lenient_tolerance_enabled());
    assert!(!DecodeSettings::strict().lenient_tolerance_enabled());
    assert!(DecodeSettings::strict().strict);
}
