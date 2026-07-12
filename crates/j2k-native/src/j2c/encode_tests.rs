// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    assign_classic_segment_layers_by_slope, assign_ht_segment_layers_by_budget, bitplane_encode,
    copy_code_block_coefficients, deinterleave_rgb8_unsigned_to_f32, deinterleave_to_f32,
    downcast_i64_coefficients_to_i32, encode, encode_all_ht_code_blocks_parallel,
    encode_all_ht_code_blocks_serial_cpu, encode_htj2k, encode_precomputed_htj2k_53,
    encode_precomputed_htj2k_53_with_accelerator, encode_precomputed_htj2k_97,
    encode_precomputed_htj2k_97_batch_with_accelerator,
    encode_precomputed_htj2k_97_with_accelerator, encode_preencoded_htj2k_97,
    encode_preencoded_htj2k_97_compact_owned_with_accelerator, encode_prepared_subbands,
    encode_prequantized_htj2k_97, encode_prequantized_htj2k_97_with_accelerator,
    encode_with_accelerator, forward_dwt53_output_from_decomposition, ht_block_encode,
    ht_layer_contributions, ht_target_coding_passes_for_options, prepare_subband,
    prepared_subband_from_preencoded_owned, public_sub_band_type, quantize,
    validate_packet_header_marker_payloads, validate_precomputed_dwt97_geometry,
    validate_precomputed_dwt_geometry, BlockCodingMode, ClassicSegmentAssignmentCandidate,
    CpuOnlyJ2kEncodeStageAccelerator, EncodeOptions, EncodedHtJ2kCodeBlock,
    HtSegmentAssignmentCandidate, J2kEncodeStageAccelerator, J2kForwardDwt53Level,
    J2kForwardDwt53Output, J2kForwardDwt97Level, J2kForwardDwt97Output, J2kSubBandType,
    PrecomputedHtj2k53Component, PrecomputedHtj2k53Image, PrecomputedHtj2k97Component,
    PrecomputedHtj2k97Image, PreencodedHtj2k97CodeBlock, PreencodedHtj2k97CompactCodeBlock,
    PreencodedHtj2k97CompactComponent, PreencodedHtj2k97CompactImage,
    PreencodedHtj2k97CompactResolution, PreencodedHtj2k97CompactSubband,
    PreencodedHtj2k97Component, PreencodedHtj2k97Image, PreencodedHtj2k97Resolution,
    PreencodedHtj2k97Subband, PreparedCodeBlockCoefficients, PrequantizedHtj2k97Component,
    PrequantizedHtj2k97Image, PrequantizedHtj2k97Resolution, PrequantizedHtj2k97Subband,
    QuantStepSize, SubBandType, HT_CPU_PARALLEL_FALLBACK_MIN_JOBS,
};
use crate::{DecodeSettings, EncodeError, Image, PrequantizedHtj2k97CodeBlock};
use alloc::{vec, vec::Vec};

fn test_preencoded_subband_payload(marker: u8) -> PreencodedHtj2k97Subband {
    PreencodedHtj2k97Subband {
        sub_band_type: J2kSubBandType::LowLow,
        num_cbs_x: 1,
        num_cbs_y: 1,
        total_bitplanes: 8,
        code_blocks: vec![PreencodedHtj2k97CodeBlock {
            width: 1,
            height: 1,
            encoded: crate::EncodedHtJ2kCodeBlock {
                data: vec![marker; 8],
                cleanup_length: 8,
                refinement_length: 0,
                num_coding_passes: 1,
                num_zero_bitplanes: 0,
            },
        }],
    }
}

#[test]
fn prepared_subband_from_owned_preencoded_moves_payload_without_clone() {
    let subband = test_preencoded_subband_payload(7);
    let original_ptr = subband.code_blocks[0].encoded.data.as_ptr() as usize;

    let prepared = prepared_subband_from_preencoded_owned(subband);
    let prepared_blocks = prepared
        .preencoded_ht_code_blocks
        .expect("preencoded payloads");

    assert_eq!(prepared_blocks[0].data.as_ptr() as usize, original_ptr);
    assert!(prepared.code_blocks[0].coefficients.is_empty());
}

#[derive(Default)]
struct RecordingPacketizationAccelerator {
    payload_base: usize,
    observed_offsets: Vec<usize>,
    observed_lengths: Vec<usize>,
}

impl crate::J2kEncodeStageAccelerator for RecordingPacketizationAccelerator {
    fn encode_packetization(
        &mut self,
        job: crate::J2kPacketizationEncodeJob<'_>,
    ) -> crate::J2kEncodeStageResult<Option<Vec<u8>>> {
        for code_block in job
            .resolutions
            .iter()
            .flat_map(|resolution| resolution.subbands.iter())
            .flat_map(|subband| subband.code_blocks.iter())
            .filter(|code_block| !code_block.data.is_empty())
        {
            self.observed_offsets
                .push((code_block.data.as_ptr() as usize) - self.payload_base);
            self.observed_lengths.push(code_block.data.len());
        }
        Ok(Some(crate::encode_j2k_packetization_scalar(job).map_err(
            |source| {
                crate::J2kEncodeStageError::backend(
                    "scalar test accelerator",
                    "packetization",
                    source,
                )
            },
        )?))
    }
}

#[test]
fn compact_preencoded_packetization_borrows_payload_ranges() {
    let (preencoded, options) = sample_preencoded_htj2k97_for_test();
    let expected = encode_preencoded_htj2k_97(&preencoded, &options).expect("owned preencoded");
    let mut payload = Vec::new();
    let mut expected_offsets = Vec::new();
    let mut expected_lengths = Vec::new();
    let components = preencoded
        .components
        .iter()
        .map(|component| PreencodedHtj2k97CompactComponent {
            x_rsiz: component.x_rsiz,
            y_rsiz: component.y_rsiz,
            resolutions: component
                .resolutions
                .iter()
                .map(|resolution| PreencodedHtj2k97CompactResolution {
                    subbands: resolution
                        .subbands
                        .iter()
                        .map(|subband| PreencodedHtj2k97CompactSubband {
                            sub_band_type: subband.sub_band_type,
                            num_cbs_x: subband.num_cbs_x,
                            num_cbs_y: subband.num_cbs_y,
                            total_bitplanes: subband.total_bitplanes,
                            code_blocks: subband
                                .code_blocks
                                .iter()
                                .map(|block| {
                                    let start = payload.len();
                                    payload.extend_from_slice(&block.encoded.data);
                                    let end = payload.len();
                                    if start != end {
                                        expected_offsets.push(start);
                                        expected_lengths.push(end - start);
                                    }
                                    PreencodedHtj2k97CompactCodeBlock {
                                        width: block.width,
                                        height: block.height,
                                        payload_range: start..end,
                                        cleanup_length: block.encoded.cleanup_length,
                                        refinement_length: block.encoded.refinement_length,
                                        num_coding_passes: block.encoded.num_coding_passes,
                                        num_zero_bitplanes: block.encoded.num_zero_bitplanes,
                                    }
                                })
                                .collect(),
                        })
                        .collect(),
                })
                .collect(),
        })
        .collect();
    let compact = PreencodedHtj2k97CompactImage {
        width: preencoded.width,
        height: preencoded.height,
        bit_depth: preencoded.bit_depth,
        signed: preencoded.signed,
        payload,
        components,
    };
    let mut accelerator = RecordingPacketizationAccelerator {
        payload_base: compact.payload.as_ptr() as usize,
        ..Default::default()
    };

    let actual = encode_preencoded_htj2k_97_compact_owned_with_accelerator(
        compact,
        &options,
        &mut accelerator,
    )
    .expect("compact preencoded");

    assert_eq!(actual, expected);
    assert_eq!(accelerator.observed_offsets, expected_offsets);
    assert_eq!(accelerator.observed_lengths, expected_lengths);
}

#[test]
fn test_encode_8bit_gray() {
    let width = 8u32;
    let height = 8u32;
    let pixels: Vec<u8> = (0..64).collect();

    let result = encode(
        &pixels,
        width,
        height,
        1,
        8,
        false,
        &EncodeOptions {
            num_decomposition_levels: 2,
            ..Default::default()
        },
    );

    assert!(result.is_ok());
    let codestream = result.unwrap();
    // Verify SOC marker
    assert_eq!(codestream[0], 0xFF);
    assert_eq!(codestream[1], 0x4F);
    // Verify EOC marker
    let len = codestream.len();
    assert_eq!(codestream[len - 2], 0xFF);
    assert_eq!(codestream[len - 1], 0xD9);
}

#[test]
fn test_encode_16bit_gray() {
    let width = 8u32;
    let height = 8u32;
    let mut pixels = Vec::with_capacity(128);
    for i in 0..64u16 {
        let val = i * 100;
        pixels.extend_from_slice(&val.to_le_bytes());
    }

    let result = encode(
        &pixels,
        width,
        height,
        1,
        16,
        false,
        &EncodeOptions {
            num_decomposition_levels: 2,
            ..Default::default()
        },
    );

    assert!(result.is_ok());
}

#[test]
fn test_encode_rgb() {
    let width = 16u32;
    let height = 16u32;
    let pixels: Vec<u8> = (0..width * height * 3).map(|i| (i & 0xFF) as u8).collect();

    let result = encode(
        &pixels,
        width,
        height,
        3,
        8,
        false,
        &EncodeOptions {
            num_decomposition_levels: 3,
            ..Default::default()
        },
    );

    assert!(result.is_ok(), "RGB encode failed: {:?}", result.err());
}

#[test]
fn encode_with_accelerator_calls_lossless_stage_hooks() {
    #[derive(Default)]
    struct CountingAccelerator {
        forward_rct: usize,
        forward_dwt53: usize,
        tier1_code_blocks: usize,
        tier1_code_block_batches: usize,
        tier1_batched_jobs: usize,
        packetization: usize,
        packetization_resolution_count: u32,
        packetization_code_block_count: u32,
        packetization_saw_payload: bool,
    }

    impl crate::J2kEncodeStageAccelerator for CountingAccelerator {
        fn encode_forward_rct(
            &mut self,
            _job: crate::J2kForwardRctJob<'_>,
        ) -> crate::J2kEncodeStageResult<bool> {
            self.forward_rct += 1;
            Ok(false)
        }

        fn encode_forward_dwt53(
            &mut self,
            _job: crate::J2kForwardDwt53Job<'_>,
        ) -> crate::J2kEncodeStageResult<Option<crate::J2kForwardDwt53Output>> {
            self.forward_dwt53 += 1;
            Ok(None)
        }

        fn encode_tier1_code_block(
            &mut self,
            _job: crate::J2kTier1CodeBlockEncodeJob<'_>,
        ) -> crate::J2kEncodeStageResult<Option<crate::EncodedJ2kCodeBlock>> {
            self.tier1_code_blocks += 1;
            Ok(None)
        }

        fn encode_tier1_code_blocks(
            &mut self,
            jobs: &[crate::J2kTier1CodeBlockEncodeJob<'_>],
        ) -> crate::J2kEncodeStageResult<Option<Vec<crate::EncodedJ2kCodeBlock>>> {
            self.tier1_code_block_batches += 1;
            self.tier1_batched_jobs += jobs.len();
            Ok(None)
        }

        fn encode_packetization(
            &mut self,
            job: crate::J2kPacketizationEncodeJob<'_>,
        ) -> crate::J2kEncodeStageResult<Option<Vec<u8>>> {
            self.packetization += 1;
            self.packetization_resolution_count = job.resolution_count;
            self.packetization_code_block_count = job.code_block_count;
            self.packetization_saw_payload = job
                .resolutions
                .iter()
                .flat_map(|resolution| resolution.subbands.iter())
                .flat_map(|subband| subband.code_blocks.iter())
                .any(|code_block| !code_block.data.is_empty());
            Ok(None)
        }
    }

    let pixels: Vec<u8> = (0..8 * 8 * 3)
        .map(|i| u8::try_from(i & 0xFF).expect("masked test pixel fits u8"))
        .collect();
    let options = EncodeOptions {
        num_decomposition_levels: 1,
        reversible: true,
        ..EncodeOptions::default()
    };
    let mut accelerator = CountingAccelerator::default();

    let codestream =
        encode_with_accelerator(&pixels, 8, 8, 3, 8, false, &options, &mut accelerator)
            .expect("encode with accelerator hooks");

    assert!(codestream.starts_with(&[0xFF, 0x4F]));
    assert_eq!(accelerator.forward_rct, 1);
    assert_eq!(accelerator.forward_dwt53, 3);
    assert!(accelerator.tier1_code_block_batches > 0);
    assert_eq!(
        accelerator.tier1_code_blocks,
        accelerator.tier1_batched_jobs
    );
    assert_eq!(accelerator.packetization, 1);
    assert_eq!(accelerator.packetization_resolution_count, 6);
    assert_eq!(
        accelerator.packetization_code_block_count,
        u32::try_from(accelerator.tier1_code_blocks).expect("test code-block count fits u32")
    );
    assert!(accelerator.packetization_saw_payload);
}

#[test]
fn cpu_only_accelerator_opts_into_parallel_block_fallback_only_for_native_cpu() {
    #[derive(Default)]
    struct ExternalAccelerator;

    impl crate::J2kEncodeStageAccelerator for ExternalAccelerator {}

    let cpu = crate::CpuOnlyJ2kEncodeStageAccelerator;
    let external = ExternalAccelerator;

    assert!(cpu.prefer_parallel_cpu_code_block_fallback());
    assert!(!external.prefer_parallel_cpu_code_block_fallback());
}

#[test]
fn cpu_parallel_block_fallback_matches_serial_classic_and_htj2k_output() {
    #[derive(Default)]
    struct SerialCpuFallbackAccelerator;

    impl crate::J2kEncodeStageAccelerator for SerialCpuFallbackAccelerator {}

    let pixels = gradient_u8(96, 80);
    for use_ht_block_coding in [false, true] {
        let options = EncodeOptions {
            num_decomposition_levels: 1,
            code_block_width_exp: 2,
            code_block_height_exp: 2,
            use_ht_block_coding,
            ..EncodeOptions::default()
        };
        let parallel =
            encode(&pixels, 96, 80, 1, 8, false, &options).expect("parallel CPU fallback encode");
        let mut serial_accelerator = SerialCpuFallbackAccelerator;
        let serial = encode_with_accelerator(
            &pixels,
            96,
            80,
            1,
            8,
            false,
            &options,
            &mut serial_accelerator,
        )
        .expect("serial CPU fallback encode");

        assert_eq!(parallel, serial);
    }
}

#[test]
fn precomputed_htj2k53_offers_ht_code_blocks_to_encode_accelerator() {
    let image = sample_precomputed_htj2k53_image();
    let options = EncodeOptions {
        num_decomposition_levels: 1,
        reversible: true,
        guard_bits: 2,
        code_block_width_exp: 2,
        code_block_height_exp: 2,
        ..EncodeOptions::default()
    };
    let mut accelerator = CountingHtEncodeAccelerator::default();

    let encoded = encode_precomputed_htj2k_53_with_accelerator(&image, &options, &mut accelerator)
        .expect("precomputed 5/3 encode accepts encode accelerator");

    assert!(encoded.starts_with(&[0xff, 0x4f]));
    assert_eq!(accelerator.deinterleave, 0);
    assert_eq!(accelerator.forward_dwt53, 0);
    assert_eq!(accelerator.forward_dwt97, 0);
    assert_eq!(accelerator.ht_batches, 1);
    assert!(accelerator.ht_jobs > 0);
    assert_eq!(accelerator.ht_single_blocks, accelerator.ht_jobs);
}

#[test]
fn precomputed_htj2k53_borrowed_coefficients_match_pixel_pipeline_codestream() {
    let width = 17_u32;
    let height = 13_u32;
    let num_pixels = usize::try_from(
        width
            .checked_mul(height)
            .expect("test image dimensions fit u32"),
    )
    .expect("test image dimensions fit usize");
    let pixels = (0..num_pixels)
        .map(|index| {
            u8::try_from((index * 37 + index / 5) & 0xff).expect("masked test sample fits u8")
        })
        .collect::<Vec<_>>();
    let options = EncodeOptions {
        num_decomposition_levels: 1,
        reversible: true,
        guard_bits: 2,
        code_block_width_exp: 2,
        code_block_height_exp: 2,
        ..EncodeOptions::default()
    };

    let expected = encode_htj2k(&pixels, width, height, 1, 8, false, &options)
        .expect("pixel-pipeline HTJ2K encode");
    let samples = deinterleave_to_f32(&pixels, num_pixels, 1, 8, false);
    let decomposition = crate::j2c::fdwt::forward_dwt(&samples[0], width, height, 1, true);
    let image = PrecomputedHtj2k53Image {
        width,
        height,
        bit_depth: 8,
        signed: false,
        components: vec![PrecomputedHtj2k53Component {
            x_rsiz: 1,
            y_rsiz: 1,
            dwt: forward_dwt53_output_from_decomposition(decomposition),
        }],
    };

    let actual =
        encode_precomputed_htj2k_53(&image, &options).expect("borrowed precomputed HTJ2K encode");

    assert_eq!(actual, expected);
}

#[test]
fn precomputed_htj2k97_offers_ht_code_blocks_to_encode_accelerator() {
    let image = sample_precomputed_htj2k97_image();
    let options = EncodeOptions {
        num_decomposition_levels: 1,
        reversible: false,
        guard_bits: 2,
        code_block_width_exp: 2,
        code_block_height_exp: 2,
        ..EncodeOptions::default()
    };
    let mut accelerator = CountingHtEncodeAccelerator::default();

    let encoded = encode_precomputed_htj2k_97_with_accelerator(&image, &options, &mut accelerator)
        .expect("precomputed 9/7 encode accepts encode accelerator");

    assert!(encoded.starts_with(&[0xff, 0x4f]));
    assert_eq!(accelerator.forward_dwt53, 0);
    assert_eq!(accelerator.forward_dwt97, 0);
    assert_eq!(accelerator.ht_batches, 1);
    assert!(accelerator.ht_jobs > 0);
    assert_eq!(accelerator.ht_single_blocks, accelerator.ht_jobs);
}

#[test]
fn precomputed_dwt_geometry_validation_rejects_recursive_mismatch_for_both_filters() {
    let mut dwt53 = sample_precomputed_htj2k53_image();
    dwt53.components[0].dwt.levels[0].low_width += 1;
    assert_eq!(
        validate_precomputed_dwt_geometry(&dwt53),
        Err("precomputed DWT recursive geometry mismatch")
    );

    let mut dwt97 = sample_precomputed_htj2k97_image();
    dwt97.components[0].dwt.levels[0].low_width += 1;
    assert_eq!(
        validate_precomputed_dwt97_geometry(&dwt97),
        Err("precomputed DWT recursive geometry mismatch")
    );
}

#[test]
fn prequantized_htj2k97_offers_ht_code_blocks_to_encode_accelerator() {
    let image = sample_precomputed_htj2k97_image();
    let options = EncodeOptions {
        num_decomposition_levels: 1,
        reversible: false,
        guard_bits: 2,
        code_block_width_exp: 2,
        code_block_height_exp: 2,
        ..EncodeOptions::default()
    };
    let prequantized = prequantized_htj2k97_image_from_precomputed_for_test(&image, &options)
        .expect("test prequantized image");
    let mut accelerator = CountingHtEncodeAccelerator::default();

    let encoded =
        encode_prequantized_htj2k_97_with_accelerator(&prequantized, &options, &mut accelerator)
            .expect("prequantized 9/7 encode accepts encode accelerator");

    assert!(encoded.starts_with(&[0xff, 0x4f]));
    assert_eq!(accelerator.forward_dwt53, 0);
    assert_eq!(accelerator.forward_dwt97, 0);
    assert_eq!(accelerator.ht_batches, 1);
    assert!(accelerator.ht_jobs > 0);
    assert_eq!(accelerator.ht_single_blocks, accelerator.ht_jobs);
}

#[test]
fn precomputed_htj2k97_batch_offers_all_ht_code_blocks_in_one_accelerator_call() {
    let images = [
        sample_precomputed_htj2k97_image(),
        sample_precomputed_htj2k97_image(),
    ];
    let options = EncodeOptions {
        num_decomposition_levels: 1,
        reversible: false,
        guard_bits: 2,
        code_block_width_exp: 2,
        code_block_height_exp: 2,
        ..EncodeOptions::default()
    };
    let mut accelerator = CountingHtEncodeAccelerator::default();

    let encoded =
        encode_precomputed_htj2k_97_batch_with_accelerator(&images, &options, &mut accelerator)
            .expect("batch precomputed 9/7 encode accepts encode accelerator");

    assert_eq!(encoded.len(), 2);
    assert!(encoded
        .iter()
        .all(|codestream| codestream.starts_with(&[0xff, 0x4f])));
    assert_eq!(accelerator.forward_dwt53, 0);
    assert_eq!(accelerator.forward_dwt97, 0);
    assert_eq!(accelerator.ht_batches, 1);
    assert!(accelerator.ht_jobs > 0);
    assert_eq!(accelerator.ht_single_blocks, accelerator.ht_jobs);
}

#[derive(Default)]
struct CountingHtEncodeAccelerator {
    deinterleave: usize,
    forward_dwt53: usize,
    forward_dwt97: usize,
    ht_batches: usize,
    ht_jobs: usize,
    ht_single_blocks: usize,
}

impl crate::J2kEncodeStageAccelerator for CountingHtEncodeAccelerator {
    fn encode_deinterleave(
        &mut self,
        _job: crate::J2kDeinterleaveToF32Job<'_>,
    ) -> crate::J2kEncodeStageResult<Option<Vec<Vec<f32>>>> {
        self.deinterleave += 1;
        Ok(None)
    }

    fn encode_forward_dwt53(
        &mut self,
        _job: crate::J2kForwardDwt53Job<'_>,
    ) -> crate::J2kEncodeStageResult<Option<crate::J2kForwardDwt53Output>> {
        self.forward_dwt53 += 1;
        Ok(None)
    }

    fn encode_forward_dwt97(
        &mut self,
        _job: crate::J2kForwardDwt97Job<'_>,
    ) -> crate::J2kEncodeStageResult<Option<crate::J2kForwardDwt97Output>> {
        self.forward_dwt97 += 1;
        Ok(None)
    }

    fn encode_ht_code_blocks(
        &mut self,
        jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
    ) -> crate::J2kEncodeStageResult<Option<Vec<crate::EncodedHtJ2kCodeBlock>>> {
        self.ht_batches += 1;
        self.ht_jobs += jobs.len();
        Ok(None)
    }

    fn encode_ht_code_block(
        &mut self,
        _job: crate::J2kHtCodeBlockEncodeJob<'_>,
    ) -> crate::J2kEncodeStageResult<Option<crate::EncodedHtJ2kCodeBlock>> {
        self.ht_single_blocks += 1;
        Ok(None)
    }
}

#[test]
fn prepare_subband_uses_fused_ht_subband_without_host_quantized_codeblocks() {
    #[derive(Default)]
    #[expect(
        clippy::struct_field_names,
        reason = "the _calls suffix makes each accelerator hook counter explicit"
    )]
    struct FusedHtSubbandAccelerator {
        subband_calls: usize,
        quantize_calls: usize,
        ht_batch_calls: usize,
    }

    impl crate::J2kEncodeStageAccelerator for FusedHtSubbandAccelerator {
        fn encode_ht_subband(
            &mut self,
            job: crate::J2kHtSubbandEncodeJob<'_>,
        ) -> crate::J2kEncodeStageResult<Option<Vec<crate::EncodedHtJ2kCodeBlock>>> {
            self.subband_calls += 1;
            let count = (job.width.div_ceil(job.code_block_width) as usize)
                .checked_mul(job.height.div_ceil(job.code_block_height) as usize)
                .ok_or_else(|| {
                    crate::J2kEncodeStageError::arithmetic_overflow("test code-block count")
                })?;
            Ok(Some(
                (0..count)
                    .map(|idx| crate::EncodedHtJ2kCodeBlock {
                        data: vec![u8::try_from(idx).expect("test block index fits"), 0],
                        cleanup_length: 2,
                        refinement_length: 0,
                        num_coding_passes: 1,
                        num_zero_bitplanes: 0,
                    })
                    .collect(),
            ))
        }

        fn encode_quantize_subband(
            &mut self,
            _job: crate::J2kQuantizeSubbandJob<'_>,
        ) -> crate::J2kEncodeStageResult<Option<Vec<i32>>> {
            self.quantize_calls += 1;
            Ok(None)
        }

        fn encode_ht_code_blocks(
            &mut self,
            _jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
        ) -> crate::J2kEncodeStageResult<Option<Vec<crate::EncodedHtJ2kCodeBlock>>> {
            self.ht_batch_calls += 1;
            Ok(None)
        }
    }

    let coefficients = vec![0.0; 16];
    let mut accelerator = FusedHtSubbandAccelerator::default();
    let prepared = prepare_subband(
        &coefficients,
        4,
        4,
        &QuantStepSize {
            exponent: 8,
            mantissa: 0,
        },
        8,
        2,
        true,
        BlockCodingMode::HighThroughput,
        2,
        2,
        SubBandType::LowLow,
        0,
        &[],
        1,
        1,
        &mut accelerator,
    )
    .expect("fused HT subband prepare");

    assert_eq!(accelerator.subband_calls, 1);
    assert_eq!(accelerator.quantize_calls, 0);
    assert!(prepared.preencoded_ht_code_blocks.is_some());
    assert!(prepared
        .code_blocks
        .iter()
        .all(|block| block.coefficients.is_empty()));

    let precincts = encode_prepared_subbands(vec![prepared], &mut accelerator)
        .expect("preencoded HT subband packet data");

    assert_eq!(accelerator.ht_batch_calls, 0);
    assert_eq!(precincts[0].code_blocks.len(), 4);
    assert_eq!(precincts[0].code_blocks[2].data, vec![2, 0]);
}

#[test]
fn ht_target_coding_passes_tracks_ht_quality_layers() {
    let mut options = EncodeOptions {
        use_ht_block_coding: true,
        reversible: false,
        num_layers: 1,
        ..EncodeOptions::default()
    };

    assert_eq!(
        ht_target_coding_passes_for_options(&options, BlockCodingMode::HighThroughput),
        1
    );

    options.num_layers = 2;
    assert_eq!(
        ht_target_coding_passes_for_options(&options, BlockCodingMode::HighThroughput),
        2
    );

    options.num_layers = 3;
    assert_eq!(
        ht_target_coding_passes_for_options(&options, BlockCodingMode::HighThroughput),
        3
    );

    options.num_layers = 4;
    assert_eq!(
        ht_target_coding_passes_for_options(&options, BlockCodingMode::HighThroughput),
        3
    );

    options.reversible = true;
    assert_eq!(
        ht_target_coding_passes_for_options(&options, BlockCodingMode::HighThroughput),
        3
    );

    options.reversible = false;
    options.use_ht_block_coding = false;
    assert_eq!(
        ht_target_coding_passes_for_options(&options, BlockCodingMode::Classic),
        1
    );
}

#[test]
#[expect(
    clippy::similar_names,
    reason = "PPM and PPT are distinct JPEG 2000 packet-header marker names"
)]
fn packet_header_validation_allows_chunked_ppm_and_ppt_payloads() {
    const MARKER_PAYLOAD_LIMIT: usize = u16::MAX as usize - 3;
    let ppm_headers = vec![vec![0_u8; MARKER_PAYLOAD_LIMIT - 2], vec![1_u8; 1]];
    let ppt_headers = vec![vec![2_u8; MARKER_PAYLOAD_LIMIT + 1]];

    validate_packet_header_marker_payloads(true, false, &[&ppm_headers])
        .expect("chunked PPM payload should validate");
    validate_packet_header_marker_payloads(false, true, &[&ppt_headers])
        .expect("chunked PPT payload should validate");
}

#[test]
fn ht_cpu_fallback_encodes_two_pass_sigprop_refinement() {
    let coefficients: Vec<i32> = (0usize..64 * 64)
        .map(|index| {
            let value = (i32::try_from(((index * 31) ^ (index / 3)) & 0x00ff)
                .expect("masked test coefficient fits i32")
                - 127)
                * 2;
            if index.is_multiple_of(11) {
                0
            } else {
                value
            }
        })
        .collect();
    let jobs = [crate::J2kHtCodeBlockEncodeJob {
        coefficients: &coefficients,
        width: 64,
        height: 64,
        total_bitplanes: 10,
        target_coding_passes: 2,
    }];

    let encoded = encode_all_ht_code_blocks_serial_cpu(&jobs).expect("two-pass CPU HT encode");

    assert_eq!(encoded.len(), 1);
    assert_eq!(encoded[0].num_coding_passes, 2);
    assert_eq!(encoded[0].ht_refinement_length, 48);
    assert_eq!(
        encoded[0].data.len(),
        encoded[0].ht_cleanup_length as usize + encoded[0].ht_refinement_length as usize
    );
    assert!(encoded[0].data[encoded[0].ht_cleanup_length as usize..]
        .iter()
        .all(|byte| *byte == 0));

    let segments = crate::j2c::ht_block_decode::HtCodeBlockSegments::from_combined_payload(
        &encoded[0].data,
        encoded[0].ht_cleanup_length,
        encoded[0].ht_refinement_length,
    )
    .expect("split HT segments");
    let mut decoded = vec![0u32; coefficients.len()];
    crate::j2c::ht_block_decode::decode_segments_validated(
        &segments,
        encoded[0].num_zero_bitplanes,
        10,
        encoded[0].num_coding_passes,
        false,
        true,
        &mut decoded,
        64,
        64,
        64,
    )
    .expect("decode two-pass HT block");
    let decoded_i32 = decoded
        .into_iter()
        .map(|value| crate::j2c::ht_block_decode::coefficient_to_i32(value, 10))
        .collect::<Vec<_>>();
    let max_abs_delta = decoded_i32
        .iter()
        .zip(&coefficients)
        .map(|(actual, expected)| actual.abs_diff(*expected))
        .max()
        .unwrap_or(0);

    assert!(
        max_abs_delta <= 1,
        "two-pass HT sigprop decode must stay within one coefficient LSB"
    );
}

#[test]
fn ht_cpu_fallback_sigprop_refinement_encodes_new_significance_bits() {
    let mut coefficients = vec![0_i32; 8 * 8];
    for row in 0..8 {
        coefficients[row * 8] = 3;
        coefficients[row * 8 + 1] = 1;
        coefficients[row * 8 + 2] = -1;
    }
    let jobs = [crate::J2kHtCodeBlockEncodeJob {
        coefficients: &coefficients,
        width: 8,
        height: 8,
        total_bitplanes: 4,
        target_coding_passes: 2,
    }];

    let encoded = encode_all_ht_code_blocks_serial_cpu(&jobs).expect("two-pass CPU HT encode");

    assert_eq!(encoded[0].num_coding_passes, 2);
    assert!(encoded[0].ht_refinement_length > 0);
    assert!(
        encoded[0].data[encoded[0].ht_cleanup_length as usize..]
            .iter()
            .any(|byte| *byte != 0),
        "sigprop refinement should encode new significance/sign bits"
    );

    let segments = crate::j2c::ht_block_decode::HtCodeBlockSegments::from_combined_payload(
        &encoded[0].data,
        encoded[0].ht_cleanup_length,
        encoded[0].ht_refinement_length,
    )
    .expect("split HT segments");
    let mut decoded = vec![0u32; coefficients.len()];
    crate::j2c::ht_block_decode::decode_segments_validated(
        &segments,
        encoded[0].num_zero_bitplanes,
        4,
        encoded[0].num_coding_passes,
        false,
        true,
        &mut decoded,
        8,
        8,
        8,
    )
    .expect("decode two-pass HT block");
    let decoded_i32 = decoded
        .into_iter()
        .map(|value| crate::j2c::ht_block_decode::coefficient_to_i32(value, 4))
        .collect::<Vec<_>>();

    assert_eq!(decoded_i32, coefficients);
}

#[test]
fn ht_cpu_fallback_encodes_three_pass_magref_refinement() {
    let mut coefficients = vec![0_i32; 8 * 8];
    for row in 0..8 {
        let base = row * 8;
        coefficients[base] = 2;
        coefficients[base + 1] = 3;
        coefficients[base + 2] = 1;
        coefficients[base + 3] = -1;
        coefficients[base + 4] = -2;
        coefficients[base + 5] = -3;
    }
    let jobs = [crate::J2kHtCodeBlockEncodeJob {
        coefficients: &coefficients,
        width: 8,
        height: 8,
        total_bitplanes: 4,
        target_coding_passes: 3,
    }];

    let encoded = encode_all_ht_code_blocks_serial_cpu(&jobs).expect("three-pass CPU HT encode");

    assert_eq!(encoded[0].num_coding_passes, 3);
    assert!(encoded[0].ht_refinement_length > 0);

    let segments = crate::j2c::ht_block_decode::HtCodeBlockSegments::from_combined_payload(
        &encoded[0].data,
        encoded[0].ht_cleanup_length,
        encoded[0].ht_refinement_length,
    )
    .expect("split HT segments");
    let mut decoded = vec![0u32; coefficients.len()];
    crate::j2c::ht_block_decode::decode_segments_validated(
        &segments,
        encoded[0].num_zero_bitplanes,
        4,
        encoded[0].num_coding_passes,
        false,
        true,
        &mut decoded,
        8,
        8,
        8,
    )
    .expect("decode three-pass HT block");
    let decoded_i32 = decoded
        .into_iter()
        .map(|value| crate::j2c::ht_block_decode::coefficient_to_i32(value, 4))
        .collect::<Vec<_>>();

    assert_eq!(decoded_i32, coefficients);
}

#[test]
fn ht_cpu_fallback_rejects_unsupported_refinement_pass_count() {
    let coefficients = vec![1_i32; 64 * 64];
    let jobs = [crate::J2kHtCodeBlockEncodeJob {
        coefficients: &coefficients,
        width: 64,
        height: 64,
        total_bitplanes: 2,
        target_coding_passes: 4,
    }];

    let err = encode_all_ht_code_blocks_serial_cpu(&jobs)
        .expect_err("CPU HT encode must reject unsupported pass requests");

    assert!(err.contains("at most three HT coding passes"));
}

#[test]
fn ht_cpu_parallel_fallback_threshold_matches_parallel_output() {
    assert_eq!(HT_CPU_PARALLEL_FALLBACK_MIN_JOBS, 4);

    let blocks: Vec<Vec<i32>> = (0..HT_CPU_PARALLEL_FALLBACK_MIN_JOBS)
        .map(|seed| {
            (0usize..64 * 64)
                .map(|index| {
                    let value = i32::try_from(((index * 31) ^ (seed * 17)) & 0x01ff)
                        .expect("masked test coefficient fits i32")
                        - 255;
                    if (index + seed).is_multiple_of(11) {
                        0
                    } else {
                        value
                    }
                })
                .collect()
        })
        .collect();
    let jobs: Vec<_> = blocks
        .iter()
        .map(|coefficients| crate::J2kHtCodeBlockEncodeJob {
            coefficients,
            width: 64,
            height: 64,
            total_bitplanes: 10,
            target_coding_passes: 1,
        })
        .collect();

    let serial =
        encode_all_ht_code_blocks_serial_cpu(&jobs[..HT_CPU_PARALLEL_FALLBACK_MIN_JOBS - 1])
            .expect("serial tiny HT encode");
    let parallel = encode_all_ht_code_blocks_parallel(&jobs[..HT_CPU_PARALLEL_FALLBACK_MIN_JOBS])
        .expect("parallel HT encode");
    let serial_threshold =
        encode_all_ht_code_blocks_serial_cpu(&jobs[..HT_CPU_PARALLEL_FALLBACK_MIN_JOBS])
            .expect("serial threshold HT encode");

    assert_eq!(serial.len(), HT_CPU_PARALLEL_FALLBACK_MIN_JOBS - 1);
    assert_eq!(parallel.len(), HT_CPU_PARALLEL_FALLBACK_MIN_JOBS);
    assert_eq!(serial_threshold.len(), parallel.len());
    for (serial, parallel) in serial_threshold.iter().zip(&parallel) {
        assert_eq!(serial.data, parallel.data);
        assert_eq!(serial.num_coding_passes, parallel.num_coding_passes);
        assert_eq!(serial.num_zero_bitplanes, parallel.num_zero_bitplanes);
    }
}

#[test]
fn code_block_extraction_copies_partial_edge_blocks_rowwise() {
    let quantized: Vec<i32> = (0..20).collect();

    let block = copy_code_block_coefficients(&quantized, 5, 3, 1, 2, 3);

    assert_eq!(block, vec![8, 9, 13, 14, 18, 19]);
}

#[test]
fn test_encode_lossy() {
    let pixels: Vec<u8> = (0..64).collect();

    let result = encode(
        &pixels,
        8,
        8,
        1,
        8,
        false,
        &EncodeOptions {
            num_decomposition_levels: 2,
            reversible: false,
            guard_bits: 2,
            ..Default::default()
        },
    );

    assert!(result.is_ok());
}

#[test]
fn prequantized_htj2k97_matches_precomputed_dwt97_codestream() {
    let image = sample_precomputed_htj2k97_image();
    let options = EncodeOptions {
        num_decomposition_levels: 1,
        reversible: false,
        guard_bits: 2,
        code_block_width_exp: 2,
        code_block_height_exp: 2,
        ..EncodeOptions::default()
    };

    let precomputed =
        encode_precomputed_htj2k_97(&image, &options).expect("precomputed DWT encode");
    let prequantized = prequantized_htj2k97_image_from_precomputed_for_test(&image, &options)
        .expect("test prequantized image");
    let direct =
        encode_prequantized_htj2k_97(&prequantized, &options).expect("prequantized encode");

    assert_eq!(direct, precomputed);
}

#[test]
fn preencoded_htj2k97_matches_prequantized_codestream() {
    let image = sample_precomputed_htj2k97_image();
    let options = EncodeOptions {
        num_decomposition_levels: 1,
        reversible: false,
        guard_bits: 2,
        code_block_width_exp: 2,
        code_block_height_exp: 2,
        ..EncodeOptions::default()
    };
    let prequantized = prequantized_htj2k97_image_from_precomputed_for_test(&image, &options)
        .expect("test prequantized image");
    let expected =
        encode_prequantized_htj2k_97(&prequantized, &options).expect("prequantized encode");
    let preencoded = preencoded_htj2k97_image_from_prequantized_for_test(&prequantized)
        .expect("test preencoded image");
    let actual = encode_preencoded_htj2k_97(&preencoded, &options).expect("preencoded encode");

    assert_eq!(actual, expected);
}

#[test]
fn preencoded_htj2k97_preserves_refinement_segments_in_packet_body() {
    let options = EncodeOptions {
        num_decomposition_levels: 0,
        reversible: false,
        guard_bits: 2,
        code_block_width_exp: 2,
        code_block_height_exp: 2,
        ..EncodeOptions::default()
    };
    let guard_bits = options.guard_bits.max(2);
    let step_sizes = quantize::compute_step_sizes_with_irreversible_profile(
        8,
        0,
        false,
        guard_bits,
        options.irreversible_quantization_scale,
        options.irreversible_quantization_subband_scales,
    );
    let total_bitplanes = guard_bits
        .saturating_add(
            u8::try_from(step_sizes[0].exponent).expect("test exponent fits supported u8 range"),
        )
        .saturating_sub(1);
    let payload = [0x12, 0x34, 0x56, 0x78];
    let image = PreencodedHtj2k97Image {
        width: 1,
        height: 1,
        bit_depth: 8,
        signed: false,
        components: vec![PreencodedHtj2k97Component {
            x_rsiz: 1,
            y_rsiz: 1,
            resolutions: vec![PreencodedHtj2k97Resolution {
                subbands: vec![PreencodedHtj2k97Subband {
                    sub_band_type: J2kSubBandType::LowLow,
                    num_cbs_x: 1,
                    num_cbs_y: 1,
                    total_bitplanes,
                    code_blocks: vec![PreencodedHtj2k97CodeBlock {
                        width: 1,
                        height: 1,
                        encoded: EncodedHtJ2kCodeBlock {
                            data: payload.to_vec(),
                            cleanup_length: 2,
                            refinement_length: 2,
                            num_coding_passes: 3,
                            num_zero_bitplanes: 0,
                        },
                    }],
                }],
            }],
        }],
    };

    let codestream =
        encode_preencoded_htj2k_97(&image, &options).expect("preencoded refinement encode");
    let eoc = codestream
        .windows(2)
        .rposition(|marker| marker == [0xff, crate::j2c::codestream::markers::EOC])
        .expect("EOC marker");

    assert_eq!(&codestream[eoc - payload.len()..eoc], payload);
}

#[test]
fn preencoded_htj2k97_rejects_empty_block_with_wrong_zero_bitplanes() {
    let (mut image, options) = sample_preencoded_htj2k97_for_test();
    let block = &mut image.components[0].resolutions[0].subbands[0].code_blocks[0];
    block.encoded = EncodedHtJ2kCodeBlock {
        data: Vec::new(),
        cleanup_length: 0,
        refinement_length: 0,
        num_coding_passes: 0,
        num_zero_bitplanes: 0,
    };

    let error = encode_preencoded_htj2k_97(&image, &options)
        .expect_err("invalid all-zero block metadata must be rejected");

    assert_eq!(
        error,
        EncodeError::InvalidInput {
            what: "empty HTJ2K code-block zero-bitplane count mismatch",
        }
    );
}

#[test]
fn preencoded_htj2k97_rejects_coded_block_with_too_many_zero_bitplanes() {
    let (mut image, options) = sample_preencoded_htj2k97_for_test();
    let subband = &mut image.components[0].resolutions[0].subbands[0];
    subband.code_blocks[0].encoded.num_zero_bitplanes = subband.total_bitplanes;

    let error = encode_preencoded_htj2k_97(&image, &options)
        .expect_err("coded block with no coded bitplanes must be rejected");

    assert_eq!(
        error,
        EncodeError::InvalidInput {
            what: "HTJ2K code-block zero-bitplane count out of range",
        }
    );
}

#[cfg(feature = "std")]
#[test]
fn preencoded_htj2k97_rejects_too_many_coding_passes_without_panic() {
    let (mut image, options) = sample_preencoded_htj2k97_for_test();
    image.components[0].resolutions[0].subbands[0].code_blocks[0]
        .encoded
        .num_coding_passes = 165;

    let result = std::panic::catch_unwind(|| encode_preencoded_htj2k_97(&image, &options));

    assert!(result.is_ok(), "invalid coding pass count must not panic");
    assert_eq!(
        result.expect("catch_unwind returned checked result"),
        Err(EncodeError::InvalidInput {
            what: "HTJ2K code-block coding pass count out of range",
        })
    );
}

#[test]
fn prequantized_htj2k97_accepts_empty_high_subbands() {
    let options = EncodeOptions {
        num_decomposition_levels: 1,
        reversible: false,
        guard_bits: 2,
        code_block_width_exp: 2,
        code_block_height_exp: 2,
        ..EncodeOptions::default()
    };
    let image = PrequantizedHtj2k97Image {
        width: 1,
        height: 1,
        bit_depth: 8,
        signed: false,
        components: vec![PrequantizedHtj2k97Component {
            x_rsiz: 1,
            y_rsiz: 1,
            resolutions: vec![
                PrequantizedHtj2k97Resolution {
                    subbands: vec![PrequantizedHtj2k97Subband {
                        sub_band_type: J2kSubBandType::LowLow,
                        num_cbs_x: 1,
                        num_cbs_y: 1,
                        total_bitplanes: 11,
                        code_blocks: vec![PrequantizedHtj2k97CodeBlock {
                            coefficients: vec![0],
                            width: 1,
                            height: 1,
                        }],
                    }],
                },
                PrequantizedHtj2k97Resolution {
                    subbands: vec![
                        empty_prequantized_subband(J2kSubBandType::HighLow),
                        empty_prequantized_subband(J2kSubBandType::LowHigh),
                        empty_prequantized_subband(J2kSubBandType::HighHigh),
                    ],
                },
            ],
        }],
    };

    let encoded =
        encode_prequantized_htj2k_97(&image, &options).expect("empty high subbands encode");

    assert!(encoded.starts_with(&[0xff, 0x4f]));
}

fn empty_prequantized_subband(sub_band_type: J2kSubBandType) -> PrequantizedHtj2k97Subband {
    PrequantizedHtj2k97Subband {
        sub_band_type,
        num_cbs_x: 0,
        num_cbs_y: 0,
        total_bitplanes: 0,
        code_blocks: Vec::new(),
    }
}

fn sample_precomputed_htj2k97_image() -> PrecomputedHtj2k97Image {
    let width = 17u32;
    let height = 13u32;
    let low_width = width.div_ceil(2);
    let low_height = height.div_ceil(2);
    let high_width = width / 2;
    let high_height = height / 2;

    PrecomputedHtj2k97Image {
        width,
        height,
        bit_depth: 8,
        signed: false,
        components: vec![PrecomputedHtj2k97Component {
            x_rsiz: 1,
            y_rsiz: 1,
            dwt: J2kForwardDwt97Output {
                ll: sample_f32_coefficients(low_width * low_height, 0.25),
                ll_width: low_width,
                ll_height: low_height,
                levels: vec![J2kForwardDwt97Level {
                    hl: sample_f32_coefficients(high_width * low_height, -0.75),
                    lh: sample_f32_coefficients(low_width * high_height, 1.25),
                    hh: sample_f32_coefficients(high_width * high_height, -1.5),
                    width,
                    height,
                    low_width,
                    low_height,
                    high_width,
                    high_height,
                }],
            },
        }],
    }
}

fn sample_precomputed_htj2k53_image() -> PrecomputedHtj2k53Image {
    let width = 17u32;
    let height = 13u32;
    let low_width = width.div_ceil(2);
    let low_height = height.div_ceil(2);
    let high_width = width / 2;
    let high_height = height / 2;

    PrecomputedHtj2k53Image {
        width,
        height,
        bit_depth: 8,
        signed: false,
        components: vec![PrecomputedHtj2k53Component {
            x_rsiz: 1,
            y_rsiz: 1,
            dwt: J2kForwardDwt53Output {
                ll: sample_f32_coefficients(low_width * low_height, 0.0),
                ll_width: low_width,
                ll_height: low_height,
                levels: vec![J2kForwardDwt53Level {
                    hl: sample_f32_coefficients(high_width * low_height, -2.0),
                    lh: sample_f32_coefficients(low_width * high_height, 2.0),
                    hh: sample_f32_coefficients(high_width * high_height, -4.0),
                    width,
                    height,
                    low_width,
                    low_height,
                    high_width,
                    high_height,
                }],
            },
        }],
    }
}

fn sample_f32_coefficients(len: u32, offset: f32) -> Vec<f32> {
    (0..len)
        .map(|idx| {
            (f32::from(u8::try_from(idx % 17).expect("test coefficient fits u8")) - 8.0) * 0.5
                + offset
        })
        .collect()
}

fn prequantized_htj2k97_image_from_precomputed_for_test(
    image: &PrecomputedHtj2k97Image,
    options: &EncodeOptions,
) -> crate::EncodeResult<PrequantizedHtj2k97Image> {
    let guard_bits = options.guard_bits.max(2);
    let step_sizes = quantize::compute_step_sizes_with_irreversible_profile(
        image.bit_depth,
        1,
        false,
        guard_bits,
        options.irreversible_quantization_scale,
        options.irreversible_quantization_subband_scales,
    );
    let cb_width = 1u32 << (options.code_block_width_exp + 2);
    let cb_height = 1u32 << (options.code_block_height_exp + 2);
    let subband = |coefficients, width, height, sub_band_type, step_size: &QuantStepSize| {
        prequantized_subband_for_test(PrequantizedSubbandForTest {
            coefficients,
            width,
            height,
            sub_band_type,
            step_size,
            bit_depth: image.bit_depth,
            guard_bits,
            cb_width,
            cb_height,
        })
    };

    let components = image
        .components
        .iter()
        .map(|component| {
            let mut resolutions = Vec::with_capacity(component.dwt.levels.len() + 1);
            resolutions.push(PrequantizedHtj2k97Resolution {
                subbands: vec![subband(
                    &component.dwt.ll,
                    component.dwt.ll_width,
                    component.dwt.ll_height,
                    SubBandType::LowLow,
                    &step_sizes[0],
                )?],
            });

            for (level_index, level) in component.dwt.levels.iter().enumerate() {
                let step_base = 1 + level_index * 3;
                resolutions.push(PrequantizedHtj2k97Resolution {
                    subbands: vec![
                        subband(
                            &level.hl,
                            level.high_width,
                            level.low_height,
                            SubBandType::HighLow,
                            &step_sizes[step_base],
                        )?,
                        subband(
                            &level.lh,
                            level.low_width,
                            level.high_height,
                            SubBandType::LowHigh,
                            &step_sizes[step_base + 1],
                        )?,
                        subband(
                            &level.hh,
                            level.high_width,
                            level.high_height,
                            SubBandType::HighHigh,
                            &step_sizes[step_base + 2],
                        )?,
                    ],
                });
            }

            Ok(PrequantizedHtj2k97Component {
                x_rsiz: component.x_rsiz,
                y_rsiz: component.y_rsiz,
                resolutions,
            })
        })
        .collect::<crate::EncodeResult<Vec<_>>>()?;

    Ok(PrequantizedHtj2k97Image {
        width: image.width,
        height: image.height,
        bit_depth: image.bit_depth,
        signed: image.signed,
        components,
    })
}

#[derive(Clone, Copy)]
struct PrequantizedSubbandForTest<'a> {
    coefficients: &'a [f32],
    width: u32,
    height: u32,
    sub_band_type: SubBandType,
    step_size: &'a QuantStepSize,
    bit_depth: u8,
    guard_bits: u8,
    cb_width: u32,
    cb_height: u32,
}

fn prequantized_subband_for_test(
    request: PrequantizedSubbandForTest<'_>,
) -> crate::EncodeResult<PrequantizedHtj2k97Subband> {
    let PrequantizedSubbandForTest {
        coefficients,
        width,
        height,
        sub_band_type,
        step_size,
        bit_depth,
        guard_bits,
        cb_width,
        cb_height,
    } = request;
    let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    let prepared = prepare_subband(
        coefficients,
        width,
        height,
        step_size,
        bit_depth,
        guard_bits,
        false,
        BlockCodingMode::HighThroughput,
        cb_width,
        cb_height,
        sub_band_type,
        0,
        &[],
        1,
        1,
        &mut accelerator,
    )?;

    Ok(PrequantizedHtj2k97Subband {
        sub_band_type: public_sub_band_type(sub_band_type),
        num_cbs_x: prepared.num_cbs_x,
        num_cbs_y: prepared.num_cbs_y,
        total_bitplanes: prepared.total_bitplanes,
        code_blocks: prepared
            .code_blocks
            .into_iter()
            .map(|block| {
                let coefficients = match block.coefficients {
                    PreparedCodeBlockCoefficients::I32(values) => values,
                    PreparedCodeBlockCoefficients::I64(values) => {
                        downcast_i64_coefficients_to_i32(&values)
                            .map_err(|what| EncodeError::Unsupported { what })?
                    }
                    PreparedCodeBlockCoefficients::Empty => Vec::new(),
                };
                Ok(PrequantizedHtj2k97CodeBlock {
                    coefficients,
                    width: block.width,
                    height: block.height,
                })
            })
            .collect::<crate::EncodeResult<Vec<_>>>()?,
    })
}

fn preencoded_htj2k97_image_from_prequantized_for_test(
    image: &PrequantizedHtj2k97Image,
) -> Result<PreencodedHtj2k97Image, &'static str> {
    let components = image
        .components
        .iter()
        .map(|component| {
            Ok(PreencodedHtj2k97Component {
                x_rsiz: component.x_rsiz,
                y_rsiz: component.y_rsiz,
                resolutions: component
                    .resolutions
                    .iter()
                    .map(|resolution| {
                        Ok(PreencodedHtj2k97Resolution {
                            subbands: resolution
                                .subbands
                                .iter()
                                .map(preencoded_subband_from_prequantized_for_test)
                                .collect::<Result<Vec<_>, &'static str>>()?,
                        })
                    })
                    .collect::<Result<Vec<_>, &'static str>>()?,
            })
        })
        .collect::<Result<Vec<_>, &'static str>>()?;

    Ok(PreencodedHtj2k97Image {
        width: image.width,
        height: image.height,
        bit_depth: image.bit_depth,
        signed: image.signed,
        components,
    })
}

fn sample_preencoded_htj2k97_for_test() -> (PreencodedHtj2k97Image, EncodeOptions) {
    let image = sample_precomputed_htj2k97_image();
    let options = EncodeOptions {
        num_decomposition_levels: 1,
        reversible: false,
        guard_bits: 2,
        code_block_width_exp: 2,
        code_block_height_exp: 2,
        ..EncodeOptions::default()
    };
    let prequantized = prequantized_htj2k97_image_from_precomputed_for_test(&image, &options)
        .expect("test prequantized image");
    let preencoded = preencoded_htj2k97_image_from_prequantized_for_test(&prequantized)
        .expect("test preencoded image");
    (preencoded, options)
}

fn preencoded_subband_from_prequantized_for_test(
    subband: &PrequantizedHtj2k97Subband,
) -> Result<PreencodedHtj2k97Subband, &'static str> {
    let code_blocks = subband
        .code_blocks
        .iter()
        .map(|block| {
            let encoded = ht_block_encode::encode_code_block(
                &block.coefficients,
                block.width,
                block.height,
                subband.total_bitplanes,
            )?;
            Ok(PreencodedHtj2k97CodeBlock {
                width: block.width,
                height: block.height,
                encoded: EncodedHtJ2kCodeBlock {
                    data: encoded.data,
                    cleanup_length: encoded.ht_cleanup_length,
                    refinement_length: encoded.ht_refinement_length,
                    num_coding_passes: encoded.num_coding_passes,
                    num_zero_bitplanes: encoded.num_zero_bitplanes,
                },
            })
        })
        .collect::<Result<Vec<_>, &'static str>>()?;

    Ok(PreencodedHtj2k97Subband {
        sub_band_type: subband.sub_band_type,
        num_cbs_x: subband.num_cbs_x,
        num_cbs_y: subband.num_cbs_y,
        total_bitplanes: subband.total_bitplanes,
        code_blocks,
    })
}

fn assert_htj2k_lossless_roundtrip(
    pixels: &[u8],
    width: u32,
    height: u32,
    bit_depth: u8,
    num_decomposition_levels: u8,
) {
    let codestream = encode_htj2k(
        pixels,
        width,
        height,
        1,
        bit_depth,
        false,
        &EncodeOptions {
            num_decomposition_levels,
            ..Default::default()
        },
    )
    .expect("HTJ2K encode");

    assert!(codestream.windows(2).any(|window| window == [0xFF, 0x50]));
    let cod_offset = codestream
        .windows(2)
        .position(|window| window == [0xFF, 0x52])
        .expect("COD marker");
    assert_eq!(codestream[cod_offset + 12], 0x40);

    let image = Image::new(
        &codestream,
        &DecodeSettings {
            resolve_palette_indices: true,
            strict: true,
            target_resolution: None,
        },
    )
    .expect("parse HT codestream");
    let decoded = image.decode_native().expect("decode HT codestream");

    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);
    assert_eq!(decoded.bit_depth, bit_depth);
    assert_eq!(decoded.data, pixels);
}

fn gradient_u8(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            pixels
                .push(u8::try_from((x * 17 + y * 31) % 256).expect("test gradient sample fits u8"));
        }
    }
    pixels
}

fn lossy_htj2k_roundtrip_u8(
    pixels: &[u8],
    width: u32,
    height: u32,
    num_decomposition_levels: u8,
) -> (Vec<u8>, usize) {
    let codestream = encode_htj2k(
        pixels,
        width,
        height,
        1,
        8,
        false,
        &EncodeOptions {
            num_decomposition_levels,
            reversible: false,
            guard_bits: 2,
            ..Default::default()
        },
    )
    .expect("lossy HT encode");

    assert!(codestream.windows(2).any(|window| window == [0xFF, 0x50]));

    let image = Image::new(
        &codestream,
        &DecodeSettings {
            resolve_palette_indices: true,
            strict: true,
            target_resolution: None,
        },
    )
    .expect("parse lossy HT codestream");
    let decoded = image.decode_native().expect("decode lossy HT codestream");

    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);
    assert_eq!(decoded.bit_depth, 8);

    (decoded.data, codestream.len())
}

fn max_abs_error(expected: &[u8], actual: &[u8]) -> u8 {
    expected
        .iter()
        .zip(actual)
        .map(|(&expected, &actual)| expected.abs_diff(actual))
        .max()
        .unwrap_or(0)
}

fn psnr_db(expected: &[u8], actual: &[u8]) -> f64 {
    let sample_count = u32::try_from(expected.len()).expect("test image sample count fits in u32");
    let mse = expected
        .iter()
        .zip(actual)
        .map(|(&expected, &actual)| {
            let diff = f64::from(expected) - f64::from(actual);
            diff * diff
        })
        .sum::<f64>()
        / f64::from(sample_count);

    if mse == 0.0 {
        f64::INFINITY
    } else {
        20.0 * 255.0f64.log10() - 10.0 * mse.log10()
    }
}

fn assert_not_flat_128(decoded: &[u8]) {
    assert!(
        decoded.iter().any(|&sample| sample != 128),
        "lossy decode collapsed to flat 128"
    );
}

#[test]
fn test_encode_high_throughput_zero_image_roundtrip() {
    let width = 4u32;
    let height = 4u32;
    let sample = 2048u16.to_le_bytes();
    let mut pixels = Vec::with_capacity((width * height * 2) as usize);
    for _ in 0..(width * height) {
        pixels.extend_from_slice(&sample);
    }

    let codestream = encode(
        &pixels,
        width,
        height,
        1,
        12,
        false,
        &EncodeOptions {
            num_decomposition_levels: 2,
            use_ht_block_coding: true,
            ..Default::default()
        },
    )
    .expect("HT all-zero encode");

    assert!(codestream.windows(2).any(|window| window == [0xFF, 0x50]));
    let cod_offset = codestream
        .windows(2)
        .position(|window| window == [0xFF, 0x52])
        .expect("COD marker");
    assert_eq!(codestream[cod_offset + 12], 0x40);

    let image = Image::new(&codestream, &DecodeSettings::default()).expect("parse HT codestream");
    let decoded = image.decode_native().expect("decode HT codestream");

    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);
    assert_eq!(decoded.bit_depth, 12);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn test_encode_high_throughput_nonzero_roundtrip() {
    let width = 1u32;
    let height = 1u32;
    let pixels = 2049u16.to_le_bytes().to_vec();

    let codestream = encode_htj2k(
        &pixels,
        width,
        height,
        1,
        12,
        false,
        &EncodeOptions {
            num_decomposition_levels: 0,
            ..Default::default()
        },
    )
    .expect("HT non-zero encode");

    assert!(codestream.windows(2).any(|window| window == [0xFF, 0x50]));
    let image = Image::new(&codestream, &DecodeSettings::default()).expect("parse HT codestream");
    let decoded = image.decode_native().expect("decode HT codestream");

    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);
    assert_eq!(decoded.bit_depth, 12);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn test_encode_high_throughput_varied_12bit_roundtrip() {
    let mut pixels = Vec::with_capacity(32);
    for i in 0u16..16 {
        pixels.extend_from_slice(&((i * 257) & 0x0FFF).to_le_bytes());
    }

    let codestream = encode_htj2k(
        &pixels,
        4,
        4,
        1,
        12,
        false,
        &EncodeOptions {
            num_decomposition_levels: 1,
            ..Default::default()
        },
    )
    .expect("HT varied encode");

    let image = Image::new(&codestream, &DecodeSettings::default()).expect("parse HT codestream");
    let decoded = image.decode_native().expect("decode HT codestream");

    assert_eq!(decoded.width, 4);
    assert_eq!(decoded.height, 4);
    assert_eq!(decoded.bit_depth, 12);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn test_encode_high_throughput_gradient_8bit_roundtrip() {
    let pixels: Vec<u8> = (0..64).collect();

    let codestream = encode_htj2k(
        &pixels,
        8,
        8,
        1,
        8,
        false,
        &EncodeOptions {
            num_decomposition_levels: 3,
            ..Default::default()
        },
    )
    .expect("HT gradient encode");

    let image = Image::new(&codestream, &DecodeSettings::default()).expect("parse HT codestream");
    let decoded = image.decode_native().expect("decode HT codestream");

    assert_eq!(decoded.width, 8);
    assert_eq!(decoded.height, 8);
    assert_eq!(decoded.bit_depth, 8);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn test_encode_high_throughput_varied_12bit_large_roundtrip() {
    let width = 16u32;
    let height = 8u32;
    let mut pixels = Vec::with_capacity((width * height * 2) as usize);
    for y in 0u16..u16::try_from(height).expect("test height fits u16") {
        for x in 0u16..u16::try_from(width).expect("test width fits u16") {
            let value = (x * 257 + y * 17) & 0x0FFF;
            pixels.extend_from_slice(&value.to_le_bytes());
        }
    }

    assert_htj2k_lossless_roundtrip(&pixels, width, height, 12, 4);
}

#[test]
fn test_encode_high_throughput_ramp_16bit_roundtrip() {
    let width = 48u32;
    let height = 24u32;
    let mut pixels = Vec::with_capacity((width * height * 2) as usize);
    for y in 0u16..u16::try_from(height).expect("test height fits u16") {
        for x in 0u16..u16::try_from(width).expect("test width fits u16") {
            let value = x * 521 + y * 997;
            pixels.extend_from_slice(&value.to_le_bytes());
        }
    }

    assert_htj2k_lossless_roundtrip(&pixels, width, height, 16, 4);
}

#[test]
fn test_encode_high_throughput_lossy_large_gradient_is_parseable() {
    let pixels = gradient_u8(128, 128);

    let (decoded, codestream_len) = lossy_htj2k_roundtrip_u8(&pixels, 128, 128, 5);

    assert!(codestream_len > 110);
    assert_not_flat_128(&decoded);
    assert!(
        psnr_db(&pixels, &decoded) >= 30.0,
        "psnr={} max_abs={}",
        psnr_db(&pixels, &decoded),
        max_abs_error(&pixels, &decoded)
    );
}

#[test]
fn test_encode_high_throughput_lossy_constant_extremes_are_not_midgray() {
    for sample in [0u8, 255] {
        let pixels = vec![sample; 64 * 64];
        let (decoded, codestream_len) = lossy_htj2k_roundtrip_u8(&pixels, 64, 64, 4);

        assert!(codestream_len > 110);
        assert_not_flat_128(&decoded);
        assert!(
            max_abs_error(&pixels, &decoded) <= 2,
            "sample={sample} max_abs={} decoded_min={} decoded_max={}",
            max_abs_error(&pixels, &decoded),
            decoded.iter().min().unwrap(),
            decoded.iter().max().unwrap()
        );
    }
}

#[test]
fn test_encode_invalid_dimensions() {
    let result = encode(&[], 0, 0, 1, 8, false, &EncodeOptions::default());
    assert!(result.is_err());
}

#[test]
fn test_encode_too_short() {
    let pixels = vec![0u8; 10]; // Way too short for 8x8
    let result = encode(&pixels, 8, 8, 1, 8, false, &EncodeOptions::default());
    assert!(result.is_err());
}

#[test]
fn test_deinterleave_rgb() {
    let pixels = vec![
        10u8, 20, 30, // pixel 0: R=10, G=20, B=30
        40, 50, 60, // pixel 1: R=40, G=50, B=60
    ];
    let comps = deinterleave_to_f32(&pixels, 2, 3, 8, false);
    assert_eq!(comps[0], vec![-118.0, -88.0]); // R
    assert_eq!(comps[1], vec![-108.0, -78.0]); // G
    assert_eq!(comps[2], vec![-98.0, -68.0]); // B
}

#[test]
fn deinterleave_rgb8_unsigned_fast_path_matches_generic_output() {
    let pixels = (0..96)
        .map(|value| {
            u8::try_from((value * 19 + value / 3) & 0xff).expect("masked test pixel fits u8")
        })
        .collect::<Vec<_>>();

    let expected = deinterleave_to_f32(&pixels, 32, 3, 8, false);
    let actual = deinterleave_rgb8_unsigned_to_f32(&pixels, 32);

    assert_eq!(actual, expected);
}

#[test]
fn test_encode_decode_roundtrip_gray_8bit() {
    use crate::{DecodeSettings, Image};

    // Constant image: all pixels = 42 — simplest possible test
    let original: Vec<u8> = vec![42u8; 64]; // 8x8, all same value
    let encoded = encode(
        &original,
        8,
        8,
        1,
        8,
        false,
        &EncodeOptions {
            num_decomposition_levels: 0,
            reversible: true,
            ..Default::default()
        },
    )
    .expect("encode failed");

    let settings = DecodeSettings {
        resolve_palette_indices: false,
        strict: false,
        target_resolution: None,
    };
    let image = Image::new(&encoded, &settings).expect("parse failed");
    let decoded = image.decode_native().expect("decode failed");

    assert_eq!(decoded.width, 8);
    assert_eq!(decoded.height, 8);
    assert_eq!(decoded.data, original, "round-trip mismatch");
}

#[test]
fn test_encode_decode_roundtrip_gray_8bit_single_dwt_level() {
    use crate::{DecodeSettings, Image};

    let original: Vec<u8> = (0..64 * 64)
        .map(|value| {
            u8::try_from((value * 37 + value / 7) & 0xFF).expect("masked test pixel fits u8")
        })
        .collect();
    let encoded = encode(
        &original,
        64,
        64,
        1,
        8,
        false,
        &EncodeOptions {
            num_decomposition_levels: 1,
            reversible: true,
            ..Default::default()
        },
    )
    .expect("encode failed");

    let image = Image::new(&encoded, &DecodeSettings::default()).expect("parse failed");
    let decoded = image.decode_native().expect("decode failed");

    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.height, 64);
    assert_eq!(decoded.data, original, "round-trip mismatch");
}

/// Precondition gate: native `encode_htj2k` must produce byte-identical output
/// across repeated invocations with the same input before CUDA parity can be
/// asserted.  96x80 with 3 components and 5 decomposition levels exercises
/// multi-codeblock subbands.
#[cfg(feature = "std")]
#[test]
fn encode_htj2k_is_byte_deterministic() {
    const WIDTH: u32 = 96;
    const HEIGHT: u32 = 80;
    const NUM_COMPONENTS: u8 = 3;
    const BIT_DEPTH: u8 = 8;
    const REPETITIONS: usize = 8;

    // Deterministic pseudo-random pixel data: simple LCG-like sequence.
    let pixel_count = (WIDTH * HEIGHT) as usize * usize::from(NUM_COMPONENTS);
    let pixels: Vec<u8> = (0..pixel_count)
        .map(|i| {
            let v = i
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            u8::try_from(v >> 56).expect("shifted test PRNG value fits u8")
        })
        .collect();

    let options = EncodeOptions {
        use_ht_block_coding: true,
        reversible: true,
        num_decomposition_levels: 5,
        validate_high_throughput_codestream: true,
        ..EncodeOptions::default()
    };

    let baseline = encode_htj2k(
        &pixels,
        WIDTH,
        HEIGHT,
        NUM_COMPONENTS.into(),
        BIT_DEPTH,
        false,
        &options,
    )
    .expect("encode_htj2k baseline failed");

    assert!(
        !baseline.is_empty(),
        "baseline codestream must not be empty"
    );

    for i in 0..REPETITIONS {
        let result = encode_htj2k(
            &pixels,
            WIDTH,
            HEIGHT,
            NUM_COMPONENTS.into(),
            BIT_DEPTH,
            false,
            &options,
        )
        .unwrap_or_else(|e| panic!("encode_htj2k repetition {i} failed: {e}"));
        assert_eq!(
            result,
            baseline,
            "encode_htj2k repetition {i} produced different bytes \
                 (len baseline={}, len result={})",
            baseline.len(),
            result.len()
        );
    }

    println!(
        "encode_htj2k_is_byte_deterministic: {} bytes, {} repetitions all identical",
        baseline.len(),
        REPETITIONS
    );
}

/// Precondition gate: prove native `encode_htj2k` round-trips 2-component
/// 8-bit lossless images exactly with independent component channels.
#[cfg(feature = "std")]
#[test]
fn native_htj2k_roundtrips_two_component_lossless() {
    const WIDTH: u32 = 32;
    const HEIGHT: u32 = 24;
    const NUM_COMPONENTS: u8 = 2;
    const BIT_DEPTH: u8 = 8;

    // Deterministic per-pixel pattern: each sample is a function of its
    // flat index so the two planes carry different, non-trivial data.
    let pixel_count = WIDTH as usize * HEIGHT as usize * usize::from(NUM_COMPONENTS);
    let pixels: Vec<u8> = (0..pixel_count)
        .map(|i| {
            u8::try_from((i.wrapping_mul(251).wrapping_add(i / 7)) & 0xFF)
                .expect("masked test pixel fits u8")
        })
        .collect();

    let codestream = encode_htj2k(
        &pixels,
        WIDTH,
        HEIGHT,
        NUM_COMPONENTS.into(),
        BIT_DEPTH,
        false,
        &EncodeOptions::default(),
    )
    .expect("native 2-component HTJ2K encode failed");

    let image = Image::new(
        &codestream,
        &DecodeSettings {
            resolve_palette_indices: true,
            strict: true,
            target_resolution: None,
        },
    )
    .expect("native 2-component HTJ2K parse failed");
    let decoded = image
        .decode_native()
        .expect("native 2-component HTJ2K decode failed");

    assert_eq!(decoded.width, WIDTH, "width mismatch");
    assert_eq!(decoded.height, HEIGHT, "height mismatch");
    assert_eq!(decoded.bit_depth, BIT_DEPTH, "bit_depth mismatch");
    assert_eq!(
        decoded.num_components,
        u16::from(NUM_COMPONENTS),
        "component count mismatch"
    );
    assert_eq!(
        decoded.data, pixels,
        "2-component HTJ2K lossless round-trip mismatch"
    );

    println!(
        "native_htj2k_roundtrips_two_component_lossless: {} bytes codestream, {} pixel bytes",
        codestream.len(),
        pixels.len()
    );
}

/// Precondition gate: prove native `encode_htj2k` round-trips 4-component
/// (e.g. RGBA) 8-bit lossless images exactly.
/// Required before a CUDA parity oracle can be established for this component count.
#[cfg(feature = "std")]
#[test]
fn native_htj2k_roundtrips_four_component_lossless() {
    const WIDTH: u32 = 32;
    const HEIGHT: u32 = 24;
    const NUM_COMPONENTS: u8 = 4;
    const BIT_DEPTH: u8 = 8;

    // Deterministic per-sample pattern across all four planes.
    let pixel_count = WIDTH as usize * HEIGHT as usize * usize::from(NUM_COMPONENTS);
    let pixels: Vec<u8> = (0..pixel_count)
        .map(|i| {
            u8::try_from((i.wrapping_mul(197).wrapping_add(i / 13)) & 0xFF)
                .expect("masked test pixel fits u8")
        })
        .collect();

    let codestream = encode_htj2k(
        &pixels,
        WIDTH,
        HEIGHT,
        NUM_COMPONENTS.into(),
        BIT_DEPTH,
        false,
        &EncodeOptions::default(),
    )
    .expect("native 4-component HTJ2K encode failed");

    let image = Image::new(
        &codestream,
        &DecodeSettings {
            resolve_palette_indices: true,
            strict: true,
            target_resolution: None,
        },
    )
    .expect("native 4-component HTJ2K parse failed");
    let decoded = image
        .decode_native()
        .expect("native 4-component HTJ2K decode failed");

    assert_eq!(decoded.width, WIDTH, "width mismatch");
    assert_eq!(decoded.height, HEIGHT, "height mismatch");
    assert_eq!(decoded.bit_depth, BIT_DEPTH, "bit_depth mismatch");
    assert_eq!(
        decoded.num_components,
        u16::from(NUM_COMPONENTS),
        "component count mismatch"
    );
    assert_eq!(
        decoded.data, pixels,
        "4-component HTJ2K lossless round-trip mismatch"
    );

    println!(
        "native_htj2k_roundtrips_four_component_lossless: {} bytes codestream, {} pixel bytes",
        codestream.len(),
        pixels.len()
    );
}

#[test]
fn classic_pcrd_assigns_limited_budget_by_distortion_slope() {
    let candidates = vec![
        ClassicSegmentAssignmentCandidate {
            block_index: 0,
            segment_index: 0,
            rate: 500,
            distortion_delta: 500.0,
        },
        ClassicSegmentAssignmentCandidate {
            block_index: 1,
            segment_index: 0,
            rate: 700,
            distortion_delta: 7_000.0,
        },
        ClassicSegmentAssignmentCandidate {
            block_index: 2,
            segment_index: 0,
            rate: 600,
            distortion_delta: 3_000.0,
        },
    ];

    let assignments = assign_classic_segment_layers_by_slope(&candidates, 2, &[256, 3_000])
        .expect("PCRD assignment");

    assert_eq!(
        assignments,
        vec![1, 0, 1],
        "the highest slope contribution should consume the constrained first-layer budget"
    );
}

#[test]
fn classic_pcrd_allows_byte_target_tolerance_for_first_legal_truncation() {
    let candidates = vec![ClassicSegmentAssignmentCandidate {
        block_index: 0,
        segment_index: 0,
        rate: 300,
        distortion_delta: 1_000.0,
    }];

    let assignments = assign_classic_segment_layers_by_slope(&candidates, 2, &[256, 1_000])
        .expect("PCRD assignment");

    assert_eq!(assignments, vec![0]);
}

#[test]
fn classic_pcrd_does_not_spend_budget_on_non_prefix_segments() {
    let candidates = vec![
        ClassicSegmentAssignmentCandidate {
            block_index: 0,
            segment_index: 0,
            rate: 1_000,
            distortion_delta: 1_000.0,
        },
        ClassicSegmentAssignmentCandidate {
            block_index: 0,
            segment_index: 1,
            rate: 500,
            distortion_delta: 10_000.0,
        },
        ClassicSegmentAssignmentCandidate {
            block_index: 1,
            segment_index: 0,
            rate: 300,
            distortion_delta: 600.0,
        },
    ];

    let assignments = assign_classic_segment_layers_by_slope(&candidates, 2, &[256, 2_000])
        .expect("PCRD assignment");

    assert_eq!(
        assignments,
        vec![1, 1, 0],
        "first-layer budget must go to the best legal prefix contribution"
    );
}

#[test]
fn ht_layer_assignment_uses_segment_budget_before_block_index() {
    let candidates = vec![
        HtSegmentAssignmentCandidate {
            block_index: 0,
            segment_index: 0,
            rate: 900,
        },
        HtSegmentAssignmentCandidate {
            block_index: 1,
            segment_index: 0,
            rate: 200,
        },
        HtSegmentAssignmentCandidate {
            block_index: 2,
            segment_index: 0,
            rate: 200,
        },
    ];

    let assignments = assign_ht_segment_layers_by_budget(&candidates, 2, &[256, 2_000])
        .expect("HTJ2K segment assignment");

    assert_eq!(
        assignments,
        vec![1, 0, 0],
        "HTJ2K early layers should be filled by segment byte budget, not block index"
    );
}

#[test]
fn ht_layer_assignment_keeps_refinement_after_cleanup() {
    let candidates = vec![
        HtSegmentAssignmentCandidate {
            block_index: 0,
            segment_index: 0,
            rate: 200,
        },
        HtSegmentAssignmentCandidate {
            block_index: 0,
            segment_index: 1,
            rate: 50,
        },
    ];

    let assignments = assign_ht_segment_layers_by_budget(&candidates, 2, &[256, 2_000])
        .expect("HTJ2K segment assignment");

    assert_eq!(
        assignments,
        vec![0, 0],
        "a refinement segment may share the cleanup layer but must not precede it"
    );
}

#[test]
fn ht_layer_contributions_split_cleanup_and_refinement_across_layers() {
    let encoded = bitplane_encode::EncodedCodeBlock {
        data: vec![0x11, 0x22, 0x33, 0x44, 0x55],
        num_coding_passes: 3,
        num_zero_bitplanes: 2,
        ht_cleanup_length: 3,
        ht_refinement_length: 2,
    };

    let contributions = ht_layer_contributions(&encoded, 2, &[0, 1]).expect("split HT layers");

    assert_eq!(contributions.len(), 2);
    assert_eq!(contributions[0].data, vec![0x11, 0x22, 0x33]);
    assert_eq!(contributions[0].ht_cleanup_length, 3);
    assert_eq!(contributions[0].ht_refinement_length, 0);
    assert_eq!(contributions[0].num_coding_passes, 1);
    assert_eq!(contributions[1].data, vec![0x44, 0x55]);
    assert_eq!(contributions[1].ht_cleanup_length, 0);
    assert_eq!(contributions[1].ht_refinement_length, 2);
    assert_eq!(contributions[1].num_coding_passes, 2);
}

#[test]
fn htj2k_lossy_quality_layers_decode_split_refinement_layer() {
    let width = 32;
    let height = 32;
    let pixels = gradient_u8(width, height);
    let codestream = encode_htj2k(
        &pixels,
        width,
        height,
        1,
        8,
        false,
        &EncodeOptions {
            num_decomposition_levels: 0,
            reversible: false,
            guard_bits: 2,
            num_layers: 2,
            ..Default::default()
        },
    )
    .expect("HTJ2K layered encode");

    let image = Image::new(
        &codestream,
        &DecodeSettings {
            resolve_palette_indices: true,
            strict: true,
            target_resolution: None,
        },
    )
    .expect("parse layered HT codestream");
    let decoded = image.decode_native().expect("decode layered HT codestream");

    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);
    assert_eq!(decoded.bit_depth, 8);
    assert_not_flat_128(&decoded.data);
    assert!(
        psnr_db(&pixels, &decoded.data) >= 30.0,
        "psnr={} max_abs={}",
        psnr_db(&pixels, &decoded.data),
        max_abs_error(&pixels, &decoded.data)
    );
}
