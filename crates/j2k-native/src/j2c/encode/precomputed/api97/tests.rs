// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::j2c::{ht_block_encode, quantize};
use crate::{
    EncodeError, EncodedHtJ2kCodeBlock, J2kForwardDwt97Output, J2kQuantizeSubbandJob,
    J2kSubBandType, PrecomputedHtj2k97Component, PreencodedHtj2k97CodeBlock,
    PreencodedHtj2k97Component, PreencodedHtj2k97Resolution, PreencodedHtj2k97Subband,
    PrequantizedHtj2k97CodeBlock, PrequantizedHtj2k97Component, PrequantizedHtj2k97Resolution,
    PrequantizedHtj2k97Subband,
};
use alloc::vec;

fn options() -> EncodeOptions {
    EncodeOptions {
        num_decomposition_levels: 0,
        reversible: false,
        guard_bits: 2,
        use_ht_block_coding: true,
        code_block_width_exp: 2,
        code_block_height_exp: 2,
        ..EncodeOptions::default()
    }
}

fn precomputed_image() -> PrecomputedHtj2k97Image {
    PrecomputedHtj2k97Image {
        width: 1,
        height: 1,
        bit_depth: 8,
        signed: false,
        components: vec![PrecomputedHtj2k97Component {
            x_rsiz: 1,
            y_rsiz: 1,
            dwt: J2kForwardDwt97Output {
                ll: vec![1.0],
                ll_width: 1,
                ll_height: 1,
                levels: Vec::new(),
            },
        }],
    }
}

fn total_bitplanes(options: &EncodeOptions) -> u8 {
    let guard_bits = options.guard_bits.max(2);
    let steps = quantize::compute_step_sizes_with_irreversible_profile(
        8,
        0,
        false,
        guard_bits,
        options.irreversible_quantization_scale,
        options.irreversible_quantization_subband_scales,
    );
    guard_bits
        .saturating_add(u8::try_from(steps[0].exponent).expect("test exponent fits u8"))
        .saturating_sub(1)
}

fn prequantized_image(options: &EncodeOptions) -> PrequantizedHtj2k97Image {
    PrequantizedHtj2k97Image {
        width: 1,
        height: 1,
        bit_depth: 8,
        signed: false,
        components: vec![PrequantizedHtj2k97Component {
            x_rsiz: 1,
            y_rsiz: 1,
            resolutions: vec![PrequantizedHtj2k97Resolution {
                subbands: vec![PrequantizedHtj2k97Subband {
                    sub_band_type: J2kSubBandType::LowLow,
                    num_cbs_x: 1,
                    num_cbs_y: 1,
                    total_bitplanes: total_bitplanes(options),
                    code_blocks: vec![PrequantizedHtj2k97CodeBlock {
                        coefficients: vec![1],
                        width: 1,
                        height: 1,
                    }],
                }],
            }],
        }],
    }
}

fn preencoded_image(options: &EncodeOptions) -> PreencodedHtj2k97Image {
    let total_bitplanes = total_bitplanes(options);
    let block = ht_block_encode::encode_code_block(&[1], 1, 1, total_bitplanes)
        .expect("test HT code-block encode");
    PreencodedHtj2k97Image {
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
                            data: block.data,
                            cleanup_length: block.ht_cleanup_length,
                            refinement_length: block.ht_refinement_length,
                            num_coding_passes: block.num_coding_passes,
                            num_zero_bitplanes: block.num_zero_bitplanes,
                        },
                    }],
                }],
            }],
        }],
    }
}

fn compact_preencoded_image(options: &EncodeOptions) -> PreencodedHtj2k97CompactImage {
    let total_bitplanes = total_bitplanes(options);
    let block = ht_block_encode::encode_code_block(&[1], 1, 1, total_bitplanes)
        .expect("test compact HT code-block encode");
    let cleanup_length = block.ht_cleanup_length;
    let refinement_length = block.ht_refinement_length;
    let num_coding_passes = block.num_coding_passes;
    let num_zero_bitplanes = block.num_zero_bitplanes;
    let payload = block.data;
    let payload_len = payload.len();

    PreencodedHtj2k97CompactImage {
        width: 1,
        height: 1,
        bit_depth: 8,
        signed: false,
        payload,
        components: vec![crate::PreencodedHtj2k97CompactComponent {
            x_rsiz: 1,
            y_rsiz: 1,
            resolutions: vec![crate::PreencodedHtj2k97CompactResolution {
                subbands: vec![crate::PreencodedHtj2k97CompactSubband {
                    sub_band_type: J2kSubBandType::LowLow,
                    num_cbs_x: 1,
                    num_cbs_y: 1,
                    total_bitplanes,
                    code_blocks: vec![crate::PreencodedHtj2k97CompactCodeBlock {
                        width: 1,
                        height: 1,
                        payload_range: 0..payload_len,
                        cleanup_length,
                        refinement_length,
                        num_coding_passes,
                        num_zero_bitplanes,
                    }],
                }],
            }],
        }],
    }
}

#[derive(Default)]
struct CoefficientPointerAccelerator {
    quantized_inputs: Vec<usize>,
    forward_dwt97_calls: usize,
}

struct FailingQuantizationAccelerator;

impl J2kEncodeStageAccelerator for FailingQuantizationAccelerator {
    fn encode_quantize_subband(
        &mut self,
        _job: J2kQuantizeSubbandJob<'_>,
    ) -> crate::J2kEncodeStageResult<Option<Vec<i32>>> {
        Err(crate::J2kEncodeStageError::internal_invariant(
            "injected precomputed 9/7 quantization failure",
        ))
    }
}

impl J2kEncodeStageAccelerator for CoefficientPointerAccelerator {
    fn encode_forward_dwt97(
        &mut self,
        _job: crate::J2kForwardDwt97Job<'_>,
    ) -> crate::J2kEncodeStageResult<Option<J2kForwardDwt97Output>> {
        self.forward_dwt97_calls += 1;
        Ok(None)
    }

    fn encode_quantize_subband(
        &mut self,
        job: J2kQuantizeSubbandJob<'_>,
    ) -> crate::J2kEncodeStageResult<Option<Vec<i32>>> {
        self.quantized_inputs
            .push(job.coefficients.as_ptr() as usize);
        Ok(None)
    }
}

#[test]
fn precomputed_97_quantization_borrows_the_source_dwt_allocation() {
    let image = precomputed_image();
    let source_ptr = image.components[0].dwt.ll.as_ptr() as usize;
    let mut accelerator = CoefficientPointerAccelerator::default();

    let codestream =
        encode_precomputed_htj2k_97_with_accelerator(&image, &options(), &mut accelerator)
            .expect("borrowed precomputed 9/7 encode");

    assert!(codestream.starts_with(&[0xff, 0x4f]));
    assert_eq!(accelerator.forward_dwt97_calls, 0);
    assert!(accelerator.quantized_inputs.contains(&source_ptr));
}

#[test]
fn public_precomputed_97_keeps_accelerator_error_category() {
    let error = encode_precomputed_htj2k_97_with_accelerator(
        &precomputed_image(),
        &options(),
        &mut FailingQuantizationAccelerator,
    )
    .expect_err("accelerator failure must remain typed");

    assert_eq!(
        error,
        EncodeError::Accelerator {
            operation: "subband quantization",
            source: crate::J2kEncodeStageError::internal_invariant(
                "injected precomputed 9/7 quantization failure",
            ),
        }
    );
}

#[test]
fn public_precomputed_97_keeps_shallow_and_deep_input_errors_typed() {
    let mut invalid_dimensions = precomputed_image();
    invalid_dimensions.width = 0;
    assert_eq!(
        encode_precomputed_htj2k_97(&invalid_dimensions, &options()),
        Err(EncodeError::InvalidInput {
            what: "invalid dimensions",
        })
    );

    let mut invalid_dwt = precomputed_image();
    invalid_dwt.components[0].dwt.ll.clear();
    assert_eq!(
        encode_precomputed_htj2k_97(&invalid_dwt, &options()),
        Err(EncodeError::InvalidInput {
            what: "accelerated DWT output length mismatch",
        })
    );

    let invalid_options = EncodeOptions {
        code_block_width_exp: u8::MAX,
        ..options()
    };
    assert_eq!(
        encode_precomputed_htj2k_97(&precomputed_image(), &invalid_options),
        Err(EncodeError::InvalidInput {
            what: "code-block width exponent exceeds supported range",
        })
    );
}

#[test]
fn public_prequantized_and_preencoded_layer_errors_keep_categories() {
    let zero_layers = EncodeOptions {
        num_layers: 0,
        ..options()
    };
    assert_eq!(
        encode_prequantized_htj2k_97(&prequantized_image(&options()), &zero_layers),
        Err(EncodeError::InvalidInput {
            what: "quality layer count must be non-zero",
        })
    );
    assert_eq!(
        encode_preencoded_htj2k_97(&preencoded_image(&options()), &zero_layers),
        Err(EncodeError::InvalidInput {
            what: "quality layer count must be non-zero",
        })
    );

    let mismatched_targets = EncodeOptions {
        quality_layer_byte_targets: vec![1, 2],
        ..options()
    };
    assert_eq!(
        encode_prequantized_htj2k_97(&prequantized_image(&options()), &mismatched_targets),
        Err(EncodeError::InvalidInput {
            what: "quality layer byte target count must match quality layer count",
        })
    );
    assert_eq!(
        encode_preencoded_htj2k_97(&preencoded_image(&options()), &mismatched_targets),
        Err(EncodeError::InvalidInput {
            what: "quality layer byte target count must match quality layer count",
        })
    );

    let multiple_layers = EncodeOptions {
        num_layers: 2,
        ..options()
    };
    assert_eq!(
        encode_preencoded_htj2k_97(&preencoded_image(&options()), &multiple_layers),
        Err(EncodeError::Unsupported {
            what: "precomputed 9/7 packet input supports one quality layer",
        })
    );
}

#[test]
fn public_compact_option_errors_keep_invalid_input_precedence() {
    let zero_layers = EncodeOptions {
        num_layers: 0,
        ..options()
    };
    assert_eq!(
        encode_preencoded_htj2k_97_compact_owned_with_accelerator(
            compact_preencoded_image(&options()),
            &zero_layers,
            &mut CpuOnlyJ2kEncodeStageAccelerator,
        ),
        Err(EncodeError::InvalidInput {
            what: "quality layer count must be non-zero",
        })
    );

    let mismatched_targets = EncodeOptions {
        quality_layer_byte_targets: vec![1, 2],
        ..options()
    };
    assert_eq!(
        encode_preencoded_htj2k_97_compact_owned_with_accelerator(
            compact_preencoded_image(&options()),
            &mismatched_targets,
            &mut CpuOnlyJ2kEncodeStageAccelerator,
        ),
        Err(EncodeError::InvalidInput {
            what: "quality layer byte target count must match quality layer count",
        })
    );

    let multiple_layers = EncodeOptions {
        num_layers: 2,
        ..options()
    };
    assert_eq!(
        encode_preencoded_htj2k_97_compact_owned_with_accelerator(
            compact_preencoded_image(&options()),
            &multiple_layers,
            &mut CpuOnlyJ2kEncodeStageAccelerator,
        ),
        Err(EncodeError::Unsupported {
            what: "compact preencoded HTJ2K encode supports one quality layer",
        })
    );

    let mutually_exclusive_headers = EncodeOptions {
        write_ppm: true,
        write_ppt: true,
        ..options()
    };
    assert_eq!(
        encode_preencoded_htj2k_97_compact_owned_with_accelerator(
            compact_preencoded_image(&options()),
            &mutually_exclusive_headers,
            &mut CpuOnlyJ2kEncodeStageAccelerator,
        ),
        Err(EncodeError::InvalidInput {
            what: "PPM and PPT packet header markers are mutually exclusive",
        })
    );

    let zero_tile_part_limit = EncodeOptions {
        tile_part_packet_limit: Some(0),
        ..options()
    };
    assert_eq!(
        encode_preencoded_htj2k_97_compact_owned_with_accelerator(
            compact_preencoded_image(&options()),
            &zero_tile_part_limit,
            &mut CpuOnlyJ2kEncodeStageAccelerator,
        ),
        Err(EncodeError::InvalidInput {
            what: "tile-part packet limit must be non-zero",
        })
    );
}

fn encode_precomputed_at_cap(
    image: &PrecomputedHtj2k97Image,
    options: &EncodeOptions,
    cap: usize,
) -> NativeEncodePipelineResult<Vec<u8>> {
    let retained_bytes = precomputed_97_image_retained_bytes(image)?;
    let session = NativeEncodeSession::try_with_cap(
        NativeEncodeRetainedInput::from_owner_bytes(image, retained_bytes),
        cap,
    )?;
    let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    encode_precomputed_for_session(image, options, &session, &mut accelerator)
}

#[test]
fn borrowed_precomputed_97_accepts_its_measured_exact_peak_and_rejects_one_byte_less() {
    let image = precomputed_image();
    let options = options();
    let retained = precomputed_97_image_retained_bytes(&image).expect("retained input bytes");
    let mut low = retained;
    let mut high = retained + 1_048_576;
    assert!(encode_precomputed_at_cap(&image, &options, high).is_ok());
    while low < high {
        let midpoint = low + (high - low) / 2;
        if encode_precomputed_at_cap(&image, &options, midpoint).is_ok() {
            high = midpoint;
        } else {
            low = midpoint + 1;
        }
    }
    let exact = encode_precomputed_at_cap(&image, &options, low)
        .expect("measured exact precomputed 9/7 cap");
    let expected = encode_precomputed_htj2k_97(&image, &options).expect("public encode");
    assert_eq!(exact, expected);
    assert!(matches!(
        encode_precomputed_at_cap(&image, &options, low - 1),
        Err(NativeEncodePipelineError::Typed(
            EncodeError::AllocationTooLarge { .. }
        ))
    ));
}

#[derive(Default)]
struct PayloadPointerAccelerator {
    observed_payloads: Vec<usize>,
}

impl J2kEncodeStageAccelerator for PayloadPointerAccelerator {
    fn encode_packetization(
        &mut self,
        job: crate::J2kPacketizationEncodeJob<'_>,
    ) -> crate::J2kEncodeStageResult<Option<Vec<u8>>> {
        self.observed_payloads.extend(
            job.resolutions
                .iter()
                .flat_map(|resolution| &resolution.subbands)
                .flat_map(|subband| &subband.code_blocks)
                .filter(|block| !block.data.is_empty())
                .map(|block| block.data.as_ptr() as usize),
        );
        Ok(None)
    }
}

#[test]
fn owned_preencoded_97_moves_payloads_through_packetization_without_copying() {
    let image = preencoded_image(&options());
    let source_ptr = image.components[0].resolutions[0].subbands[0].code_blocks[0]
        .encoded
        .data
        .as_ptr() as usize;
    let mut accelerator = PayloadPointerAccelerator::default();

    let codestream =
        encode_preencoded_htj2k_97_owned_with_accelerator(image, &options(), &mut accelerator)
            .expect("owned preencoded 9/7 encode");

    assert!(codestream.starts_with(&[0xff, 0x4f]));
    assert!(accelerator.observed_payloads.contains(&source_ptr));
}

#[test]
fn borrowed_prequantized_and_owned_preencoded_keep_byte_parity() {
    let options = options();
    let prequantized = prequantized_image(&options);
    let preencoded = preencoded_image(&options);

    let quantized =
        encode_prequantized_htj2k_97(&prequantized, &options).expect("prequantized encode");
    let preencoded = encode_preencoded_htj2k_97_owned_with_accelerator(
        preencoded,
        &options,
        &mut CpuOnlyJ2kEncodeStageAccelerator,
    )
    .expect("owned preencoded encode");

    assert_eq!(preencoded, quantized);
}

fn encode_prequantized_at_cap(
    image: &PrequantizedHtj2k97Image,
    options: &EncodeOptions,
    cap: usize,
) -> NativeEncodePipelineResult<Vec<u8>> {
    let retained = prequantized_97_image_retained_bytes(image)?;
    let session = NativeEncodeSession::try_with_cap(
        NativeEncodeRetainedInput::from_owner_bytes(image, retained),
        cap,
    )?;
    let plan = prepare_prequantized_plan(image, options, &session)?;
    orchestrator::encode_plan(plan, &session, &mut CpuOnlyJ2kEncodeStageAccelerator)
}

fn encode_borrowed_preencoded_at_cap(
    image: &PreencodedHtj2k97Image,
    options: &EncodeOptions,
    cap: usize,
) -> NativeEncodePipelineResult<Vec<u8>> {
    let retained = preencoded_97_image_retained_bytes(image)?;
    let session = NativeEncodeSession::try_with_cap(
        NativeEncodeRetainedInput::from_owner_bytes(image, retained),
        cap,
    )?;
    let plan = prepare_borrowed_preencoded_plan(image, options, &session)?;
    orchestrator::encode_plan(plan, &session, &mut CpuOnlyJ2kEncodeStageAccelerator)
}

fn measured_minimum_cap(mut succeeds: impl FnMut(usize) -> bool, low: usize) -> usize {
    let mut low = low;
    let mut high = low + 1_048_576;
    assert!(succeeds(high), "test upper cap must be sufficient");
    while low < high {
        let midpoint = low + (high - low) / 2;
        if succeeds(midpoint) {
            high = midpoint;
        } else {
            low = midpoint + 1;
        }
    }
    low
}

#[test]
fn copied_packet_inputs_enforce_exact_aggregate_caps() {
    let options = options();
    let prequantized = prequantized_image(&options);
    let prequantized_retained =
        prequantized_97_image_retained_bytes(&prequantized).expect("prequantized retained bytes");
    let prequantized_cap = measured_minimum_cap(
        |cap| encode_prequantized_at_cap(&prequantized, &options, cap).is_ok(),
        prequantized_retained,
    );
    assert!(encode_prequantized_at_cap(&prequantized, &options, prequantized_cap).is_ok());
    assert!(matches!(
        encode_prequantized_at_cap(&prequantized, &options, prequantized_cap - 1),
        Err(NativeEncodePipelineError::Typed(
            EncodeError::AllocationTooLarge { .. }
        ))
    ));

    let preencoded = preencoded_image(&options);
    let preencoded_retained =
        preencoded_97_image_retained_bytes(&preencoded).expect("preencoded retained bytes");
    let preencoded_cap = measured_minimum_cap(
        |cap| encode_borrowed_preencoded_at_cap(&preencoded, &options, cap).is_ok(),
        preencoded_retained,
    );
    assert!(encode_borrowed_preencoded_at_cap(&preencoded, &options, preencoded_cap).is_ok());
    assert!(matches!(
        encode_borrowed_preencoded_at_cap(&preencoded, &options, preencoded_cap - 1),
        Err(NativeEncodePipelineError::Typed(
            EncodeError::AllocationTooLarge { .. }
        ))
    ));
}

#[test]
fn public_compact_preencoded_97_keeps_invalid_input_category() {
    let image = PreencodedHtj2k97CompactImage {
        width: 0,
        height: 1,
        bit_depth: 8,
        signed: false,
        payload: Vec::new(),
        components: Vec::new(),
    };

    assert_eq!(
        encode_preencoded_htj2k_97_compact_owned_with_accelerator(
            image,
            &options(),
            &mut CpuOnlyJ2kEncodeStageAccelerator,
        ),
        Err(EncodeError::InvalidInput {
            what: "invalid dimensions",
        })
    );
}
