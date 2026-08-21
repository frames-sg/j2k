// SPDX-License-Identifier: MIT OR Apache-2.0

//! Prepared-decoder resident batch eligibility inspection.

use super::plan::rgb8_metal_output_dimensions_for_op;
use super::source::{decoder_resident_restart_interval_mcus, decoder_resident_sampling_family};
use crate::{batch, Codec, Decoder, JpegMetalResidentBatchReport};
use j2k_core::PixelFormat;

impl Codec {
    /// Inspect a cached RGB8 decoder batch for reusable Metal resident output.
    ///
    /// The report exposes whether the batch is resident-output eligible and,
    /// when eligible, the exact output dimensions and tile capacity callers
    /// should allocate before dispatch.
    #[doc(hidden)]
    #[expect(
        clippy::too_many_lines,
        reason = "the ordered fail-closed eligibility checks keep the first unsupported batch reason deterministic"
    )]
    pub fn inspect_rgb8_decoder_batch_metal_output(
        decoders: &[&Decoder<'_>],
        op: j2k_jpeg::JpegDecodeOp,
    ) -> JpegMetalResidentBatchReport {
        if decoders.is_empty() {
            return JpegMetalResidentBatchReport {
                op,
                tile_count: 0,
                output_dimensions: None,
                eligibility: j2k_jpeg::JpegBackendEligibility {
                    eligible: true,
                    reason: None,
                },
            };
        }

        let mut output_dimensions = None;
        let mut sampling_family = None;
        for decoder in decoders {
            let request = j2k_jpeg::JpegCapabilityRequest {
                op,
                fmt: PixelFormat::Rgb8,
            };
            let report = j2k_jpeg::JpegCapabilityReport::for_decoder(decoder.inner(), request);
            let eligibility = report.metal_resident_rgb8_batch_output();
            if !eligibility.eligible {
                return JpegMetalResidentBatchReport {
                    op,
                    tile_count: decoders.len(),
                    output_dimensions: None,
                    eligibility,
                };
            }

            if decoder.fast444_packet().is_none()
                && decoder.fast422_packet().is_none()
                && decoder.fast420_packet().is_none()
            {
                return JpegMetalResidentBatchReport {
                    op,
                    tile_count: decoders.len(),
                    output_dimensions: None,
                    eligibility: j2k_jpeg::JpegBackendEligibility {
                        eligible: false,
                        reason: Some(
                            "JPEG Metal reusable resident batch output requires cached fast-packet state",
                        ),
                    },
                };
            }

            let Some(dimensions) =
                rgb8_metal_output_dimensions_for_op(decoder.inner().info().dimensions, op)
            else {
                return JpegMetalResidentBatchReport {
                    op,
                    tile_count: decoders.len(),
                    output_dimensions: None,
                    eligibility,
                };
            };
            if let Some(first) = output_dimensions {
                if first != dimensions {
                    return JpegMetalResidentBatchReport {
                        op,
                        tile_count: decoders.len(),
                        output_dimensions: None,
                        eligibility: j2k_jpeg::JpegBackendEligibility {
                            eligible: false,
                            reason: Some(
                                "JPEG Metal reusable RGB8 batch output requires matching output dimensions",
                            ),
                        },
                    };
                }
            } else {
                output_dimensions = Some(dimensions);
            }

            let decoder_sampling_family = decoder_resident_sampling_family(decoder);
            if let Some(first) = sampling_family {
                if first != decoder_sampling_family {
                    return JpegMetalResidentBatchReport {
                        op,
                        tile_count: decoders.len(),
                        output_dimensions: None,
                        eligibility: j2k_jpeg::JpegBackendEligibility {
                            eligible: false,
                            reason: Some(
                                "JPEG Metal reusable resident batch output requires one batch to use the same fast-packet sampling family",
                            ),
                        },
                    };
                }
            } else {
                sampling_family = Some(decoder_sampling_family);
            }

            if op == j2k_jpeg::JpegDecodeOp::Full
                && matches!(
                    decoder_sampling_family,
                    batch::SamplingFamily::Fast422 | batch::SamplingFamily::Fast444
                )
                && decoder_resident_restart_interval_mcus(decoder) != 0
            {
                return JpegMetalResidentBatchReport {
                    op,
                    tile_count: decoders.len(),
                    output_dimensions: None,
                    eligibility: j2k_jpeg::JpegBackendEligibility {
                        eligible: false,
                        reason: Some(
                            "JPEG Metal reusable resident batch output does not support restart-coded full-tile 4:2:2 or 4:4:4 batches",
                        ),
                    },
                };
            }
        }

        JpegMetalResidentBatchReport {
            op,
            tile_count: decoders.len(),
            output_dimensions,
            eligibility: j2k_jpeg::JpegBackendEligibility {
                eligible: true,
                reason: None,
            },
        }
    }
}
