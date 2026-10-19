// SPDX-License-Identifier: MIT OR Apache-2.0

//! Prepared-decoder resident batch eligibility inspection.

use super::plan::rgb8_metal_output_dimensions_for_op;
use crate::{Codec, Decoder, JpegMetalResidentBatchReport};
use j2k_core::PixelFormat;

impl Codec {
    /// Inspect a cached RGB8 decoder batch for reusable Metal resident output.
    ///
    /// The report exposes whether the batch is resident-output eligible and,
    /// when eligible, the exact output dimensions and tile capacity callers
    /// should allocate before dispatch.
    #[doc(hidden)]
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
