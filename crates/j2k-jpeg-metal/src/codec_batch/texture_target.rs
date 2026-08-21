// SPDX-License-Identifier: MIT OR Apache-2.0

//! Caller-owned Metal texture batch validation, resizing, and submission.

use super::request::{
    MetalTextureBatchTarget, Rgb8MetalBatchOp, Rgb8MetalBatchRequest, Rgb8MetalBatchSource,
};
use crate::{
    compute, Codec, Decoder, Error, MetalBackendSession, MetalBatchTextureOutput, MetalTextureTile,
};
use j2k_core::{Downscale, Rect};

const fn unsupported_reason(op: Rgb8MetalBatchOp) -> &'static str {
    match op {
        Rgb8MetalBatchOp::Full => {
            "JPEG Metal texture batch output currently supports batchable full-tile RGB8 fast 4:2:0, 4:2:2, or 4:4:4 inputs"
        }
        Rgb8MetalBatchOp::Scaled(_) => {
            "JPEG Metal texture scaled batch output currently supports batchable RGB8 fast 4:2:0, 4:2:2, or 4:4:4 inputs with half, quarter, or eighth scaling"
        }
        Rgb8MetalBatchOp::RegionScaled { .. } => {
            "JPEG Metal texture region-scaled batch output currently supports batchable RGB8 fast 4:2:0, 4:2:2, or 4:4:4 inputs with matching output shapes"
        }
    }
}

impl Codec {
    /// Decode a batched RGB8 JPEG request into caller-owned Metal RGBA8 textures.
    pub fn decode_rgb8_batch_into_textures_with_session(
        request: Rgb8MetalBatchRequest<'_, '_>,
        target: MetalTextureBatchTarget<'_>,
        session: &MetalBackendSession,
    ) -> Result<Vec<Result<MetalTextureTile, Error>>, Error> {
        if request.source.is_empty() {
            return Ok(Vec::new());
        }

        let resizable = matches!(target, MetalTextureBatchTarget::Resizable(_));
        let (plan, tile_count) =
            Self::plan_rgb8_metal_batch(request.source, request.op, resizable)?;
        let output: &MetalBatchTextureOutput = match target {
            MetalTextureBatchTarget::Reusable(output) => output,
            MetalTextureBatchTarget::Resizable(output) => {
                if let Rgb8MetalBatchSource::Decoders(decoders) = request.source {
                    let report = Self::inspect_rgb8_decoder_batch_metal_output(
                        decoders,
                        Self::rgb8_batch_jpeg_decode_op(request.op),
                    );
                    output.ensure_rgba8_batch_report(session, &report)?;
                }
                let Some(output_dimensions) = plan.output_dimensions else {
                    return Ok(Vec::new());
                };
                output.ensure_rgba8_tiles(session, output_dimensions, tile_count)?;
                output
            }
        };

        let results = match request.op {
            Rgb8MetalBatchOp::Full => {
                compute::batch_entry::decode_full_rgb8_batch_into_textures_with_session(
                    &plan.requests,
                    output,
                    session,
                )?
            }
            Rgb8MetalBatchOp::Scaled(_) | Rgb8MetalBatchOp::RegionScaled { .. } => {
                compute::batch_entry::decode_region_scaled_rgb8_batch_into_textures_with_session(
                    &plan.requests,
                    output,
                    session,
                )?
            }
        };
        results.ok_or(Error::capability_rejected(
            j2k_core::CapabilityRejection::unsupported_operation(unsupported_reason(request.op)),
        ))
    }

    /// Decode a full-tile decoder batch into resizable Metal RGBA8 textures.
    pub fn decode_rgb8_decoder_batch_into_resizable_metal_textures_with_session(
        decoders: &[&Decoder<'_>],
        output: &mut MetalBatchTextureOutput,
        session: &MetalBackendSession,
    ) -> Result<Vec<Result<MetalTextureTile, Error>>, Error> {
        Self::decode_rgb8_batch_into_textures_with_session(
            Rgb8MetalBatchRequest {
                source: Rgb8MetalBatchSource::Decoders(decoders),
                op: Rgb8MetalBatchOp::Full,
            },
            MetalTextureBatchTarget::Resizable(output),
            session,
        )
    }

    /// Decode a region-scaled RGB8 JPEG batch into resizable Metal RGBA8 textures.
    pub fn decode_rgb8_region_scaled_batch_into_resizable_metal_textures_with_session(
        inputs: &[&[u8]],
        roi: Rect,
        scale: Downscale,
        output: &mut MetalBatchTextureOutput,
        session: &MetalBackendSession,
    ) -> Result<Vec<Result<MetalTextureTile, Error>>, Error> {
        Self::decode_rgb8_batch_into_textures_with_session(
            Rgb8MetalBatchRequest {
                source: Rgb8MetalBatchSource::Bytes(inputs),
                op: Rgb8MetalBatchOp::RegionScaled { roi, scale },
            },
            MetalTextureBatchTarget::Resizable(output),
            session,
        )
    }
}
