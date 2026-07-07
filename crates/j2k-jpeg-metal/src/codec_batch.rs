// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{BackendRequest, Downscale, PixelFormat, Rect, TileBatchDecodeSubmit};
use j2k_jpeg::{DecoderContext as CpuDecoderContext, ScratchPool as CpuScratchPool};

#[cfg(target_os = "macos")]
use j2k_jpeg::Decoder as CpuDecoder;

use crate::{batch, session, Codec, Error, MetalDecodeRequest, MetalSession};

#[cfg(target_os = "macos")]
use crate::{
    compute, scaled_dims, Decoder, JpegMetalResidentBatchReport, MetalBackendSession,
    MetalBatchOutputBuffer, MetalBatchTextureOutput, MetalTextureTile, Surface,
};

/// Inputs for a batched RGB8 Metal decode: raw JPEG bytes or pre-parsed
/// decoders that carry cached Metal fast-packet state.
#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub enum Rgb8MetalBatchSource<'a, 'b> {
    /// Raw JPEG byte streams, parsed per call.
    Bytes(&'a [&'a [u8]]),
    /// Already parsed `Decoder` wrappers; reuses their cached Metal
    /// fast-packet state when building the resident batch request.
    Decoders(&'a [&'a Decoder<'b>]),
}

#[cfg(target_os = "macos")]
impl Rgb8MetalBatchSource<'_, '_> {
    fn is_empty(&self) -> bool {
        match self {
            Rgb8MetalBatchSource::Bytes(inputs) => inputs.is_empty(),
            Rgb8MetalBatchSource::Decoders(decoders) => decoders.is_empty(),
        }
    }
}

/// Geometry op applied to every tile of a batched RGB8 Metal decode.
#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rgb8MetalBatchOp {
    /// Full-tile decode at native dimensions.
    Full,
    /// Whole-tile downscale (half, quarter, or eighth).
    Scaled(Downscale),
    /// Scaled decode of one region, shared by every tile in the batch.
    RegionScaled {
        /// Region of interest to decode from every source tile.
        roi: Rect,
        /// Downscale factor applied to the selected region.
        scale: Downscale,
    },
}

/// A batched RGB8 Metal decode request: what to decode and how.
#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub struct Rgb8MetalBatchRequest<'a, 'b> {
    /// Source JPEG bytes or prepared decoders for the batch.
    pub source: Rgb8MetalBatchSource<'a, 'b>,
    /// Geometry operation applied to each source tile.
    pub op: Rgb8MetalBatchOp,
}

/// Caller-owned Metal buffer target for a batched RGB8 decode.
#[cfg(target_os = "macos")]
pub enum MetalBufferBatchTarget<'a> {
    /// Reuse the buffer as-is; its shape must already fit the batch.
    Reusable(&'a MetalBatchOutputBuffer),
    /// Grow the buffer to fit the batch before decoding.
    Resizable(&'a mut MetalBatchOutputBuffer),
}

/// Caller-owned Metal RGBA8 texture target for a batched RGB8 decode.
#[cfg(target_os = "macos")]
pub enum MetalTextureBatchTarget<'a> {
    /// Reuse the texture set as-is; its shape must already fit the batch.
    Reusable(&'a MetalBatchTextureOutput),
    /// Grow the texture set to fit the batch before decoding.
    Resizable(&'a mut MetalBatchTextureOutput),
}

#[cfg(target_os = "macos")]
struct Rgb8MetalBatchPlan {
    requests: Vec<batch::QueuedRequest>,
    output_dimensions: Option<(u32, u32)>,
}

#[cfg(target_os = "macos")]
fn rgb8_metal_output_dimensions_for_op(
    full_dimensions: (u32, u32),
    op: j2k_jpeg::JpegDecodeOp,
) -> Option<(u32, u32)> {
    match op {
        j2k_jpeg::JpegDecodeOp::Full => Some(full_dimensions),
        j2k_jpeg::JpegDecodeOp::Scaled(scale) => Some(scaled_dims(full_dimensions, scale)),
        j2k_jpeg::JpegDecodeOp::RegionScaled { roi, scale } => {
            let scaled = Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            }
            .scaled_covering(scale);
            Some((scaled.w, scaled.h))
        }
        j2k_jpeg::JpegDecodeOp::Region(_) => None,
    }
}

#[cfg(target_os = "macos")]
fn decoder_resident_sampling_family(decoder: &Decoder<'_>) -> batch::SamplingFamily {
    if decoder.fast420_packet().is_some() {
        batch::SamplingFamily::Fast420
    } else if decoder.fast422_packet().is_some() {
        batch::SamplingFamily::Fast422
    } else if decoder.fast444_packet().is_some() {
        batch::SamplingFamily::Fast444
    } else {
        batch::SamplingFamily::Other
    }
}

#[cfg(target_os = "macos")]
fn decoder_resident_restart_interval_mcus(decoder: &Decoder<'_>) -> u32 {
    if let Some(packet) = decoder.fast420_packet() {
        packet.restart_interval_mcus
    } else if let Some(packet) = decoder.fast422_packet() {
        packet.restart_interval_mcus
    } else if let Some(packet) = decoder.fast444_packet() {
        packet.restart_interval_mcus
    } else {
        0
    }
}

impl Codec {
    #[cfg(target_os = "macos")]
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

    #[cfg(target_os = "macos")]
    fn observe_rgb8_batch_output_dimensions(
        first_output_dimensions: &mut Option<(u32, u32)>,
        output_dimensions: (u32, u32),
    ) -> Result<(), Error> {
        if let Some(first) = *first_output_dimensions {
            if first != output_dimensions {
                return Err(Error::UnsupportedMetalRequest {
                    reason:
                        "JPEG Metal reusable RGB8 batch output requires matching output dimensions",
                });
            }
        } else {
            *first_output_dimensions = Some(output_dimensions);
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn rgb8_metal_batch_requests(
        inputs: &[&[u8]],
        mut op_for_decoder: impl FnMut(&CpuDecoder<'_>) -> batch::BatchOp,
    ) -> Result<Vec<batch::QueuedRequest>, Error> {
        let plan = Self::rgb8_metal_batch_requests_with_output_dimensions(inputs, |decoder| {
            (op_for_decoder(decoder), decoder.info().dimensions)
        })?;
        Ok(plan.requests)
    }

    #[cfg(target_os = "macos")]
    fn rgb8_metal_batch_requests_with_output_dimensions(
        inputs: &[&[u8]],
        mut op_and_dimensions_for_decoder: impl FnMut(&CpuDecoder<'_>) -> (batch::BatchOp, (u32, u32)),
    ) -> Result<Rgb8MetalBatchPlan, Error> {
        let mut state = session::SessionState::default();
        let mut requests = Vec::with_capacity(inputs.len());
        let mut first_output_dimensions = None;
        for input in inputs {
            let decoder = CpuDecoder::new(input)?;
            let (op, output_dimensions) = op_and_dimensions_for_decoder(&decoder);
            Self::observe_rgb8_batch_output_dimensions(
                &mut first_output_dimensions,
                output_dimensions,
            )?;
            let input = state.intern_input_slice(input);
            let (fast444_packet, fast422_packet, fast420_packet) =
                state.resolve_fast_packets(&input, BackendRequest::Metal);
            requests.push(batch::QueuedRequest::new_shared(
                input,
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                op,
                fast444_packet,
                fast422_packet,
                fast420_packet,
            ));
        }
        Ok(Rgb8MetalBatchPlan {
            requests,
            output_dimensions: first_output_dimensions,
        })
    }

    #[cfg(target_os = "macos")]
    fn rgb8_metal_decoder_batch_requests_with_output_dimensions(
        decoders: &[&Decoder<'_>],
        mut op_and_dimensions_for_decoder: impl FnMut(&Decoder<'_>) -> (batch::BatchOp, (u32, u32)),
    ) -> Result<Rgb8MetalBatchPlan, Error> {
        let mut requests = Vec::with_capacity(decoders.len());
        let mut first_output_dimensions = None;
        for decoder in decoders {
            let (op, output_dimensions) = op_and_dimensions_for_decoder(decoder);
            Self::observe_rgb8_batch_output_dimensions(
                &mut first_output_dimensions,
                output_dimensions,
            )?;
            requests.push(decoder.rgb8_metal_request(op));
        }
        Ok(Rgb8MetalBatchPlan {
            requests,
            output_dimensions: first_output_dimensions,
        })
    }

    #[cfg(target_os = "macos")]
    fn rgb8_batch_op_and_dimensions(
        op: Rgb8MetalBatchOp,
        dimensions: (u32, u32),
    ) -> (batch::BatchOp, (u32, u32)) {
        match op {
            Rgb8MetalBatchOp::Full => (batch::BatchOp::Full, dimensions),
            Rgb8MetalBatchOp::Scaled(scale) => {
                let (w, h) = dimensions;
                (
                    batch::BatchOp::RegionScaled {
                        roi: Rect { x: 0, y: 0, w, h },
                        scale,
                    },
                    scaled_dims((w, h), scale),
                )
            }
            Rgb8MetalBatchOp::RegionScaled { roi, scale } => {
                let scaled = roi.scaled_covering(scale);
                (
                    batch::BatchOp::RegionScaled { roi, scale },
                    (scaled.w, scaled.h),
                )
            }
        }
    }

    #[cfg(target_os = "macos")]
    fn rgb8_batch_jpeg_decode_op(op: Rgb8MetalBatchOp) -> j2k_jpeg::JpegDecodeOp {
        match op {
            Rgb8MetalBatchOp::Full => j2k_jpeg::JpegDecodeOp::Full,
            Rgb8MetalBatchOp::Scaled(scale) => j2k_jpeg::JpegDecodeOp::Scaled(scale),
            Rgb8MetalBatchOp::RegionScaled { roi, scale } => j2k_jpeg::JpegDecodeOp::RegionScaled {
                roi: roi.into(),
                scale,
            },
        }
    }

    #[cfg(target_os = "macos")]
    fn plan_rgb8_metal_batch(
        source: Rgb8MetalBatchSource<'_, '_>,
        op: Rgb8MetalBatchOp,
        track_output_dimensions: bool,
    ) -> Result<(Rgb8MetalBatchPlan, usize), Error> {
        match source {
            Rgb8MetalBatchSource::Bytes(inputs) => {
                if track_output_dimensions {
                    Self::rgb8_metal_batch_requests_with_output_dimensions(inputs, |decoder| {
                        Self::rgb8_batch_op_and_dimensions(op, decoder.info().dimensions)
                    })
                    .map(|plan| (plan, inputs.len()))
                } else {
                    Self::rgb8_metal_batch_requests(inputs, |decoder| {
                        Self::rgb8_batch_op_and_dimensions(op, decoder.info().dimensions).0
                    })
                    .map(|requests| {
                        (
                            Rgb8MetalBatchPlan {
                                requests,
                                output_dimensions: None,
                            },
                            inputs.len(),
                        )
                    })
                }
            }
            Rgb8MetalBatchSource::Decoders(decoders) => {
                Self::rgb8_metal_decoder_batch_requests_with_output_dimensions(
                    decoders,
                    |decoder| {
                        Self::rgb8_batch_op_and_dimensions(op, decoder.inner().info().dimensions)
                    },
                )
                .map(|plan| (plan, decoders.len()))
            }
        }
    }

    #[cfg(target_os = "macos")]
    const fn rgb8_buffer_batch_unsupported_reason(op: Rgb8MetalBatchOp) -> &'static str {
        match op {
            Rgb8MetalBatchOp::Full => {
                "JPEG Metal reusable batch output currently supports batchable full-tile RGB8 fast 4:2:0, 4:2:2, or 4:4:4 inputs"
            }
            Rgb8MetalBatchOp::Scaled(_) => {
                "JPEG Metal reusable scaled batch output currently supports batchable RGB8 fast 4:2:0, 4:2:2, or 4:4:4 inputs with half, quarter, or eighth scaling"
            }
            Rgb8MetalBatchOp::RegionScaled { .. } => {
                "JPEG Metal reusable region-scaled batch output currently supports batchable RGB8 fast 4:2:0, 4:2:2, or 4:4:4 inputs with matching output shapes"
            }
        }
    }

    #[cfg(target_os = "macos")]
    const fn rgb8_texture_batch_unsupported_reason(op: Rgb8MetalBatchOp) -> &'static str {
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

    #[cfg(target_os = "macos")]
    /// Decode a batched RGB8 JPEG request into a caller-owned Metal buffer.
    ///
    /// This is the single buffer-output entry point for full, scaled, and
    /// region-scaled batches sourced from raw bytes or pre-parsed decoders;
    /// `MetalBufferBatchTarget::Resizable` grows the buffer to fit before
    /// decoding.
    pub fn decode_rgb8_batch_into_buffer_with_session(
        request: Rgb8MetalBatchRequest<'_, '_>,
        target: MetalBufferBatchTarget<'_>,
        session: &MetalBackendSession,
    ) -> Result<Vec<Result<Surface, Error>>, Error> {
        if request.source.is_empty() {
            return Ok(Vec::new());
        }

        let resizable = matches!(target, MetalBufferBatchTarget::Resizable(_));
        let (plan, tile_count) =
            Self::plan_rgb8_metal_batch(request.source, request.op, resizable)?;
        let output: &MetalBatchOutputBuffer = match target {
            MetalBufferBatchTarget::Reusable(output) => output,
            MetalBufferBatchTarget::Resizable(output) => {
                if let Rgb8MetalBatchSource::Decoders(decoders) = request.source {
                    let report = Self::inspect_rgb8_decoder_batch_metal_output(
                        decoders,
                        Self::rgb8_batch_jpeg_decode_op(request.op),
                    );
                    output.ensure_rgb8_batch_report(session, &report)?;
                }
                let Some(output_dimensions) = plan.output_dimensions else {
                    return Ok(Vec::new());
                };
                output.ensure_rgb8_tiles(session, output_dimensions, tile_count)?;
                output
            }
        };

        let results = match request.op {
            Rgb8MetalBatchOp::Full => compute::decode_full_rgb8_batch_into_output_with_session(
                &plan.requests,
                output,
                session,
            )?,
            Rgb8MetalBatchOp::Scaled(_) | Rgb8MetalBatchOp::RegionScaled { .. } => {
                compute::decode_region_scaled_rgb8_batch_into_output_with_session(
                    &plan.requests,
                    output,
                    session,
                )?
            }
        };
        results.ok_or(Error::UnsupportedMetalRequest {
            reason: Self::rgb8_buffer_batch_unsupported_reason(request.op),
        })
    }

    #[cfg(target_os = "macos")]
    /// Decode a batched RGB8 JPEG request into caller-owned Metal RGBA8 textures.
    ///
    /// This is the single texture-output entry point for full, scaled, and
    /// region-scaled batches sourced from raw bytes or pre-parsed decoders;
    /// `MetalTextureBatchTarget::Resizable` grows the texture set to fit
    /// before decoding.
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
            Rgb8MetalBatchOp::Full => compute::decode_full_rgb8_batch_into_textures_with_session(
                &plan.requests,
                output,
                session,
            )?,
            Rgb8MetalBatchOp::Scaled(_) | Rgb8MetalBatchOp::RegionScaled { .. } => {
                compute::decode_region_scaled_rgb8_batch_into_textures_with_session(
                    &plan.requests,
                    output,
                    session,
                )?
            }
        };
        results.ok_or(Error::UnsupportedMetalRequest {
            reason: Self::rgb8_texture_batch_unsupported_reason(request.op),
        })
    }

    #[cfg(target_os = "macos")]
    /// Decode a full-tile RGB8 JPEG decoder batch into resizable caller-owned
    /// Metal RGBA8 textures.
    ///
    /// Convenience wrapper over [`Codec::decode_rgb8_batch_into_textures_with_session`]
    /// for the resident whole-slide tile path.
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

    #[cfg(target_os = "macos")]
    /// Decode a region-scaled RGB8 JPEG batch into a resizable caller-owned
    /// Metal buffer.
    ///
    /// Convenience wrapper over [`Codec::decode_rgb8_batch_into_buffer_with_session`]
    /// for the viewport composition path.
    pub fn decode_rgb8_region_scaled_batch_into_resizable_metal_buffer_with_session(
        inputs: &[&[u8]],
        roi: Rect,
        scale: Downscale,
        output: &mut MetalBatchOutputBuffer,
        session: &MetalBackendSession,
    ) -> Result<Vec<Result<Surface, Error>>, Error> {
        Self::decode_rgb8_batch_into_buffer_with_session(
            Rgb8MetalBatchRequest {
                source: Rgb8MetalBatchSource::Bytes(inputs),
                op: Rgb8MetalBatchOp::RegionScaled { roi, scale },
            },
            MetalBufferBatchTarget::Resizable(output),
            session,
        )
    }

    #[cfg(target_os = "macos")]
    /// Decode a region-scaled RGB8 JPEG batch into resizable caller-owned
    /// Metal RGBA8 textures.
    ///
    /// Convenience wrapper over [`Codec::decode_rgb8_batch_into_textures_with_session`]
    /// for the viewport composition path.
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

    /// Submit a tile decode request into a reusable Metal session.
    #[doc(hidden)]
    pub fn submit_tile_request_to_device(
        ctx: &mut j2k_core::DecoderContext<CpuDecoderContext>,
        session: &mut MetalSession,
        pool: &mut CpuScratchPool,
        input: &[u8],
        request: MetalDecodeRequest,
    ) -> Result<<Self as TileBatchDecodeSubmit>::SubmittedSurface, Error> {
        let _ = (ctx, pool);
        let slot = {
            let mut state = session.shared.lock()?;
            let input = state.intern_input_slice(input);
            let (fast444_packet, fast422_packet, fast420_packet) =
                state.resolve_fast_packets(&input, request.backend);
            state.queue_request(batch::QueuedRequest::new_shared(
                input,
                request.fmt,
                request.backend,
                request.op.batch_op(),
                fast444_packet,
                fast422_packet,
                fast420_packet,
            ))
        };
        Ok(batch::MetalSubmission {
            session: session.shared.clone(),
            slot,
        })
    }
}
