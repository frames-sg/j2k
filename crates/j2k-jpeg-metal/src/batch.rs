// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(all(test, target_os = "macos"))]
use std::sync::Arc;

use j2k_core::{BackendRequest, DeviceSubmission, Downscale, PixelFormat, Rect};
use j2k_jpeg::adapter::JpegPlanCacheError;
#[cfg(all(test, target_os = "macos"))]
use j2k_jpeg::adapter::{JpegFast420PacketV1, JpegFast422PacketV1, JpegFast444PacketV1};

use crate::{session::SharedSession, Error, SharedJpegFastPacket, SharedJpegInput, Surface};

mod flush;
mod grouping;

use flush::{flush_if_needed, take_surface};
#[cfg(test)]
use grouping::grouped_request_metadata_bytes;
use grouping::{add_execution_external_live_bytes, group_compatible_requests};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BatchOp {
    Full,
    Region(Rect),
    Scaled(Downscale),
    RegionScaled { roi: Rect, scale: Downscale },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct BatchKey {
    fmt: PixelFormat,
    backend: BackendRequest,
    kind: BatchKind,
    shape: BatchShape,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum BatchKind {
    Full,
    Region { dims: (u32, u32) },
    Scaled { scale: Downscale },
    RegionScaled { dims: (u32, u32), scale: Downscale },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum SamplingFamily {
    Unknown,
    Fast420,
    Fast422,
    Fast444,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PlaneModeHint {
    Unknown,
    YCbCr,
    Rgb,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct BatchShape {
    pub(crate) restart_interval: Option<u16>,
    pub(crate) checkpoint_count: usize,
    pub(crate) sampling_family: SamplingFamily,
    pub(crate) plane_mode: PlaneModeHint,
}

impl BatchShape {
    pub(crate) const fn unknown() -> Self {
        Self {
            restart_interval: None,
            checkpoint_count: 0,
            sampling_family: SamplingFamily::Unknown,
            plane_mode: PlaneModeHint::Unknown,
        }
    }

    pub(crate) const fn from_summary(
        summary: j2k_jpeg::adapter::DeviceBatchSummary,
        color_space: j2k_jpeg::ColorSpace,
    ) -> Self {
        Self {
            restart_interval: summary.restart_interval,
            checkpoint_count: summary.checkpoint_count,
            sampling_family: if summary.matches_fast_420 {
                SamplingFamily::Fast420
            } else if summary.matches_fast_422 {
                SamplingFamily::Fast422
            } else if summary.matches_fast_444 {
                SamplingFamily::Fast444
            } else {
                SamplingFamily::Other
            },
            plane_mode: match color_space {
                j2k_jpeg::ColorSpace::YCbCr => PlaneModeHint::YCbCr,
                j2k_jpeg::ColorSpace::Rgb => PlaneModeHint::Rgb,
                j2k_jpeg::ColorSpace::Grayscale
                | j2k_jpeg::ColorSpace::Cmyk
                | j2k_jpeg::ColorSpace::Ycck => PlaneModeHint::Unknown,
            },
        }
    }
}

#[derive(Clone)]
pub(crate) struct QueuedRequest {
    pub(crate) input: SharedJpegInput,
    pub(crate) fmt: PixelFormat,
    pub(crate) backend: BackendRequest,
    pub(crate) op: BatchOp,
    pub(crate) fast_packet: Option<SharedJpegFastPacket>,
    shape: BatchShape,
    execution_cache_retained_bytes: usize,
    execution_external_live_bytes: usize,
    execution_collective_owner_bytes: usize,
    pub(crate) output_slot: usize,
}

impl QueuedRequest {
    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn new(
        input: Arc<[u8]>,
        fmt: PixelFormat,
        backend: BackendRequest,
        op: BatchOp,
        fast444_packet: Option<Arc<JpegFast444PacketV1>>,
        fast422_packet: Option<Arc<JpegFast422PacketV1>>,
        fast420_packet: Option<Arc<JpegFast420PacketV1>>,
    ) -> Self {
        let decoder = j2k_jpeg::Decoder::new(input.as_ref()).expect("queued test JPEG decoder");
        let shape = BatchShape::from_summary(
            j2k_jpeg::adapter::summarize_device_batch(&decoder, 4),
            decoder.info().color_space,
        );
        let fast_packet = crate::fast_packets::build_test_shared_fast_packet(
            input.as_ref(),
            fast444_packet,
            fast422_packet,
            fast420_packet,
        );
        let shared_input = SharedJpegInput::try_copy_from_slice(input.as_ref())
            .expect("queued test shared JPEG input");
        drop(input);
        Self {
            input: shared_input,
            fmt,
            backend,
            op,
            fast_packet,
            shape,
            execution_cache_retained_bytes: 0,
            execution_external_live_bytes: 0,
            execution_collective_owner_bytes: 0,
            output_slot: usize::MAX,
        }
    }

    pub(crate) fn new_shared(
        input: SharedJpegInput,
        fmt: PixelFormat,
        backend: BackendRequest,
        op: BatchOp,
        fast_packet: Option<SharedJpegFastPacket>,
        shape: BatchShape,
    ) -> Self {
        Self {
            input,
            fmt,
            backend,
            op,
            fast_packet,
            shape,
            execution_cache_retained_bytes: 0,
            execution_external_live_bytes: 0,
            execution_collective_owner_bytes: 0,
            output_slot: usize::MAX,
        }
    }

    pub(crate) fn with_output_slot(mut self, output_slot: usize) -> Self {
        self.output_slot = output_slot;
        self
    }

    pub(crate) fn key(&self) -> BatchKey {
        BatchKey {
            fmt: self.fmt,
            backend: self.backend,
            kind: match self.op {
                BatchOp::Full => BatchKind::Full,
                BatchOp::Region(roi) => BatchKind::Region {
                    dims: (roi.w, roi.h),
                },
                BatchOp::Scaled(scale) => BatchKind::Scaled { scale },
                BatchOp::RegionScaled { roi, scale } => {
                    let scaled = roi.scaled_covering(scale);
                    BatchKind::RegionScaled {
                        dims: (scaled.w, scaled.h),
                        scale,
                    }
                }
            },
            shape: self.shape,
        }
    }

    #[cfg(target_os = "macos")]
    pub(crate) const fn plane_mode_hint(&self) -> PlaneModeHint {
        self.shape.plane_mode
    }

    pub(crate) const fn set_execution_owner_baseline(
        &mut self,
        cache_retained_bytes: usize,
        external_live_bytes: usize,
    ) {
        self.execution_cache_retained_bytes = cache_retained_bytes;
        self.execution_external_live_bytes = external_live_bytes;
    }

    pub(crate) const fn execution_cache_retained_bytes(&self) -> usize {
        self.execution_cache_retained_bytes
    }

    pub(crate) const fn execution_external_live_bytes(&self) -> usize {
        self.execution_external_live_bytes
    }

    pub(crate) const fn execution_collective_owner_bytes(&self) -> usize {
        self.execution_collective_owner_bytes
    }

    pub(crate) fn retained_input_bytes(&self) -> Result<usize, JpegPlanCacheError> {
        self.input.retained_cache_bytes()
    }

    pub(crate) fn retained_packet_bytes(&self) -> Result<usize, JpegPlanCacheError> {
        self.fast_packet
            .as_ref()
            .map_or(Ok(0), SharedJpegFastPacket::retained_cache_bytes)
    }
}

pub(crate) fn execution_cache_retained_bytes(requests: &[QueuedRequest]) -> Result<usize, Error> {
    let retained_bytes = requests
        .first()
        .map_or(0, QueuedRequest::execution_cache_retained_bytes);
    if requests
        .iter()
        .any(|request| request.execution_cache_retained_bytes() != retained_bytes)
    {
        return Err(j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
            "JPEG Metal batch mixes different live-cache owner baselines",
        )
        .into());
    }
    Ok(retained_bytes)
}

pub(crate) fn execution_owner_baseline(
    requests: &[QueuedRequest],
) -> Result<(usize, usize, usize), Error> {
    let cache_retained_bytes = execution_cache_retained_bytes(requests)?;
    let external_live_bytes = requests
        .first()
        .map_or(0, QueuedRequest::execution_external_live_bytes);
    if requests
        .iter()
        .any(|request| request.execution_external_live_bytes() != external_live_bytes)
    {
        return Err(j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
            "JPEG Metal batch mixes different external owner baselines",
        )
        .into());
    }
    let collective_owner_bytes = requests
        .first()
        .map_or(0, QueuedRequest::execution_collective_owner_bytes);
    if requests
        .iter()
        .any(|request| request.execution_collective_owner_bytes() != collective_owner_bytes)
    {
        return Err(j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
            "JPEG Metal batch mixes different collective owner baselines",
        )
        .into());
    }
    Ok((
        cache_retained_bytes,
        external_live_bytes,
        collective_owner_bytes,
    ))
}

pub(crate) fn stamp_execution_owner_baseline(
    requests: &mut [QueuedRequest],
    cache_retained_bytes: usize,
    external_live_bytes: usize,
) {
    for request in requests {
        request.set_execution_owner_baseline(cache_retained_bytes, external_live_bytes);
    }
}

pub(crate) fn stamp_execution_collective_owner_bytes(
    requests: &mut [QueuedRequest],
    collective_owner_bytes: usize,
) {
    for request in requests {
        request.execution_collective_owner_bytes = collective_owner_bytes;
    }
}

pub struct MetalSubmission {
    pub(crate) session: SharedSession,
    pub(crate) slot: usize,
}

#[doc(hidden)]
impl DeviceSubmission for MetalSubmission {
    type Output = Surface;
    type Error = Error;

    fn wait(self) -> Result<Self::Output, Self::Error> {
        let mut session = self.session.lock()?;
        flush_if_needed(&mut session)?;
        take_surface(&mut session, self.slot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use j2k_core::{BufferError, DEFAULT_MAX_HOST_ALLOCATION_BYTES};
    use j2k_jpeg::{JpegEncodeError, JpegError};

    #[test]
    fn batched_decode_error_cloning_preserves_metal_unavailable() {
        assert!(matches!(
            Error::MetalUnavailable.clone(),
            Error::MetalUnavailable
        ));
    }

    #[test]
    fn batched_decode_error_cloning_preserves_routing_errors() {
        assert!(matches!(
            Error::UnsupportedBackend {
                request: BackendRequest::Cuda,
            }
            .clone(),
            Error::UnsupportedBackend {
                request: BackendRequest::Cuda
            }
        ));
        assert!(matches!(
            Error::UnsupportedMetalRequest {
                reason: "unsupported test shape",
            }
            .clone(),
            Error::UnsupportedMetalRequest {
                reason: "unsupported test shape"
            }
        ));
    }

    #[test]
    fn batched_decode_error_cloning_preserves_typed_codec_and_buffer_failures() {
        let errors = [
            Error::Decode(JpegError::UnexpectedEoi {
                mcu_at: 3,
                mcu_total: 4,
            }),
            Error::Encode(JpegEncodeError::HostAllocationFailed { bytes: 4096 }),
            Error::Buffer(BufferError::OutputTooSmall {
                required: 8,
                have: 7,
            }),
        ];

        assert!(matches!(
            errors[0].clone(),
            Error::Decode(JpegError::UnexpectedEoi {
                mcu_at: 3,
                mcu_total: 4
            })
        ));
        assert!(matches!(
            errors[1].clone(),
            Error::Encode(JpegEncodeError::HostAllocationFailed { bytes: 4096 })
        ));
        assert!(matches!(
            errors[2].clone(),
            Error::Buffer(BufferError::OutputTooSmall {
                required: 8,
                have: 7
            })
        ));
    }

    #[test]
    fn grouped_execution_persists_all_group_metadata_and_owner_baselines() {
        const JPEG: &[u8] = include_bytes!("../fixtures/jpeg/baseline_420_16x16.jpg");
        let input = SharedJpegInput::try_copy_from_slice(JPEG).expect("shared JPEG input");
        let mut queued = vec![
            QueuedRequest::new_shared(
                input.clone(),
                PixelFormat::Rgb8,
                BackendRequest::Cpu,
                BatchOp::Full,
                None,
                BatchShape::unknown(),
            ),
            QueuedRequest::new_shared(
                input.clone(),
                PixelFormat::Gray8,
                BackendRequest::Cpu,
                BatchOp::Full,
                None,
                BatchShape::unknown(),
            ),
            QueuedRequest::new_shared(
                input,
                PixelFormat::Rgba8,
                BackendRequest::Cpu,
                BatchOp::Full,
                None,
                BatchShape::unknown(),
            ),
        ];
        let batches = group_compatible_requests(&mut queued).expect("three compatible groups");
        assert_eq!(batches.len(), 3);
        let grouped_bytes =
            grouped_request_metadata_bytes(batches.capacity(), &batches).expect("group bytes");
        let collective_owner_bytes = batches[0][0].execution_collective_owner_bytes();
        assert!(collective_owner_bytes > 0);
        for request in batches.iter().flatten() {
            assert_eq!(
                request.execution_collective_owner_bytes(),
                collective_owner_bytes
            );
            assert_eq!(request.execution_external_live_bytes(), grouped_bytes);
        }

        let base_live = crate::plan_owner_ledger::batch_execution_budget(
            "JPEG Metal grouped execution exact bound",
            &batches[0],
        )
        .expect("base grouped execution")
        .live_bytes();
        let exact_additional = DEFAULT_MAX_HOST_ALLOCATION_BYTES - base_live;
        let mut exact = batches[0].clone();
        add_execution_external_live_bytes(&mut exact, exact_additional)
            .expect("exact grouped baseline increment");
        assert_eq!(
            crate::plan_owner_ledger::batch_execution_budget(
                "JPEG Metal grouped execution exact bound",
                &exact,
            )
            .expect("all remaining groups at exact cap")
            .live_bytes(),
            DEFAULT_MAX_HOST_ALLOCATION_BYTES
        );

        let mut over = batches[0].clone();
        add_execution_external_live_bytes(&mut over, exact_additional + 1)
            .expect("one-over baseline increment");
        assert!(matches!(
            crate::plan_owner_ledger::batch_execution_budget(
                "JPEG Metal grouped execution exact bound",
                &over,
            ),
            Err(Error::BatchInfrastructure(
                j2k_core::BatchInfrastructureError::AllocationTooLarge {
                    what: "JPEG Metal grouped execution exact bound",
                    requested,
                    cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                }
            )) if requested == DEFAULT_MAX_HOST_ALLOCATION_BYTES + 1
        ));
    }
}
