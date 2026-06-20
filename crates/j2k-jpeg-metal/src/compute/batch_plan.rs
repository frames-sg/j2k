// SPDX-License-Identifier: Apache-2.0

use j2k_core::{BackendRequest, Rect};
use j2k_jpeg::{
    adapter::{
        JpegEntropyCheckpointV1, JpegFast420PacketV1, JpegFast422PacketV1, JpegFast444PacketV1,
    },
    Decoder as CpuDecoder,
};
use metal::{Buffer, MTLResourceOptions};

use super::{entropy_checkpoints_buffer, fast444_plane_mode, pixel_format_to_out_format};
use super::{MetalRuntime, PlaneMode};
use crate::{batch, Error, Surface};

const AUTO_METAL_MIN_BATCH_REQUESTS: usize = 8;
const AUTO_METAL_MIN_BATCH_EDGE: u32 = 512;

pub(super) enum BatchedFastPacket<'a> {
    Fast420(&'a JpegFast420PacketV1),
    Fast422(&'a JpegFast422PacketV1),
    Fast444(&'a JpegFast444PacketV1, PlaneMode),
}

pub(super) struct BatchedDecodeItem {
    pub(super) request_index: usize,
    pub(super) surface: Surface,
    pub(super) status_buffer: Buffer,
    pub(super) decode_threads: u32,
    pub(super) _decode_resources: Vec<Buffer>,
}

#[derive(Default)]
pub(super) struct BatchDeviceBufferCache {
    packet_buffers: Vec<SharedPacketDeviceBuffers>,
}

struct SharedPacketDeviceBuffers {
    entropy_ptr: usize,
    entropy_len: usize,
    checkpoints_ptr: usize,
    checkpoints_len: usize,
    entropy_buffer: Buffer,
    entropy_checkpoints_buffer: Buffer,
}

impl BatchDeviceBufferCache {
    pub(super) fn packet_buffers(
        &mut self,
        runtime: &MetalRuntime,
        entropy_bytes: &[u8],
        entropy_checkpoints: &[JpegEntropyCheckpointV1],
    ) -> Result<(Buffer, Buffer), Error> {
        let entropy_ptr = entropy_bytes.as_ptr() as usize;
        let entropy_len = entropy_bytes.len();
        let checkpoints_ptr = entropy_checkpoints.as_ptr() as usize;
        let checkpoints_len = entropy_checkpoints.len();
        if let Some(entry) = self.packet_buffers.iter().find(|entry| {
            entry.entropy_ptr == entropy_ptr
                && entry.entropy_len == entropy_len
                && entry.checkpoints_ptr == checkpoints_ptr
                && entry.checkpoints_len == checkpoints_len
        }) {
            return Ok((
                entry.entropy_buffer.clone(),
                entry.entropy_checkpoints_buffer.clone(),
            ));
        }

        let entropy_buffer = runtime.device.new_buffer_with_data(
            entropy_bytes.as_ptr().cast(),
            entropy_bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let entropy_checkpoints_buffer =
            entropy_checkpoints_buffer(&runtime.device, entropy_checkpoints)?;
        self.packet_buffers.push(SharedPacketDeviceBuffers {
            entropy_ptr,
            entropy_len,
            checkpoints_ptr,
            checkpoints_len,
            entropy_buffer: entropy_buffer.clone(),
            entropy_checkpoints_buffer: entropy_checkpoints_buffer.clone(),
        });
        Ok((entropy_buffer, entropy_checkpoints_buffer))
    }
}

fn request_allows_batched_packet(
    requests: &[batch::QueuedRequest],
    request: &batch::QueuedRequest,
    restart_interval_mcus: u32,
    dimensions: (u32, u32),
) -> bool {
    match request.backend {
        BackendRequest::Metal => true,
        BackendRequest::Auto => match request.op {
            batch::BatchOp::RegionScaled { .. } => false,
            _ => {
                requests.len() >= AUTO_METAL_MIN_BATCH_REQUESTS
                    && (restart_interval_mcus != 0
                        || auto_batch_work_is_large_enough(request, dimensions))
            }
        },
        BackendRequest::Cpu | BackendRequest::Cuda => false,
    }
}

fn auto_batch_work_is_large_enough(request: &batch::QueuedRequest, dimensions: (u32, u32)) -> bool {
    let dims = match request.op {
        batch::BatchOp::Full | batch::BatchOp::Scaled(_) => dimensions,
        batch::BatchOp::Region(roi) => (roi.w, roi.h),
        batch::BatchOp::RegionScaled { .. } => return false,
    };
    dims.0 >= AUTO_METAL_MIN_BATCH_EDGE && dims.1 >= AUTO_METAL_MIN_BATCH_EDGE
}

pub(super) fn batched_fast_packets(
    requests: &[batch::QueuedRequest],
) -> Result<Option<Vec<BatchedFastPacket<'_>>>, Error> {
    if requests.is_empty() {
        return Ok(None);
    }

    let mut packets = Vec::with_capacity(requests.len());
    for request in requests {
        let batchable_op = match request.op {
            batch::BatchOp::Full
            | batch::BatchOp::Region(_)
            | batch::BatchOp::Scaled(
                j2k_core::Downscale::Half
                | j2k_core::Downscale::Quarter
                | j2k_core::Downscale::Eighth,
            )
            | batch::BatchOp::RegionScaled {
                scale:
                    j2k_core::Downscale::Half
                    | j2k_core::Downscale::Quarter
                    | j2k_core::Downscale::Eighth,
                ..
            } => true,
            batch::BatchOp::Scaled(_) | batch::BatchOp::RegionScaled { .. } => false,
        };
        if !batchable_op
            || !matches!(
                request.backend,
                BackendRequest::Auto | BackendRequest::Metal
            )
            || pixel_format_to_out_format(request.fmt).is_none()
        {
            return Ok(None);
        }

        if let Some(packet) = request.fast420_packet.as_deref() {
            if !request_allows_batched_packet(
                requests,
                request,
                packet.restart_interval_mcus,
                packet.dimensions,
            ) {
                return Ok(None);
            }
            packets.push(BatchedFastPacket::Fast420(packet));
            continue;
        }

        if let Some(packet) = request.fast422_packet.as_deref() {
            if !request_allows_batched_packet(
                requests,
                request,
                packet.restart_interval_mcus,
                packet.dimensions,
            ) {
                return Ok(None);
            }
            packets.push(BatchedFastPacket::Fast422(packet));
            continue;
        }

        if let Some(packet) = request.fast444_packet.as_deref() {
            if !request_allows_batched_packet(
                requests,
                request,
                packet.restart_interval_mcus,
                packet.dimensions,
            ) {
                return Ok(None);
            }
            let decoder = CpuDecoder::new(request.input.as_ref())?;
            packets.push(BatchedFastPacket::Fast444(
                packet,
                fast444_plane_mode(&decoder),
            ));
            continue;
        }

        return Ok(None);
    }

    Ok(Some(packets))
}

pub(super) fn core_rect_to_jpeg(rect: Rect) -> j2k_jpeg::Rect {
    j2k_jpeg::Rect {
        x: rect.x,
        y: rect.y,
        w: rect.w,
        h: rect.h,
    }
}
