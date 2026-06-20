// SPDX-License-Identifier: Apache-2.0

use std::{ffi::OsStr, time::Duration};

use j2k_jpeg::adapter::JpegEntropyCheckpointV1;
use j2k_jpeg::Decoder as CpuDecoder;
use metal::Buffer;

use crate::buffers::MetalBatchScratch;
use crate::{batch, Error, Surface};

use super::{
    checked_u32, decode_error_from_cpu, entropy_checkpoint_hosts, first_decode_error_status,
    MetalRuntime,
};

const FAST420_BATCH_TIMING_ENV: &str = "J2K_JPEG_METAL_FAST420_BATCH_TIMING";

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FastBatchDecodeMode {
    Fused,
    #[cfg(test)]
    SplitCoeffIdct,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct FastBatchTiming {
    pub(super) accepted: Duration,
    pub(super) entropy_concat: Duration,
    pub(super) buffer_alloc: Duration,
    pub(super) encode_decode: Duration,
    pub(super) wait_decode: Duration,
    pub(super) encode_pack: Duration,
    pub(super) wait_pack: Duration,
    pub(super) total: Duration,
}

#[cfg(target_os = "macos")]
impl FastBatchTiming {
    fn micros(duration: Duration) -> u128 {
        duration.as_micros()
    }

    pub(super) fn log(
        self,
        tag: &'static str,
        label: &str,
        tile_count: usize,
        dimensions: (u32, u32),
        segment_count: usize,
    ) {
        j2k_profile::emit_profile_row_now(
            "jpeg",
            "decode",
            tag,
            &[
                ("mode", label.to_string()),
                ("tiles", tile_count.to_string()),
                ("dimensions", format!("{}x{}", dimensions.0, dimensions.1)),
                ("segments", segment_count.to_string()),
                ("accepted_us", Self::micros(self.accepted).to_string()),
                (
                    "entropy_concat_us",
                    Self::micros(self.entropy_concat).to_string(),
                ),
                (
                    "buffer_alloc_us",
                    Self::micros(self.buffer_alloc).to_string(),
                ),
                (
                    "encode_decode_us",
                    Self::micros(self.encode_decode).to_string(),
                ),
                ("wait_decode_us", Self::micros(self.wait_decode).to_string()),
                ("encode_pack_us", Self::micros(self.encode_pack).to_string()),
                ("wait_pack_us", Self::micros(self.wait_pack).to_string()),
                ("total_us", Self::micros(self.total).to_string()),
            ],
        );
    }
}

#[cfg(target_os = "macos")]
pub(super) fn fast_batch_decode_mode() -> FastBatchDecodeMode {
    FastBatchDecodeMode::Fused
}

#[cfg(target_os = "macos")]
pub(super) fn fast420_batch_timing_enabled() -> bool {
    fast420_batch_timing_value_enabled(std::env::var_os(FAST420_BATCH_TIMING_ENV).as_deref())
}

#[cfg(target_os = "macos")]
pub(super) fn fast420_batch_timing_value_enabled(value: Option<&OsStr>) -> bool {
    value.is_some_and(|value| value == OsStr::new("1"))
}

#[cfg(target_os = "macos")]
pub(super) struct BatchEntropyBuffers {
    pub(super) payload: Buffer,
    pub(super) offsets: Buffer,
    pub(super) lens: Buffer,
    pub(super) checkpoints: Buffer,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(super) struct BatchEntropyBufferKeys {
    pub(super) payload: &'static str,
    pub(super) offsets: &'static str,
    pub(super) lens: &'static str,
    pub(super) checkpoints: &'static str,
}

#[cfg(target_os = "macos")]
pub(super) fn batch_entropy_buffers<'a>(
    runtime: &MetalRuntime,
    scratch: &mut MetalBatchScratch,
    keys: BatchEntropyBufferKeys,
    entropy_bytes_iter: impl Iterator<Item = &'a [u8]> + Clone,
    entropy_checkpoints_iter: impl Iterator<Item = &'a [JpegEntropyCheckpointV1]> + Clone,
    tile_count: usize,
    segment_count: usize,
) -> Result<Option<BatchEntropyBuffers>, Error> {
    let total_entropy_len = entropy_bytes_iter
        .clone()
        .map(<[u8]>::len)
        .try_fold(0usize, usize::checked_add)
        .ok_or_else(|| Error::MetalKernel {
            message: "JPEG Metal region scaled batch entropy length overflowed".to_string(),
        })?;
    if total_entropy_len == 0 {
        return Ok(None);
    }

    let mut entropy_bytes = Vec::with_capacity(total_entropy_len);
    let mut entropy_offsets = Vec::with_capacity(tile_count);
    let mut entropy_lens = Vec::with_capacity(tile_count);
    let mut entropy_checkpoints = Vec::with_capacity(tile_count * segment_count);
    for (bytes, checkpoints) in entropy_bytes_iter.zip(entropy_checkpoints_iter) {
        entropy_offsets.push(checked_u32(
            entropy_bytes.len(),
            "region scaled batch entropy offset",
        )?);
        entropy_lens.push(checked_u32(
            bytes.len(),
            "region scaled batch entropy length",
        )?);
        entropy_bytes.extend_from_slice(bytes);
        entropy_checkpoints.extend(checkpoints.iter().copied());
    }

    let checkpoints = entropy_checkpoint_hosts(&entropy_checkpoints)?;
    Ok(Some(BatchEntropyBuffers {
        payload: scratch.shared_buffer_with_bytes(&runtime.device, keys.payload, &entropy_bytes),
        offsets: scratch.shared_buffer_with_slice(&runtime.device, keys.offsets, &entropy_offsets),
        lens: scratch.shared_buffer_with_slice(&runtime.device, keys.lens, &entropy_lens),
        checkpoints: scratch.shared_buffer_with_slice(
            &runtime.device,
            keys.checkpoints,
            &checkpoints,
        ),
    }))
}

#[cfg(target_os = "macos")]
pub(super) fn region_scaled_batch_error_results(
    requests: &[batch::QueuedRequest],
    status_buffer: &Buffer,
    total_decode_threads: u32,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    let Some(status) = first_decode_error_status(status_buffer, total_decode_threads) else {
        return Ok(None);
    };
    let mut results = Vec::with_capacity(requests.len());
    for request in requests {
        let decoder = CpuDecoder::new(request.input.as_ref())?;
        results.push(Err(decode_error_from_cpu(&decoder, request.fmt, status)));
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
pub(super) fn texture_batch_error_results(
    requests: &[batch::QueuedRequest],
    status_buffer: &Buffer,
    total_decode_threads: u32,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    let Some(status) = first_decode_error_status(status_buffer, total_decode_threads) else {
        return Ok(None);
    };
    let mut results = Vec::with_capacity(requests.len());
    for request in requests {
        let decoder = CpuDecoder::new(request.input.as_ref())?;
        results.push(Err(decode_error_from_cpu(&decoder, request.fmt, status)));
    }
    Ok(Some(results))
}
