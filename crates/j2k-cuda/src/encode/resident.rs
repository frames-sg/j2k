// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::Duration;

use j2k_core::{DeviceSubmission, PixelFormat, ReadySubmission};
use j2k_cuda_runtime::CudaDeviceBuffer;

use crate::runtime::cuda_error;

use super::CudaEncodeStageTimings;

/// CUDA-resident lossless J2K/HTJ2K encode input tile.
#[derive(Debug, Clone, Copy)]
pub struct CudaLosslessEncodeTile<'a> {
    /// Source CUDA buffer containing interleaved Gray/RGB/RGBA pixels.
    pub buffer: &'a CudaDeviceBuffer,
    /// Byte offset of the first source pixel in `buffer`.
    pub byte_offset: usize,
    /// Width of the valid input region in pixels.
    pub width: u32,
    /// Height of the valid input region in pixels.
    pub height: u32,
    /// Number of bytes between consecutive input rows.
    pub pitch_bytes: usize,
    /// Encoded image width in pixels.
    pub output_width: u32,
    /// Encoded image height in pixels.
    pub output_height: u32,
    /// Pixel format of the source buffer.
    pub format: PixelFormat,
}

/// Residency decisions used by a lossless CUDA device-buffer encode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaLosslessEncodeResidency {
    /// Whether coefficient preparation ran on CUDA.
    pub coefficient_prep_used: bool,
    /// Whether packetization ran on CUDA.
    pub packetization_used: bool,
    /// Whether final codestream assembly stayed resident on CUDA.
    pub codestream_assembly_used: bool,
}

/// Lossless CUDA device-buffer encode output with host codestream bytes and timings.
#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)]
pub struct CudaLosslessEncodeOutcome {
    /// Encoded J2K codestream.
    pub encoded: j2k::EncodedJ2k,
    /// Whether the input buffer had to be copied or padded.
    pub input_copy_used: bool,
    /// Residency decisions for encode stages.
    pub resident: CudaLosslessEncodeResidency,
    /// Time spent copying or padding input.
    pub input_copy_duration: Duration,
    /// End-to-end encode duration for this tile.
    pub encode_duration: Duration,
    /// GPU-only duration when timestamp data is available.
    pub gpu_duration: Option<Duration>,
    /// Time spent validating encoded output.
    pub validation_duration: Duration,
    /// Time spent materializing CUDA output into host codestream bytes.
    pub host_readback_duration: Duration,
    /// CUDA encode stage timing buckets collected for this tile.
    pub stage_timings: CudaEncodeStageTimings,
}

/// CUDA-resident codestream bytes produced by a CUDA lossless encode.
#[derive(Debug)]
pub struct CudaResidentCodestreamBuffer {
    pub(super) buffer: CudaDeviceBuffer,
    pub(super) byte_len: usize,
}

impl CudaResidentCodestreamBuffer {
    /// CUDA buffer containing the codestream bytes.
    pub fn buffer(&self) -> &CudaDeviceBuffer {
        &self.buffer
    }

    /// Codestream byte length.
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    /// Download the resident codestream bytes.
    pub fn download(&self) -> Result<Vec<u8>, crate::Error> {
        let mut bytes = vec![0u8; self.byte_len];
        self.buffer.copy_to_host(&mut bytes).map_err(cuda_error)?;
        Ok(bytes)
    }

    /// Consume this value and return the owned CUDA buffer.
    pub fn into_buffer(self) -> CudaDeviceBuffer {
        self.buffer
    }
}

/// CUDA lossless encode output with host metadata and CUDA-resident codestream bytes.
#[derive(Debug)]
pub struct CudaEncodedJ2k {
    /// Host-visible encode metadata and codestream bytes.
    pub encoded: j2k::EncodedJ2k,
    /// CUDA-resident codestream bytes.
    pub codestream: CudaResidentCodestreamBuffer,
}

impl CudaEncodedJ2k {
    /// Borrow the host-visible encoded J2K metadata and bytes.
    pub fn encoded(&self) -> &j2k::EncodedJ2k {
        &self.encoded
    }

    /// Borrow the CUDA-resident codestream buffer.
    pub fn codestream(&self) -> &CudaResidentCodestreamBuffer {
        &self.codestream
    }

    /// Consume this value and return host metadata plus the CUDA-resident buffer.
    pub fn into_parts(self) -> (j2k::EncodedJ2k, CudaResidentCodestreamBuffer) {
        (self.encoded, self.codestream)
    }
}

/// Lossless CUDA device-buffer encode output with CUDA-resident codestream bytes.
#[derive(Debug)]
#[doc(hidden)]
pub struct CudaLosslessBufferEncodeOutcome {
    /// CUDA-resident encoded J2K output.
    pub encoded: CudaEncodedJ2k,
    /// Host-codestream encode outcome used to assemble and validate metadata.
    pub host_outcome: CudaLosslessEncodeOutcome,
    /// Time spent uploading codestream bytes into the resident CUDA buffer.
    pub codestream_upload_duration: Duration,
}

/// Submitted single-tile CUDA lossless encode.
#[derive(Debug)]
pub struct SubmittedJ2kLosslessCudaEncode {
    pub(super) inner: ReadySubmission<j2k::EncodedJ2k, crate::Error>,
}

/// Submitted multi-tile CUDA lossless encode.
#[derive(Debug)]
pub struct SubmittedJ2kLosslessCudaEncodeBatch {
    pub(super) inner: ReadySubmission<Vec<j2k::EncodedJ2k>, crate::Error>,
}

#[doc(hidden)]
impl DeviceSubmission for SubmittedJ2kLosslessCudaEncode {
    type Output = j2k::EncodedJ2k;
    type Error = crate::Error;

    fn wait(self) -> Result<Self::Output, Self::Error> {
        self.inner.wait()
    }
}

#[doc(hidden)]
impl DeviceSubmission for SubmittedJ2kLosslessCudaEncodeBatch {
    type Output = Vec<j2k::EncodedJ2k>;
    type Error = crate::Error;

    fn wait(self) -> Result<Self::Output, Self::Error> {
        self.inner.wait()
    }
}
