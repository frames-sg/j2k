// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::{
    error::CudaError,
    execution::{CudaExecutionStats, CudaLaunchMode},
    kernels::CudaKernel,
    memory::{pooled_device_buffer, CudaDeviceBuffer, CudaPooledDeviceBuffer},
};

use super::output_regions::ValidatedHtj2kOutputLayout;

#[doc(hidden)]
/// HTJ2K code-block decode job consumed by the CUDA entropy kernel launcher.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaHtj2kCodeBlockJob {
    /// Byte offset into the contiguous compressed payload buffer.
    pub payload_offset: u64,
    /// Code-block width in coefficients.
    pub width: u32,
    /// Code-block height in coefficients.
    pub height: u32,
    /// Combined cleanup/refinement byte length.
    pub payload_len: u32,
    /// Cleanup segment length in bytes.
    pub cleanup_length: u32,
    /// Refinement segment length in bytes.
    pub refinement_length: u32,
    /// Missing most-significant bit planes.
    pub missing_bit_planes: u8,
    /// Total coded bitplanes for this code block's sub-band.
    pub num_bitplanes: u8,
    /// Number of HT coding passes present.
    pub number_of_coding_passes: u8,
    /// Output row stride, in coefficients.
    pub output_stride: u32,
    /// Output offset, in coefficients, into the destination plane.
    pub output_offset: u32,
    /// Dequantization multiplier for decoded coefficient values.
    pub dequantization_step: f32,
    /// Vertically causal context mode flag.
    pub stripe_causal: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CudaHtj2kCodeBlockKernelJob {
    pub(crate) coded_offset: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) coded_len: u32,
    pub(crate) cleanup_length: u32,
    pub(crate) refinement_length: u32,
    pub(crate) missing_msbs: u32,
    pub(crate) num_bitplanes: u32,
    pub(crate) number_of_coding_passes: u32,
    pub(crate) output_stride: u32,
    pub(crate) output_offset: u32,
    pub(crate) dequantization_step: f32,
    pub(crate) stripe_causal: u32,
}

/// One output buffer and its code-block jobs for batched HTJ2K cleanup decode.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct CudaHtj2kCleanupTarget<'a> {
    /// Device buffer receiving decoded integer coefficient bits.
    pub coefficients: &'a CudaDeviceBuffer,
    /// Code-block jobs that write into `coefficients`.
    pub jobs: &'a [CudaHtj2kCodeBlockJob],
    /// Number of coefficient words available in `coefficients`.
    pub output_words: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CudaHtj2kCleanupMultiKernelJob {
    pub(crate) output_ptr: u64,
    pub(crate) coded_offset: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) coded_len: u32,
    pub(crate) cleanup_length: u32,
    pub(crate) refinement_length: u32,
    pub(crate) missing_msbs: u32,
    pub(crate) num_bitplanes: u32,
    pub(crate) number_of_coding_passes: u32,
    pub(crate) output_stride: u32,
    pub(crate) output_offset: u32,
    pub(crate) dequantization_step: f32,
    pub(crate) stripe_causal: u32,
    pub(crate) reserved_tail: u32,
}

/// One output buffer and its code-block jobs for batched HTJ2K dequantization.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct CudaHtj2kDequantizeTarget<'a> {
    /// Device buffer containing decoded integer coefficient bits.
    pub coefficients: &'a CudaDeviceBuffer,
    /// Code-block jobs that write into `coefficients`.
    pub jobs: &'a [CudaHtj2kCodeBlockJob],
    /// Number of coefficient words available in `coefficients`.
    pub output_words: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CudaHtj2kDequantizeKernelJob {
    pub(crate) output_ptr: u64,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) output_stride: u32,
    pub(crate) output_offset: u32,
    pub(crate) num_bitplanes: u32,
    pub(crate) reserved: u32,
    pub(crate) dequantization_step: f32,
    pub(crate) reserved_tail: u32,
}

#[doc(hidden)]
/// Static HTJ2K entropy lookup tables uploaded for CUDA code-block decode.
#[derive(Clone, Copy, Debug)]
pub struct CudaHtj2kDecodeTables<'a> {
    /// HT cleanup VLC table for first quad row contexts.
    pub vlc_table0: &'a [u16; 1024],
    /// HT cleanup VLC table for subsequent quad row contexts.
    pub vlc_table1: &'a [u16; 1024],
    /// HT cleanup UVLC table for first quad row contexts.
    pub uvlc_table0: &'a [u16; 320],
    /// HT cleanup UVLC table for subsequent quad row contexts.
    pub uvlc_table1: &'a [u16; 256],
}

/// Status written by the CUDA HTJ2K entropy decoder for one code-block job.
#[doc(hidden)]
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaHtj2kStatus {
    /// Zero on success; nonzero values are kernel-defined failures.
    pub code: u32,
    /// Kernel-defined failure detail.
    pub detail: u32,
    /// Reserved for ABI stability.
    pub reserved0: u32,
    /// Reserved for ABI stability.
    pub reserved1: u32,
}

impl CudaHtj2kStatus {
    /// Return true when the CUDA kernel reported success.
    pub fn is_ok(self) -> bool {
        self.code == HTJ2K_STATUS_OK
    }
}

/// CUDA event timings for resident HTJ2K decode stages.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaHtj2kDecodeStageTimings {
    /// HT cleanup entropy decode dispatch time, in microseconds.
    pub ht_cleanup_us: u128,
    /// HT refinement work time, in microseconds.
    ///
    /// The current CUDA entropy kernel executes cleanup and refinement for a
    /// code-block in one dispatch. When a batch contains refinement segments,
    /// this records that fused dispatch time so higher-level profiles expose
    /// refinement-bearing work instead of silently reporting zero.
    pub ht_refine_us: u128,
    /// Sign/magnitude dequantization time, in microseconds.
    pub dequant_us: u128,
    /// Host-observed status download time, in microseconds.
    pub status_d2h_us: u128,
}

/// Device-resident HTJ2K entropy decode result.
#[doc(hidden)]
#[derive(Debug)]
pub struct CudaHtj2kDecodeOutput {
    pub(crate) coefficients: CudaDeviceBuffer,
    pub(crate) execution: CudaExecutionStats,
    pub(crate) statuses: Vec<CudaHtj2kStatus>,
    pub(crate) stage_timings: CudaHtj2kDecodeStageTimings,
}

impl CudaHtj2kDecodeOutput {
    /// Device buffer containing decoded f32 coefficients.
    pub fn coefficients(&self) -> &CudaDeviceBuffer {
        &self.coefficients
    }

    /// CUDA execution counters for the decode dispatch.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }

    /// Per-code-block kernel status rows downloaded after dispatch.
    pub fn statuses(&self) -> &[CudaHtj2kStatus] {
        &self.statuses
    }

    /// CUDA event timings for the decode stages inside this output.
    pub fn stage_timings(&self) -> CudaHtj2kDecodeStageTimings {
        self.stage_timings
    }

    /// Split output into device coefficients, execution counters, and statuses.
    pub fn into_parts(self) -> (CudaDeviceBuffer, CudaExecutionStats, Vec<CudaHtj2kStatus>) {
        (self.coefficients, self.execution, self.statuses)
    }
}

/// Device-resident HTJ2K entropy decode result borrowed from a CUDA buffer pool.
#[doc(hidden)]
#[derive(Debug)]
pub struct CudaPooledHtj2kDecodeOutput {
    pub(crate) coefficients: CudaPooledDeviceBuffer,
    pub(crate) execution: CudaExecutionStats,
    pub(crate) statuses: Vec<CudaHtj2kStatus>,
    pub(crate) stage_timings: CudaHtj2kDecodeStageTimings,
}

impl CudaPooledHtj2kDecodeOutput {
    /// Device buffer containing decoded f32 coefficients.
    pub fn coefficients(&self) -> Option<&CudaDeviceBuffer> {
        self.coefficients.as_device_buffer()
    }

    /// CUDA execution counters for the decode dispatch.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }

    /// Per-code-block kernel status rows downloaded after dispatch.
    pub fn statuses(&self) -> &[CudaHtj2kStatus] {
        &self.statuses
    }

    /// CUDA event timings for the decode stages inside this output.
    pub fn stage_timings(&self) -> CudaHtj2kDecodeStageTimings {
        self.stage_timings
    }

    /// Split output into pooled device coefficients, execution counters, and statuses.
    pub fn into_parts(
        self,
    ) -> (
        CudaPooledDeviceBuffer,
        CudaExecutionStats,
        Vec<CudaHtj2kStatus>,
    ) {
        (self.coefficients, self.execution, self.statuses)
    }
}

/// Device-resident static HTJ2K cleanup decode lookup tables.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct CudaHtj2kDecodeTableResources {
    pub(crate) inner: Arc<CudaHtj2kDecodeTableResourceInner>,
}

#[derive(Debug)]
pub(crate) struct CudaHtj2kDecodeTableResourceInner {
    pub(crate) vlc_table0: CudaDeviceBuffer,
    pub(crate) vlc_table1: CudaDeviceBuffer,
    pub(crate) uvlc_table0: CudaDeviceBuffer,
    pub(crate) uvlc_table1: CudaDeviceBuffer,
}

/// Device-resident J2K decode payload with optional HTJ2K lookup tables.
#[doc(hidden)]
#[derive(Debug)]
pub struct CudaHtj2kDecodeResources {
    pub(crate) payload: CudaHtj2kDecodePayload,
    pub(crate) payload_len: usize,
    pub(crate) tables: Option<CudaHtj2kDecodeTableResources>,
}

#[derive(Debug)]
pub(crate) enum CudaHtj2kDecodePayload {
    Owned(CudaDeviceBuffer),
    Pooled(CudaPooledDeviceBuffer),
}

#[derive(Clone, Copy)]
pub(super) struct Htj2kDecodeKernelTables<'a> {
    pub(super) vlc_table0: &'a CudaDeviceBuffer,
    pub(super) vlc_table1: &'a CudaDeviceBuffer,
    pub(super) uvlc_table0: &'a CudaDeviceBuffer,
    pub(super) uvlc_table1: &'a CudaDeviceBuffer,
}

#[derive(Clone, Copy)]
pub(super) struct Htj2kDecodeCodeblocksLaunch<'a> {
    pub(super) payload: &'a CudaDeviceBuffer,
    pub(super) coefficients: &'a CudaDeviceBuffer,
    pub(super) jobs: &'a CudaDeviceBuffer,
    pub(super) tables: Htj2kDecodeKernelTables<'a>,
    pub(super) statuses: &'a CudaDeviceBuffer,
    pub(super) job_count: usize,
    pub(super) mode: CudaLaunchMode,
}

#[derive(Clone, Copy)]
pub(super) struct Htj2kDecodeCodeblocksMultiLaunch<'a> {
    pub(super) kernel: CudaKernel,
    pub(super) payload: &'a CudaDeviceBuffer,
    pub(super) jobs: &'a CudaDeviceBuffer,
    pub(super) tables: Htj2kDecodeKernelTables<'a>,
    pub(super) statuses: &'a CudaDeviceBuffer,
    pub(super) job_count: usize,
    pub(super) mode: CudaLaunchMode,
}

pub(super) struct ValidatedHtj2kKernelJobs {
    pub(super) jobs: Vec<CudaHtj2kCodeBlockKernelJob>,
    pub(super) output_layout: ValidatedHtj2kOutputLayout,
}

impl CudaHtj2kDecodePayload {
    pub(crate) fn buffer(&self) -> Result<&CudaDeviceBuffer, CudaError> {
        match self {
            Self::Owned(buffer) => Ok(buffer),
            Self::Pooled(buffer) => pooled_device_buffer(buffer),
        }
    }
}

pub(super) fn htj2k_decode_kernel_tables(
    resources: &CudaHtj2kDecodeResources,
) -> Result<Htj2kDecodeKernelTables<'_>, CudaError> {
    let tables = resources
        .tables
        .as_ref()
        .ok_or_else(|| CudaError::InvalidArgument {
            message: "HTJ2K decode requires resident lookup tables".to_string(),
        })?;
    Ok(Htj2kDecodeKernelTables {
        vlc_table0: &tables.inner.vlc_table0,
        vlc_table1: &tables.inner.vlc_table1,
        uvlc_table0: &tables.inner.uvlc_table0,
        uvlc_table1: &tables.inner.uvlc_table1,
    })
}

pub(crate) const HTJ2K_STATUS_OK: u32 = 0;

pub(crate) const HTJ2K_STATUS_UNSUPPORTED: u32 = 2;
