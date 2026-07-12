// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    error::CudaError, execution::CudaExecutionStats, htj2k_decode::HTJ2K_STATUS_OK,
    memory::CudaDeviceBuffer,
};
use j2k_core::host_capacity_bytes;

#[doc(hidden)]
/// Static HTJ2K cleanup encoder lookup tables uploaded for CUDA code-block encode.
#[derive(Clone, Copy, Debug)]
pub struct CudaHtj2kEncodeTables<'a> {
    /// HT cleanup encoder VLC table for first quad row contexts.
    pub vlc_table0: &'a [u16; 2048],
    /// HT cleanup encoder VLC table for subsequent quad row contexts.
    pub vlc_table1: &'a [u16; 2048],
    /// Packed HT cleanup encoder UVLC table rows, six bytes per row.
    pub uvlc_table: &'a [u8],
}

/// Device-resident HTJ2K cleanup encode lookup tables reused across sub-band dispatches.
#[doc(hidden)]
#[derive(Debug)]
pub struct CudaHtj2kEncodeResources {
    pub(crate) vlc_table0: CudaDeviceBuffer,
    pub(crate) vlc_table1: CudaDeviceBuffer,
    pub(crate) uvlc_table: CudaDeviceBuffer,
}

impl CudaHtj2kEncodeResources {
    pub(super) fn launch_tables(&self) -> CudaHtj2kEncodeLaunchTables<'_> {
        CudaHtj2kEncodeLaunchTables {
            vlc_table0: &self.vlc_table0,
            vlc_table1: &self.vlc_table1,
            uvlc_table: &self.uvlc_table,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct CudaHtj2kEncodeLaunchTables<'a> {
    pub(super) vlc_table0: &'a CudaDeviceBuffer,
    pub(super) vlc_table1: &'a CudaDeviceBuffer,
    pub(super) uvlc_table: &'a CudaDeviceBuffer,
}

pub(super) struct CudaHtj2kEncodeCodeblocksLaunch<'a> {
    pub(super) coefficients: &'a CudaDeviceBuffer,
    pub(super) output: &'a CudaDeviceBuffer,
    pub(super) jobs: &'a CudaDeviceBuffer,
    pub(super) tables: CudaHtj2kEncodeLaunchTables<'a>,
    pub(super) statuses: &'a CudaDeviceBuffer,
    pub(super) job_count: usize,
}

pub(super) struct CudaHtj2kEncodeMultiInputLaunch<'a> {
    pub(super) output: &'a CudaDeviceBuffer,
    pub(super) jobs: &'a CudaDeviceBuffer,
    pub(super) tables: CudaHtj2kEncodeLaunchTables<'a>,
    pub(super) statuses: &'a CudaDeviceBuffer,
    pub(super) job_count: usize,
}

pub(crate) const HTJ2K_ENCODE_MAX_CODEBLOCK_WIDTH: u32 = 1024;

pub(crate) const HTJ2K_ENCODE_MAX_CODEBLOCK_SAMPLES: usize = 4096;

#[doc(hidden)]
/// One HTJ2K code-block encode job consumed by the CUDA batch encoder.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaHtj2kEncodeCodeBlockJob {
    /// Offset, in i32 coefficients, into the contiguous coefficient buffer.
    pub coefficient_offset: u32,
    /// Code-block width in coefficients.
    pub width: u32,
    /// Code-block height in coefficients.
    pub height: u32,
    /// Total coded bitplanes for this code block's sub-band.
    pub total_bitplanes: u8,
    /// Requested HT coding passes. `1` emits cleanup-only output; `2` emits a
    /// zero `SigProp` segment for exactly representable blocks; `3` emits
    /// `SigProp` bits for newly significant magnitude-3 samples plus `MagRef`
    /// bits for cleanup-significant samples.
    pub target_coding_passes: u8,
}

#[doc(hidden)]
/// One HTJ2K code-block region consumed from a strided resident coefficient buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaHtj2kEncodeCodeBlockRegionJob {
    /// Offset, in i32 coefficients, to the top-left coefficient of this code block.
    pub coefficient_offset: u32,
    /// Source row stride in i32 coefficients.
    pub coefficient_stride: u32,
    /// Code-block width in coefficients.
    pub width: u32,
    /// Code-block height in coefficients.
    pub height: u32,
    /// Total coded bitplanes for this code block's sub-band.
    pub total_bitplanes: u8,
    /// Requested HT coding passes. `1` emits cleanup-only output; `2` emits a
    /// zero `SigProp` segment for exactly representable blocks; `3` emits
    /// `SigProp` bits for newly significant magnitude-3 samples plus `MagRef`
    /// bits for cleanup-significant samples.
    pub target_coding_passes: u8,
}

/// Resident coefficient buffer and jobs for a multi-input HTJ2K encode batch.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct CudaHtj2kEncodeResidentTarget<'a> {
    /// Device buffer containing quantized i32 coefficients.
    pub coefficients: &'a CudaDeviceBuffer,
    /// Number of i32 coefficients available in `coefficients`.
    pub coefficient_count: usize,
    /// Code-block jobs that read from `coefficients`.
    pub jobs: &'a [CudaHtj2kEncodeCodeBlockJob],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CudaHtj2kEncodeKernelJob {
    pub(crate) coefficient_offset: u32,
    pub(crate) coefficient_stride: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) total_bitplanes: u32,
    pub(crate) output_offset: u32,
    pub(crate) output_capacity: u32,
    pub(crate) target_coding_passes: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CudaHtj2kEncodeMultiInputKernelJob {
    pub(crate) coefficient_ptr: u64,
    pub(crate) coefficient_offset: u32,
    pub(crate) coefficient_stride: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) total_bitplanes: u32,
    pub(crate) output_offset: u32,
    pub(crate) output_capacity: u32,
    pub(crate) target_coding_passes: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CudaHtj2kEncodeCompactJob {
    pub(crate) source_offset: u32,
    pub(crate) compact_offset: u32,
    pub(crate) data_len: u32,
    pub(crate) reserved: u32,
}

/// Status written by the CUDA HTJ2K code-block cleanup-pass encoder.
#[doc(hidden)]
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaHtj2kEncodeStatus {
    /// Zero on success; nonzero values are kernel-defined failures.
    pub code: u32,
    /// Kernel-defined failure detail.
    pub detail: u32,
    /// Encoded payload byte length.
    pub data_len: u32,
    /// Number of coding passes in the encoded payload.
    pub number_of_coding_passes: u32,
    /// Number of missing most-significant bitplanes.
    pub missing_bit_planes: u32,
    /// Reserved for ABI stability.
    pub reserved0: u32,
    /// Reserved for ABI stability.
    pub reserved1: u32,
    /// Reserved for ABI stability.
    pub reserved2: u32,
}

impl CudaHtj2kEncodeStatus {
    /// Return true when the CUDA kernel reported success.
    pub fn is_ok(self) -> bool {
        self.code == HTJ2K_STATUS_OK
    }
}

/// CUDA event timings for HTJ2K cleanup-pass encode stages.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaHtj2kEncodeStageTimings {
    /// Total HT cleanup-pass encode, compaction, and required result readback time, in microseconds.
    pub ht_encode_us: u128,
    /// HT cleanup-pass encode kernel time, in microseconds.
    pub ht_kernel_us: u128,
    /// Status-buffer device-to-host readback time, in microseconds.
    pub ht_status_readback_us: u128,
    /// Encoded-byte compaction kernel time, in microseconds.
    pub ht_compact_us: u128,
    /// Compacted encoded-byte device-to-host readback time, in microseconds.
    pub ht_output_readback_us: u128,
}

impl CudaHtj2kEncodeStageTimings {
    pub(super) fn from_parts(
        ht_kernel_us: u128,
        ht_status_readback_us: u128,
        ht_compact_us: u128,
        ht_output_readback_us: u128,
    ) -> Self {
        Self {
            ht_encode_us: ht_kernel_us
                .saturating_add(ht_status_readback_us)
                .saturating_add(ht_compact_us)
                .saturating_add(ht_output_readback_us),
            ht_kernel_us,
            ht_status_readback_us,
            ht_compact_us,
            ht_output_readback_us,
        }
    }
}

/// Host-visible HTJ2K cleanup-pass encode result produced by a CUDA kernel.
#[doc(hidden)]
#[derive(Debug)]
pub struct CudaHtj2kEncodedCodeBlock {
    pub(crate) data: Vec<u8>,
    pub(crate) status: CudaHtj2kEncodeStatus,
    pub(crate) execution: CudaExecutionStats,
    pub(crate) stage_timings: CudaHtj2kEncodeStageTimings,
}

impl CudaHtj2kEncodedCodeBlock {
    /// Encoded cleanup-pass payload bytes.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    impl_cuda_htj2k_encoded_status_accessors!();

    /// Consume this code block and return its encoded payload plus segment
    /// metadata.
    pub fn into_parts(self) -> (Vec<u8>, u32, u32, u8, u8) {
        (
            self.data,
            htj2k_encoded_cleanup_length(self.status),
            htj2k_encoded_refinement_length(self.status),
            htj2k_encoded_num_coding_passes(self.status),
            htj2k_encoded_num_zero_bitplanes(self.status),
        )
    }
}

pub(crate) fn htj2k_encoded_cleanup_length(status: CudaHtj2kEncodeStatus) -> u32 {
    if status.number_of_coding_passes <= 1 {
        status.data_len
    } else {
        status.reserved0
    }
}

pub(crate) fn htj2k_encoded_refinement_length(status: CudaHtj2kEncodeStatus) -> u32 {
    if status.number_of_coding_passes <= 1 {
        0
    } else {
        status.reserved1
    }
}

pub(crate) fn htj2k_encoded_num_coding_passes(status: CudaHtj2kEncodeStatus) -> u8 {
    u8::try_from(status.number_of_coding_passes).unwrap_or(u8::MAX)
}

pub(crate) fn htj2k_encoded_num_zero_bitplanes(status: CudaHtj2kEncodeStatus) -> u8 {
    u8::try_from(status.missing_bit_planes).unwrap_or(u8::MAX)
}

pub(super) fn empty_htj2k_encoded_code_blocks() -> CudaHtj2kEncodedCodeBlocks {
    CudaHtj2kEncodedCodeBlocks {
        code_blocks: Vec::new(),
        execution: CudaExecutionStats::default(),
        stage_timings: CudaHtj2kEncodeStageTimings::default(),
    }
}

pub(super) fn validate_resident_coefficient_capacity(
    coefficients: &CudaDeviceBuffer,
    coefficient_count: usize,
) -> Result<(), CudaError> {
    let available_coefficients = coefficients.typed_view::<i32>()?.len();
    if available_coefficients < coefficient_count {
        return Err(CudaError::OutputTooSmall {
            required: coefficient_count
                .checked_mul(std::mem::size_of::<i32>())
                .ok_or(CudaError::LengthTooLarge {
                    len: coefficient_count,
                })?,
            have: coefficients.byte_len(),
        });
    }

    Ok(())
}

/// Host-visible HTJ2K cleanup-pass encode batch produced by one CUDA kernel dispatch.
#[doc(hidden)]
#[derive(Debug)]
pub struct CudaHtj2kEncodedCodeBlocks {
    pub(crate) code_blocks: Vec<CudaHtj2kEncodedCodeBlock>,
    pub(crate) execution: CudaExecutionStats,
    pub(crate) stage_timings: CudaHtj2kEncodeStageTimings,
}

impl CudaHtj2kEncodedCodeBlocks {
    /// Encoded cleanup code-block payloads, in the same order as the submitted jobs.
    pub fn code_blocks(&self) -> &[CudaHtj2kEncodedCodeBlock] {
        &self.code_blocks
    }

    /// Consume the batch and return its per-code-block outputs.
    pub fn into_code_blocks(self) -> Vec<CudaHtj2kEncodedCodeBlock> {
        self.code_blocks
    }

    /// Allocator-reported bytes retained by the outer result and its payloads.
    #[doc(hidden)]
    pub fn host_capacity_bytes(&self) -> usize {
        self.code_blocks.iter().fold(
            host_capacity_bytes::<CudaHtj2kEncodedCodeBlock>(self.code_blocks.capacity()),
            |bytes, block| bytes.saturating_add(host_capacity_bytes::<u8>(block.data.capacity())),
        )
    }

    /// CUDA execution counters for the batch encode dispatch.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }

    /// CUDA event timings for the batch encode dispatch.
    pub fn stage_timings(&self) -> CudaHtj2kEncodeStageTimings {
        self.stage_timings
    }
}
