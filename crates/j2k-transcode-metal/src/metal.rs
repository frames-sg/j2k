// SPDX-License-Identifier: MIT OR Apache-2.0

//! Metal runtime for direct DCT-grid to one-level wavelet projection.

use std::sync::Arc;
use std::time::Instant;

use core::mem::size_of;

use j2k_core::{BackendKind, DeviceMemoryRange};
use j2k_metal_support::{
    checked_buffer_read_vec, checked_buffer_write, checked_command_buffer, checked_command_queue,
    checked_compute_command_encoder, commit_and_wait, system_default_device, MetalPipelineLoader,
    MetalSupportError,
};
use j2k_transcode::{
    htj2k97_subband_delta, htj2k97_subband_total_bitplanes, idct_blocks_to_signed_samples_rayon,
    DctGridToDwt53Job, DctGridToDwt97Job, DctGridToHtj2k97CodeBlockJob,
    DctGridToReversibleDwt53Job, Dwt53TwoDimensional, Dwt97BatchStageTimings, Dwt97TwoDimensional,
    Htj2k97CodeBlockOptions, J2kSubBandType, PrequantizedHtj2k97CodeBlock,
    PrequantizedHtj2k97Component, PrequantizedHtj2k97Resolution, PrequantizedHtj2k97Subband,
    ResidentBufferRef, ResidentColorModel, ResidentComponentGeometry, ResidentDctCoefficientOrder,
    ResidentDctGridLayout, ResidentDwtSubband, ResidentDwtSubbandKind, ResidentDwtSubbandLayout,
    ResidentHandoffError, ResidentJpegDctGrid, ResidentSampleInfo, ResidentSampling,
    ReversibleDwt53FirstLevel, TranscodeStageError,
};
use metal::{
    foreign_types::ForeignType, Buffer, CommandBufferRef, CommandQueue, ComputeCommandEncoderRef,
    ComputePipelineState, Device, MTLSize,
};

use crate::weights::{SparseDwt53WeightRows, SparseDwt97WeightRows, SparseWeightRow};
use crate::MetalTranscodeError;

mod runtime;
pub use self::runtime::MetalTranscodeSession;
use self::runtime::{
    Dct97ColumnLiftParams, Dct97IdctRowLiftParams, Dct97QuantizeCodeblocksParams,
    DctBatchProjectionParams, DctProjectionParams, MetalRuntime, MetalSparseRow, MetalSparseRows,
    MetalWeightTap, Reversible53ProjectionParams,
};
mod allocation;
use self::allocation::{
    checked_device_element_count, checked_device_workspace_bytes, checked_host_element_count,
    checked_host_workspace_bytes, private_buffer_for_len, shared_buffer_for_len,
    shared_buffer_with_slice, try_transcode_vec_from_slice, try_transcode_vec_with_capacity,
};
mod reversible;
pub(crate) use self::reversible::{
    dispatch_dct_grid_to_dwt53, dispatch_dct_grid_to_reversible_dwt53,
    dispatch_dct_grid_to_reversible_dwt53_batch,
};
mod irreversible;
use self::irreversible::dispatch_dct_grid_to_dwt53_with_runtime;
pub(crate) use self::irreversible::{
    dispatch_dct_grid_to_dwt97, dispatch_dct_grid_to_dwt97_batch,
    dispatch_dct_grid_to_htj2k97_codeblock_batch,
};
mod projection;
#[cfg(test)]
use self::projection::projection_dispatch_sizes;
use self::projection::{
    bind_projection_band_buffers, bind_projection_input_buffers,
    dispatch_projected_bands_batch_with_runtime, dispatch_projected_bands_with_runtime,
    dispatch_projection_threads, staged_threads_per_group, ProjectedBands, ProjectionBatchJob,
    ProjectionJob,
};
mod resident;
use self::resident::{
    dispatch_projection_batch_bands, dwt97_codeblock_output_buffers,
    dwt97_codeblock_output_transfer_bytes, dwt97_codeblock_output_transfer_count,
    projection_batch_output_buffers, projection_batch_output_transfer_bytes,
    projection_batch_output_transfer_count, projection_batch_private_output_buffers,
    projection_batch_shape, projection_batch_weight_buffers, read_projected_batch_outputs,
    validate_resident_dct_handoffs_for_dwt97_jobs, validate_resident_dct_handoffs_for_htj2k_jobs,
    validate_resident_dwt_handoffs_for_dwt97_jobs, validate_resident_dwt_handoffs_for_htj2k_jobs,
    Dwt97CodeBlockOutputBuffers, ProjectionBatchOutputBuffers, ProjectionBatchShape,
};
mod codeblock_output;
use self::codeblock_output::{
    read_prequantized_97_codeblock_outputs, validate_codeblock_output_host_workspace,
};
mod geometry;
use self::geometry::{
    checked_batch_len, code_block_len_from_exp, dispatch_band, dispatch_band_batch,
    dispatch_reversible_band, dwt97_quantize_inv_delta, dwt97_total_bitplanes,
    reversible_band_geometry, u32_param, upload_sparse_rows,
    validate_codeblock_projection_allocations, validate_dwt97_batch_geometry,
    validate_dwt97_codeblock_batch_geometry, validate_float_projection_allocations, validate_grid,
    validate_htj2k97_codeblock_options, validate_reversible_batch_geometry, BandGeometry,
    BatchBandGeometry, ReversibleBatchKernelGeometry,
};
mod buffers;
use self::buffers::{
    buffer_with_slice, dwt97_batch_blocks_buffer, dwt97_block_value_count, dwt97_blocks_buffer,
    dwt97_codeblock_batch_blocks_buffer, idct8_basis_table, output_buffer, output_i32_buffer,
    private_f32_buffer, read_f32_buffer, read_f32_buffer_at, read_i32_buffer_at,
};

const METAL_DCT_KERNEL_FAILED: &str = "Metal DCT wavelet projection failed";
const METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID: &str =
    "Metal reversible DCT 5/3 job has unsupported grid geometry";
const METAL_DCT53_UNSUPPORTED_GRID: &str = "Metal DCT 5/3 job has unsupported grid geometry";
const METAL_DCT97_UNSUPPORTED_GRID: &str = "Metal DCT 9/7 job has unsupported grid geometry";
const DWT97_STAGED_MAX_AXIS: usize = 1024;
const DWT97_STAGED_ROWS_PER_GROUP: usize = 2;
const DWT97_STAGED_COLUMNS_PER_GROUP: usize = 4;
const DWT97_STAGED_THREADS_PER_GROUP: u64 = 256;
const DWT97_BLOCK_COEFFICIENTS: usize = 64;
const METAL_READBACK_CHUNK_BYTES: usize = 64 * 1024;

#[cfg(test)]
mod tests;
