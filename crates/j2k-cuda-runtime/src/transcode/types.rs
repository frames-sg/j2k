// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::memory::{CudaBufferPool, CudaDeviceBuffer, CudaPooledDeviceBuffer};

/// Reversible 5/3 transcode bands downloaded from the device. Layout matches
/// `j2k_transcode::accelerator::ReversibleDwt53FirstLevel`.
#[derive(Clone, Debug, PartialEq, Eq)]
#[doc(hidden)]
pub struct CudaTranscodeReversible53Bands {
    /// Low-horizontal, low-vertical band (`low_width * low_height`).
    pub ll: Vec<i32>,
    /// High-horizontal, low-vertical band (`high_width * low_height`).
    pub hl: Vec<i32>,
    /// Low-horizontal, high-vertical band (`low_width * high_height`).
    pub lh: Vec<i32>,
    /// High-horizontal, high-vertical band (`high_width * high_height`).
    pub hh: Vec<i32>,
    /// Width of horizontally low-pass bands.
    pub low_width: usize,
    /// Height of vertically low-pass bands.
    pub low_height: usize,
    /// Width of horizontally high-pass bands.
    pub high_width: usize,
    /// Height of vertically high-pass bands.
    pub high_height: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct Reversible53Dims {
    pub(crate) block_cols: i32,
    pub(crate) width: i32,
    pub(crate) height: i32,
    pub(crate) low_width: i32,
    pub(crate) high_width: i32,
}

#[derive(Clone, Copy)]
pub(crate) struct DctBlockGrid {
    pub(crate) block_count: usize,
    pub(crate) expected_coeffs: usize,
    pub(crate) low_width: usize,
    pub(crate) low_height: usize,
    pub(crate) high_width: usize,
    pub(crate) high_height: usize,
    pub(crate) dims: Reversible53Dims,
}

/// Irreversible single-level 9/7 transcode bands downloaded from the device.
/// Device math is f32; callers widen to f64 (parity is within tolerance).
#[derive(Clone, Debug, PartialEq)]
#[doc(hidden)]
pub struct CudaTranscodeDwt97Bands {
    /// Low-horizontal, low-vertical band (`low_width * low_height`).
    pub ll: Vec<f32>,
    /// High-horizontal, low-vertical band (`high_width * low_height`).
    pub hl: Vec<f32>,
    /// Low-horizontal, high-vertical band (`low_width * high_height`).
    pub lh: Vec<f32>,
    /// High-horizontal, high-vertical band (`high_width * high_height`).
    pub hh: Vec<f32>,
    /// Width of horizontally low-pass bands.
    pub low_width: usize,
    /// Height of vertically low-pass bands.
    pub low_height: usize,
    /// Width of horizontally high-pass bands.
    pub high_width: usize,
    /// Height of vertically high-pass bands.
    pub high_height: usize,
}

/// Per-subband inverse step sizes and code-block geometry for the fused 9/7
/// code-block quantization batch. The dispatch layer derives the deltas from
/// the `j2k-transcode` code-block oracle so the numbers stay authoritative.
#[derive(Clone, Copy, Debug)]
#[doc(hidden)]
pub struct CudaHtj2k97QuantizeParams {
    /// `1/Δ` for the LL subband.
    pub inv_delta_ll: f32,
    /// `1/Δ` for the HL subband.
    pub inv_delta_hl: f32,
    /// `1/Δ` for the LH subband.
    pub inv_delta_lh: f32,
    /// `1/Δ` for the HH subband.
    pub inv_delta_hh: f32,
    /// Code-block width in coefficients (`1 << (code_block_width_exp + 2)`).
    pub cb_width: usize,
    /// Code-block height in coefficients (`1 << (code_block_height_exp + 2)`).
    pub cb_height: usize,
}

/// Shared same-geometry shape for CUDA 9/7 DCT-grid batches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[doc(hidden)]
pub struct CudaDwt97BatchGeometry {
    /// Number of items in the batch.
    pub item_count: usize,
    /// Number of 8x8 DCT blocks per item row.
    pub block_cols: usize,
    /// Number of 8x8 DCT blocks per item column.
    pub block_rows: usize,
    /// Logical item width in samples.
    pub width: usize,
    /// Logical item height in samples.
    pub height: usize,
}

/// Host f32 input plus caller-owned transient buffer pool for a CUDA 9/7 DWT batch.
#[derive(Clone, Copy)]
#[doc(hidden)]
pub struct CudaDwt97BatchWithPoolRequest<'a> {
    /// Natural-order 8x8 DCT coefficients for every item in the batch.
    pub blocks: &'a [f32],
    /// Shared geometry for every item in `blocks`.
    pub geometry: CudaDwt97BatchGeometry,
    /// Pool used for transient device buffers.
    pub pool: &'a CudaBufferPool,
}

/// Host f32 input plus caller-owned transient buffer pool for a CUDA 9/7
/// code-block quantization batch.
#[derive(Clone, Copy)]
#[doc(hidden)]
pub struct CudaHtj2k97CodeblockBatchWithPoolRequest<'a> {
    /// Natural-order 8x8 DCT coefficients for every item in the batch.
    pub blocks: &'a [f32],
    /// Shared geometry for every item in `blocks`.
    pub geometry: CudaDwt97BatchGeometry,
    /// Per-subband quantization and code-block geometry.
    pub params: CudaHtj2k97QuantizeParams,
    /// Pool used for transient device buffers.
    pub pool: &'a CudaBufferPool,
}

/// Host i16 input plus caller-owned transient buffer pool for a CUDA 9/7
/// code-block quantization batch.
#[derive(Clone, Copy)]
#[doc(hidden)]
pub struct CudaHtj2k97I16CodeblockBatchWithPoolRequest<'a> {
    /// Natural-order 8x8 DCT coefficients for every item in the batch.
    pub blocks: &'a [i16],
    /// Shared geometry for every item in `blocks`.
    pub geometry: CudaDwt97BatchGeometry,
    /// Per-subband quantization and code-block geometry.
    pub params: CudaHtj2k97QuantizeParams,
    /// Pool used for transient device buffers.
    pub pool: &'a CudaBufferPool,
}

#[derive(Clone, Copy)]
pub(crate) struct Dwt97CodeblockBandBuffers<'a> {
    pub(crate) ll: &'a CudaDeviceBuffer,
    pub(crate) hl: &'a CudaDeviceBuffer,
    pub(crate) lh: &'a CudaDeviceBuffer,
    pub(crate) hh: &'a CudaDeviceBuffer,
}

/// Per-item raw code-block-major quantized 9/7 bands from the fused batch.
///
/// Each band concatenates `item_count` per-item subband buffers in code-block
/// -major order (outer code-block row, inner code-block column, each block
/// row-major), matching the `j2k-transcode` code-block oracle layout. The
/// dispatch layer reslices these into prequantized HTJ2K components.
#[derive(Clone, Debug, PartialEq, Eq)]
#[doc(hidden)]
pub struct CudaHtj2k97CodeblockBands {
    /// LL subband (`item_count * low_width * low_height`).
    pub ll: Vec<i32>,
    /// HL subband (`item_count * high_width * low_height`).
    pub hl: Vec<i32>,
    /// LH subband (`item_count * low_width * high_height`).
    pub lh: Vec<i32>,
    /// HH subband (`item_count * high_width * high_height`).
    pub hh: Vec<i32>,
    /// Number of items in the batch.
    pub item_count: usize,
    /// Width of horizontally low-pass bands.
    pub low_width: usize,
    /// Height of vertically low-pass bands.
    pub low_height: usize,
    /// Width of horizontally high-pass bands.
    pub high_width: usize,
    /// Height of vertically high-pass bands.
    pub high_height: usize,
}

/// Device-resident per-item raw code-block-major quantized 9/7 bands from the
/// fused transcode batch.
#[derive(Debug)]
#[doc(hidden)]
pub struct CudaHtj2k97DeviceCodeblockBands {
    /// LL subband (`item_count * low_width * low_height`).
    pub ll: CudaPooledDeviceBuffer,
    /// HL subband (`item_count * high_width * low_height`).
    pub hl: CudaPooledDeviceBuffer,
    /// LH subband (`item_count * low_width * high_height`).
    pub lh: CudaPooledDeviceBuffer,
    /// HH subband (`item_count * high_width * high_height`).
    pub hh: CudaPooledDeviceBuffer,
    /// Number of items in the batch.
    pub item_count: usize,
    /// Width of horizontally low-pass bands.
    pub low_width: usize,
    /// Height of vertically low-pass bands.
    pub low_height: usize,
    /// Width of horizontally high-pass bands.
    pub high_width: usize,
    /// Height of vertically high-pass bands.
    pub high_height: usize,
}

/// Device-resident 9/7 batch bands produced by the shared staged pipeline.
pub(crate) struct Dwt97BatchDeviceBands {
    pub(crate) ll: CudaPooledDeviceBuffer,
    pub(crate) lh: CudaPooledDeviceBuffer,
    pub(crate) hl: CudaPooledDeviceBuffer,
    pub(crate) hh: CudaPooledDeviceBuffer,
    pub(crate) low_width: usize,
    pub(crate) low_height: usize,
    pub(crate) high_width: usize,
    pub(crate) high_height: usize,
}

#[derive(Clone, Copy)]
pub(crate) enum Dwt97BatchInput<'a> {
    F32(&'a [f32]),
    I16(&'a [i16]),
}

#[derive(Clone, Copy)]
pub(super) struct Dwt97BatchDeviceRequest<'a> {
    pub(super) input: Dwt97BatchInput<'a>,
    pub(super) geometry: CudaDwt97BatchGeometry,
    pub(super) pool: &'a CudaBufferPool,
}

#[derive(Clone, Copy)]
pub(super) struct Htj2k97I16ResidentFusedRequest<'a> {
    pub(super) blocks: &'a [i16],
    pub(super) geometry: CudaDwt97BatchGeometry,
    pub(super) params: CudaHtj2k97QuantizeParams,
    pub(super) pool: &'a CudaBufferPool,
}

pub(super) struct Dwt97ColumnLiftBatchLaunch<'a> {
    pub(super) rows_buffer: &'a CudaDeviceBuffer,
    pub(super) band_width: i32,
    pub(super) height: i32,
    pub(super) low_height: i32,
    pub(super) high_height: i32,
    pub(super) items: u32,
    pub(super) low_out: &'a CudaDeviceBuffer,
    pub(super) high_out: &'a CudaDeviceBuffer,
}

pub(super) struct Dwt97ColumnLiftQuantizeCodeblocksBatchLaunch<'a> {
    pub(super) column: Dwt97ColumnLiftBatchLaunch<'a>,
    pub(super) cb_width: i32,
    pub(super) cb_height: i32,
    pub(super) inv_delta_low: f32,
    pub(super) inv_delta_high: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Dwt97ColumnLiftQuantizeCodeblocksParams {
    pub(super) cb_width: i32,
    pub(super) cb_height: i32,
    pub(super) inv_delta_low: f32,
    pub(super) inv_delta_high: f32,
}

// SAFETY: `Dwt97ColumnLiftQuantizeCodeblocksParams` is `#[repr(C)]` and
// contains only CUDA scalar ABI fields passed by value through a
// kernel-parameter pointer.
unsafe impl crate::execution::CudaKernelParam for Dwt97ColumnLiftQuantizeCodeblocksParams {}

pub(super) struct Dwt97QuantizeCodeblocksLaunch<'a> {
    pub(super) band: &'a CudaDeviceBuffer,
    pub(super) output: &'a CudaDeviceBuffer,
    pub(super) width: i32,
    pub(super) height: i32,
    pub(super) cb_width: i32,
    pub(super) cb_height: i32,
    pub(super) inv_delta: f32,
    pub(super) items: u32,
}
