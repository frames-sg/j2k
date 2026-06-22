use crate::{
    build_flags::{
        dwt97_fused_column_quantize_disabled, DWT97_ROW_LIFT_COOP_ROWS_PER_BLOCK,
        DWT97_ROW_LIFT_COOP_THREADS_X, DWT97_ROW_LIFT_MAX_WIDTH,
        PINNED_POOLED_I16_UPLOAD_MAX_BYTES, TRANSCODE_PTX_BUILT_FROM_CUDA,
    },
    bytes::i16_slice_as_bytes,
    context::CudaContext,
    driver::CuFunction,
    error::CudaError,
    execution::cuda_kernel_param,
    j2k_encode::CudaDwt97BatchStageTimings,
    kernels::{
        self, copy_u8_launch_geometry, j2k_dwt53_launch_geometry, with_grid_y, with_grid_z,
        CudaKernel,
    },
    memory::{pooled_device_buffer, CudaBufferPool, CudaDeviceBuffer, CudaPooledDeviceBuffer},
};
use std::os::raw::c_uint;

/// Reversible 5/3 transcode bands downloaded from the device. Layout matches
/// `j2k_transcode::accelerator::ReversibleDwt53FirstLevel`.
#[derive(Clone, Debug, PartialEq, Eq)]
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

impl CudaContext {
    /// Compute one reversible integer 5/3 level directly from dequantized 8x8
    /// DCT blocks, bit-exact with the `j2k-transcode` scalar oracle.
    ///
    /// `dequantized_blocks` holds `block_cols * block_rows` natural-order blocks
    /// of 64 `i16` coefficients. `width`/`height` are the logical component
    /// dimensions (<= `block_cols*8` / `block_rows*8`).
    #[allow(clippy::too_many_lines)]
    pub fn j2k_transcode_reversible_dwt53(
        &self,
        dequantized_blocks: &[i16],
        block_cols: usize,
        block_rows: usize,
        width: usize,
        height: usize,
    ) -> Result<CudaTranscodeReversible53Bands, CudaError> {
        if !TRANSCODE_PTX_BUILT_FROM_CUDA {
            return Err(CudaError::InvalidArgument {
                message: "CUDA transcode kernels were not built (nvcc unavailable at build time)"
                    .to_string(),
            });
        }
        let grid = validate_dct_block_grid(
            block_cols,
            block_rows,
            width,
            height,
            1,
            dequantized_blocks.len(),
            "reversible 5/3 transcode job has unsupported grid geometry",
        )?;
        let DctBlockGrid {
            block_count,
            expected_coeffs,
            low_width,
            low_height,
            high_width,
            high_height,
            dims,
        } = grid;

        self.inner.set_current()?;

        let alloc_i32 = |count: usize| -> Result<CudaDeviceBuffer, CudaError> {
            let bytes = count
                .checked_mul(std::mem::size_of::<i32>())
                .ok_or(CudaError::LengthTooLarge { len: count })?;
            self.allocate(bytes)
        };
        let samples = alloc_i32(expected_coeffs)?;
        let v_low = alloc_i32(width * low_height)?;
        let v_high = alloc_i32(width * high_height)?;
        let ll = alloc_i32(low_width * low_height)?;
        let hl = alloc_i32(high_width * low_height)?;
        let lh = alloc_i32(low_width * high_height)?;
        let hh = alloc_i32(high_width * high_height)?;

        // SAFETY: `dequantized_blocks` is a live `&[i16]`; reinterpreting it as a
        // byte slice of `len * 2` bytes for upload is a read-only view with the
        // same lifetime and no alignment requirement on the destination.
        let block_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                dequantized_blocks.as_ptr().cast::<u8>(),
                std::mem::size_of_val(dequantized_blocks),
            )
        };
        let blocks_dev = self.upload(block_bytes)?;

        self.launch_transcode_reversible53_idct(&blocks_dev, &samples, block_count)?;
        if low_height > 0 {
            self.launch_transcode_reversible53_vertical(
                CudaKernel::TranscodeReversible53VerticalLow,
                &samples,
                dims,
                &v_low,
                checked_i32(low_height)?,
            )?;
            self.launch_transcode_reversible53_horizontal(
                CudaKernel::TranscodeReversible53HorizontalLow,
                &v_low,
                dims,
                checked_i32(low_height)?,
                &ll,
                &hl,
            )?;
        }
        if high_height > 0 {
            self.launch_transcode_reversible53_vertical(
                CudaKernel::TranscodeReversible53VerticalHigh,
                &samples,
                dims,
                &v_high,
                checked_i32(high_height)?,
            )?;
            self.launch_transcode_reversible53_horizontal(
                CudaKernel::TranscodeReversible53HorizontalHigh,
                &v_high,
                dims,
                checked_i32(high_height)?,
                &lh,
                &hh,
            )?;
        }

        Ok(CudaTranscodeReversible53Bands {
            ll: Self::download_i32_band(&ll, low_width * low_height)?,
            hl: Self::download_i32_band(&hl, high_width * low_height)?,
            lh: Self::download_i32_band(&lh, low_width * high_height)?,
            hh: Self::download_i32_band(&hh, high_width * high_height)?,
            low_width,
            low_height,
            high_width,
            high_height,
        })
    }

    fn launch_transcode_reversible53_idct(
        &self,
        blocks: &CudaDeviceBuffer,
        samples: &CudaDeviceBuffer,
        block_count: usize,
    ) -> Result<(), CudaError> {
        if block_count == 0 {
            return Ok(());
        }
        let function = self.transcode_kernel_function(CudaKernel::TranscodeReversible53Idct)?;
        let mut blocks_ptr = blocks.device_ptr();
        let mut samples_ptr = samples.device_ptr();
        let mut count = u32::try_from(block_count)
            .map_err(|_| CudaError::LengthTooLarge { len: block_count })?;
        let mut params = cuda_kernel_params!(blocks_ptr, samples_ptr, count);
        let geometry = copy_u8_launch_geometry(block_count)
            .ok_or(CudaError::LengthTooLarge { len: block_count })?;
        self.launch_kernel(function, geometry, &mut params)
    }

    fn launch_transcode_reversible53_vertical(
        &self,
        kernel: CudaKernel,
        samples: &CudaDeviceBuffer,
        dims: Reversible53Dims,
        out: &CudaDeviceBuffer,
        out_rows: i32,
    ) -> Result<(), CudaError> {
        let function = self.transcode_kernel_function(kernel)?;
        let mut samples_ptr = samples.device_ptr();
        let mut block_cols = dims.block_cols;
        let mut width = dims.width;
        let mut height = dims.height;
        let mut out_ptr = out.device_ptr();
        let mut rows = out_rows;
        let mut params = cuda_kernel_params!(samples_ptr, block_cols, width, height, out_ptr, rows);
        let grid_w = u32::try_from(dims.width).map_err(|_| CudaError::LengthTooLarge { len: 0 })?;
        let grid_h = u32::try_from(out_rows).map_err(|_| CudaError::LengthTooLarge { len: 0 })?;
        let geometry = j2k_dwt53_launch_geometry(grid_w, grid_h)
            .ok_or(CudaError::LengthTooLarge { len: 0 })?;
        self.launch_kernel(function, geometry, &mut params)
    }

    fn launch_transcode_reversible53_horizontal(
        &self,
        kernel: CudaKernel,
        rows_buffer: &CudaDeviceBuffer,
        dims: Reversible53Dims,
        n_rows: i32,
        low_out: &CudaDeviceBuffer,
        high_out: &CudaDeviceBuffer,
    ) -> Result<(), CudaError> {
        let row_count =
            usize::try_from(n_rows).map_err(|_| CudaError::LengthTooLarge { len: 0 })?;
        if row_count == 0 {
            return Ok(());
        }
        let function = self.transcode_kernel_function(kernel)?;
        let mut rows_ptr = rows_buffer.device_ptr();
        let mut width = dims.width;
        let mut rows = n_rows;
        let mut low_width = dims.low_width;
        let mut high_width = dims.high_width;
        let mut low_ptr = low_out.device_ptr();
        let mut high_ptr = high_out.device_ptr();
        let mut params =
            cuda_kernel_params!(rows_ptr, width, rows, low_width, high_width, low_ptr, high_ptr);
        let geometry = copy_u8_launch_geometry(row_count)
            .ok_or(CudaError::LengthTooLarge { len: row_count })?;
        self.launch_kernel(function, geometry, &mut params)
    }

    fn transcode_kernel_function(&self, kernel: CudaKernel) -> Result<CuFunction, CudaError> {
        #[cfg(feature = "cuda-oxide-transcode")]
        {
            if crate::build_flags::cuda_oxide_transcode_enabled()
                && kernel.is_transcode_reversible53_stage()
            {
                return self.inner.cuda_oxide_transcode_kernel_function(kernel);
            }
        }
        self.inner.kernel_function(kernel)
    }
}

/// Irreversible single-level 9/7 transcode bands downloaded from the device.
/// Device math is f32; callers widen to f64 (parity is within tolerance).
#[derive(Clone, Debug, PartialEq)]
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

impl Dwt97BatchInput<'_> {
    fn len(self) -> usize {
        match self {
            Self::F32(blocks) => blocks.len(),
            Self::I16(blocks) => blocks.len(),
        }
    }

    fn upload(self, pool: &CudaBufferPool) -> Result<CudaPooledDeviceBuffer, CudaError> {
        match self {
            Self::F32(blocks) => pool.upload_f32(blocks),
            Self::I16(blocks) => {
                let bytes = i16_slice_as_bytes(blocks);
                if should_use_pinned_pooled_i16_upload(bytes.len()) {
                    pool.upload_pinned(bytes)
                } else {
                    pool.upload(bytes)
                }
            }
        }
    }
}

pub(crate) fn should_use_pinned_pooled_i16_upload(byte_len: usize) -> bool {
    byte_len <= PINNED_POOLED_I16_UPLOAD_MAX_BYTES
}

pub(crate) fn validate_dct_block_grid(
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
    item_count: usize,
    coeff_len: usize,
    invalid_message: &'static str,
) -> Result<DctBlockGrid, CudaError> {
    let block_count = block_cols
        .checked_mul(block_rows)
        .ok_or(CudaError::LengthTooLarge { len: block_cols })?;
    let covered_w = block_cols
        .checked_mul(8)
        .ok_or(CudaError::LengthTooLarge { len: block_cols })?;
    let covered_h = block_rows
        .checked_mul(8)
        .ok_or(CudaError::LengthTooLarge { len: block_rows })?;
    let per_item_coeffs = block_count
        .checked_mul(64)
        .ok_or(CudaError::LengthTooLarge { len: block_count })?;
    let expected_coeffs =
        per_item_coeffs
            .checked_mul(item_count)
            .ok_or(CudaError::LengthTooLarge {
                len: per_item_coeffs,
            })?;
    if item_count == 0
        || width == 0
        || height == 0
        || width > covered_w
        || height > covered_h
        || coeff_len != expected_coeffs
    {
        return Err(CudaError::InvalidArgument {
            message: invalid_message.to_string(),
        });
    }

    let low_width = width.div_ceil(2);
    let low_height = height.div_ceil(2);
    let high_width = width / 2;
    let high_height = height / 2;
    Ok(DctBlockGrid {
        block_count,
        expected_coeffs,
        low_width,
        low_height,
        high_width,
        high_height,
        dims: Reversible53Dims {
            block_cols: checked_i32(block_cols)?,
            width: checked_i32(width)?,
            height: checked_i32(height)?,
            low_width: checked_i32(low_width)?,
            high_width: checked_i32(high_width)?,
        },
    })
}

pub(crate) fn checked_i32(value: usize) -> Result<i32, CudaError> {
    i32::try_from(value).map_err(|_| CudaError::LengthTooLarge { len: value })
}

impl CudaContext {
    /// Compute one irreversible single-level 9/7 transform directly from
    /// dequantized 8x8 DCT blocks (`block_cols * block_rows` blocks of 64 `f32`
    /// natural-order coefficients), matching the `j2k-transcode` scalar
    /// oracle within f32 tolerance.
    #[allow(clippy::too_many_lines)]
    pub fn j2k_transcode_dwt97(
        &self,
        blocks: &[f32],
        block_cols: usize,
        block_rows: usize,
        width: usize,
        height: usize,
    ) -> Result<CudaTranscodeDwt97Bands, CudaError> {
        if !TRANSCODE_PTX_BUILT_FROM_CUDA {
            return Err(CudaError::InvalidArgument {
                message: "CUDA transcode kernels were not built (nvcc unavailable at build time)"
                    .to_string(),
            });
        }
        let grid = validate_dct_block_grid(
            block_cols,
            block_rows,
            width,
            height,
            1,
            blocks.len(),
            "9/7 transcode job has unsupported grid geometry",
        )?;
        let DctBlockGrid {
            expected_coeffs: _,
            low_width,
            low_height,
            high_width,
            high_height,
            dims,
            ..
        } = grid;

        self.inner.set_current()?;

        let alloc_f32 = |count: usize| -> Result<CudaDeviceBuffer, CudaError> {
            let bytes = count
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or(CudaError::LengthTooLarge { len: count })?;
            self.allocate(bytes)
        };
        let spatial = alloc_f32(width * height)?;
        let row_low = alloc_f32(height * low_width)?;
        let row_high = alloc_f32(height * high_width)?;
        let ll = alloc_f32(low_width * low_height)?;
        let lh = alloc_f32(low_width * high_height)?;
        let hl = alloc_f32(high_width * low_height)?;
        let hh = alloc_f32(high_width * high_height)?;

        let blocks_dev = self.upload_f32(blocks)?;

        self.launch_transcode_dwt97_idct(dims, &blocks_dev, &spatial)?;
        self.launch_transcode_dwt97_row_lift(dims, &spatial, &row_low, &row_high)?;
        if dims.low_width > 0 {
            self.launch_transcode_dwt97_column_lift(
                &row_low,
                dims.low_width,
                dims.height,
                &ll,
                &lh,
            )?;
        }
        if dims.high_width > 0 {
            self.launch_transcode_dwt97_column_lift(
                &row_high,
                dims.high_width,
                dims.height,
                &hl,
                &hh,
            )?;
        }

        Ok(CudaTranscodeDwt97Bands {
            ll: Self::download_f32_band(&ll, low_width * low_height)?,
            hl: Self::download_f32_band(&hl, high_width * low_height)?,
            lh: Self::download_f32_band(&lh, low_width * high_height)?,
            hh: Self::download_f32_band(&hh, high_width * high_height)?,
            low_width,
            low_height,
            high_width,
            high_height,
        })
    }

    fn launch_transcode_dwt97_idct(
        &self,
        dims: Reversible53Dims,
        blocks: &CudaDeviceBuffer,
        spatial: &CudaDeviceBuffer,
    ) -> Result<(), CudaError> {
        let function = self.inner.kernel_function(CudaKernel::TranscodeDwt97Idct)?;
        let mut blocks_ptr = blocks.device_ptr();
        let mut block_cols = dims.block_cols;
        let mut width = dims.width;
        let mut height = dims.height;
        let mut spatial_ptr = spatial.device_ptr();
        let mut params = cuda_kernel_params!(blocks_ptr, block_cols, width, height, spatial_ptr);
        let grid_w = u32::try_from(dims.width).map_err(|_| CudaError::LengthTooLarge { len: 0 })?;
        let grid_h =
            u32::try_from(dims.height).map_err(|_| CudaError::LengthTooLarge { len: 0 })?;
        let geometry = j2k_dwt53_launch_geometry(grid_w, grid_h)
            .ok_or(CudaError::LengthTooLarge { len: 0 })?;
        self.launch_kernel(function, geometry, &mut params)
    }

    fn launch_transcode_dwt97_row_lift(
        &self,
        dims: Reversible53Dims,
        spatial: &CudaDeviceBuffer,
        row_low: &CudaDeviceBuffer,
        row_high: &CudaDeviceBuffer,
    ) -> Result<(), CudaError> {
        let function = self
            .inner
            .kernel_function(CudaKernel::TranscodeDwt97RowLift)?;
        let mut spatial_ptr = spatial.device_ptr();
        let mut width = dims.width;
        let mut height = dims.height;
        let mut low_width = dims.low_width;
        let mut high_width = dims.high_width;
        let mut low_ptr = row_low.device_ptr();
        let mut high_ptr = row_high.device_ptr();
        let mut params = cuda_kernel_params!(
            spatial_ptr,
            width,
            height,
            low_width,
            high_width,
            low_ptr,
            high_ptr
        );
        let rows =
            usize::try_from(dims.height).map_err(|_| CudaError::LengthTooLarge { len: 0 })?;
        let geometry =
            copy_u8_launch_geometry(rows).ok_or(CudaError::LengthTooLarge { len: rows })?;
        self.launch_kernel(function, geometry, &mut params)
    }

    fn launch_transcode_dwt97_column_lift(
        &self,
        rows_buffer: &CudaDeviceBuffer,
        band_width: i32,
        height: i32,
        low_out: &CudaDeviceBuffer,
        high_out: &CudaDeviceBuffer,
    ) -> Result<(), CudaError> {
        let columns =
            usize::try_from(band_width).map_err(|_| CudaError::LengthTooLarge { len: 0 })?;
        if columns == 0 {
            return Ok(());
        }
        let function = self
            .inner
            .kernel_function(CudaKernel::TranscodeDwt97ColumnLift)?;
        let mut rows_ptr = rows_buffer.device_ptr();
        let mut band = band_width;
        let mut rows = height;
        let mut low_ptr = low_out.device_ptr();
        let mut high_ptr = high_out.device_ptr();
        let mut params = cuda_kernel_params!(rows_ptr, band, rows, low_ptr, high_ptr);
        let geometry =
            copy_u8_launch_geometry(columns).ok_or(CudaError::LengthTooLarge { len: columns })?;
        self.launch_kernel(function, geometry, &mut params)
    }
}

impl CudaContext {
    /// Compute a same-geometry batch of irreversible single-level 9/7 transforms
    /// with one batched launch per stage, returning per-item bands plus real
    /// backend stage timings. All jobs must share geometry (`block_cols`,
    /// `block_rows`, `width`, `height`); `blocks` is the items' natural-order
    /// `f32` coefficients laid out contiguously (`item_count * block_cols *
    /// block_rows * 64`). Bit-identical to running `j2k_transcode_dwt97` per item.
    #[allow(clippy::similar_names)]
    pub fn j2k_transcode_dwt97_batch(
        &self,
        blocks: &[f32],
        item_count: usize,
        block_cols: usize,
        block_rows: usize,
        width: usize,
        height: usize,
    ) -> Result<(Vec<CudaTranscodeDwt97Bands>, CudaDwt97BatchStageTimings), CudaError> {
        let pool = self.buffer_pool();
        self.j2k_transcode_dwt97_batch_with_pool(
            blocks, item_count, block_cols, block_rows, width, height, &pool,
        )
    }

    /// Compute a same-geometry batch of irreversible single-level 9/7 transforms
    /// while reusing device buffers from `pool` for transient stage storage.
    #[allow(clippy::too_many_arguments, clippy::similar_names)]
    pub fn j2k_transcode_dwt97_batch_with_pool(
        &self,
        blocks: &[f32],
        item_count: usize,
        block_cols: usize,
        block_rows: usize,
        width: usize,
        height: usize,
        pool: &CudaBufferPool,
    ) -> Result<(Vec<CudaTranscodeDwt97Bands>, CudaDwt97BatchStageTimings), CudaError> {
        let (bands, pack_upload_us, idct_row_lift_us, column_lift_us) = self
            .transcode_dwt97_batch_to_device(
                blocks, item_count, block_cols, block_rows, width, height, pool,
            )?;
        let Dwt97BatchDeviceBands {
            ll,
            lh,
            hl,
            hh,
            low_width,
            low_height,
            high_width,
            high_height,
        } = bands;

        let ll_size = low_width * low_height;
        let lh_size = low_width * high_height;
        let hl_size = high_width * low_height;
        let hh_size = high_width * high_height;

        let (outputs, readback_us) = self.time_default_stream_us(|| {
            let ll_all = Self::download_pooled_f32_band(&ll, item_count * ll_size)?;
            let lh_all = Self::download_pooled_f32_band(&lh, item_count * lh_size)?;
            let hl_all = Self::download_pooled_f32_band(&hl, item_count * hl_size)?;
            let hh_all = Self::download_pooled_f32_band(&hh, item_count * hh_size)?;
            let mut outputs = Vec::with_capacity(item_count);
            for item in 0..item_count {
                outputs.push(CudaTranscodeDwt97Bands {
                    ll: ll_all[item * ll_size..(item + 1) * ll_size].to_vec(),
                    hl: hl_all[item * hl_size..(item + 1) * hl_size].to_vec(),
                    lh: lh_all[item * lh_size..(item + 1) * lh_size].to_vec(),
                    hh: hh_all[item * hh_size..(item + 1) * hh_size].to_vec(),
                    low_width,
                    low_height,
                    high_width,
                    high_height,
                });
            }
            Ok(outputs)
        })?;

        Ok((
            outputs,
            CudaDwt97BatchStageTimings {
                pack_upload_us,
                idct_row_lift_us,
                column_lift_us,
                quantize_codeblock_us: 0,
                ht_encode_us: 0,
                ht_codeblock_dispatches: 0,
                readback_us,
            },
        ))
    }

    /// Compute a same-geometry batch directly into device-resident
    /// prequantized HTJ2K code-block coefficients: staged 9/7 followed by
    /// per-subband deadzone quantization into code-block-major `i32` layout.
    /// `params` carries the per-subband inverse step sizes and the code-block
    /// geometry.
    #[allow(clippy::too_many_arguments)]
    pub fn j2k_transcode_htj2k97_codeblock_batch_resident(
        &self,
        blocks: &[f32],
        item_count: usize,
        block_cols: usize,
        block_rows: usize,
        width: usize,
        height: usize,
        params: CudaHtj2k97QuantizeParams,
    ) -> Result<(CudaHtj2k97DeviceCodeblockBands, CudaDwt97BatchStageTimings), CudaError> {
        let pool = self.buffer_pool();
        self.j2k_transcode_htj2k97_codeblock_batch_resident_with_pool(
            blocks, item_count, block_cols, block_rows, width, height, params, &pool,
        )
    }

    /// Compute a same-geometry batch directly into device-resident
    /// prequantized HTJ2K code-block coefficients while reusing transient stage
    /// buffers from `pool`.
    #[allow(clippy::similar_names, clippy::too_many_arguments)]
    pub fn j2k_transcode_htj2k97_codeblock_batch_resident_with_pool(
        &self,
        blocks: &[f32],
        item_count: usize,
        block_cols: usize,
        block_rows: usize,
        width: usize,
        height: usize,
        params: CudaHtj2k97QuantizeParams,
        pool: &CudaBufferPool,
    ) -> Result<(CudaHtj2k97DeviceCodeblockBands, CudaDwt97BatchStageTimings), CudaError> {
        let (bands, pack_upload_us, idct_row_lift_us, column_lift_us) = self
            .transcode_dwt97_batch_to_device(
                blocks, item_count, block_cols, block_rows, width, height, pool,
            )?;
        let low_width = bands.low_width;
        let low_height = bands.low_height;
        let high_width = bands.high_width;
        let high_height = bands.high_height;
        let items =
            u32::try_from(item_count).map_err(|_| CudaError::LengthTooLarge { len: item_count })?;

        let alloc_i32 = |count: usize| -> Result<CudaPooledDeviceBuffer, CudaError> {
            let bytes = count
                .checked_mul(std::mem::size_of::<i32>())
                .ok_or(CudaError::LengthTooLarge { len: count })?;
            pool.take(bytes)
        };
        let ll_size = low_width * low_height;
        let lh_size = low_width * high_height;
        let hl_size = high_width * low_height;
        let hh_size = high_width * high_height;

        let ll_q = alloc_i32(item_count * ll_size)?;
        let lh_q = alloc_i32(item_count * lh_size)?;
        let hl_q = alloc_i32(item_count * hl_size)?;
        let hh_q = alloc_i32(item_count * hh_size)?;

        let ((), quantize_codeblock_us) = self.time_default_stream_us(|| {
            self.launch_transcode_dwt97_quantize_codeblock_bands(
                &bands,
                Dwt97CodeblockBandBuffers {
                    ll: pooled_device_buffer(&ll_q)?,
                    hl: pooled_device_buffer(&hl_q)?,
                    lh: pooled_device_buffer(&lh_q)?,
                    hh: pooled_device_buffer(&hh_q)?,
                },
                params,
                items,
            )
        })?;

        Ok((
            CudaHtj2k97DeviceCodeblockBands {
                ll: ll_q,
                hl: hl_q,
                lh: lh_q,
                hh: hh_q,
                item_count,
                low_width,
                low_height,
                high_width,
                high_height,
            },
            CudaDwt97BatchStageTimings {
                pack_upload_us,
                idct_row_lift_us,
                column_lift_us,
                quantize_codeblock_us,
                ht_encode_us: 0,
                ht_codeblock_dispatches: 0,
                readback_us: 0,
            },
        ))
    }

    /// Compute a same-geometry i16 batch directly into device-resident
    /// prequantized HTJ2K code-block coefficients while reusing transient stage
    /// buffers from `pool`.
    #[allow(clippy::similar_names, clippy::too_many_arguments)]
    pub fn j2k_transcode_htj2k97_codeblock_i16_batch_resident_with_pool(
        &self,
        blocks: &[i16],
        item_count: usize,
        block_cols: usize,
        block_rows: usize,
        width: usize,
        height: usize,
        params: CudaHtj2k97QuantizeParams,
        pool: &CudaBufferPool,
    ) -> Result<(CudaHtj2k97DeviceCodeblockBands, CudaDwt97BatchStageTimings), CudaError> {
        if !dwt97_fused_column_quantize_disabled() {
            return self.j2k_transcode_htj2k97_codeblock_i16_batch_resident_fused_with_pool(
                blocks, item_count, block_cols, block_rows, width, height, params, pool,
            );
        }

        let (bands, pack_upload_us, idct_row_lift_us, column_lift_us) = self
            .transcode_dwt97_i16_batch_to_device(
                blocks, item_count, block_cols, block_rows, width, height, pool,
            )?;
        let low_width = bands.low_width;
        let low_height = bands.low_height;
        let high_width = bands.high_width;
        let high_height = bands.high_height;
        let items =
            u32::try_from(item_count).map_err(|_| CudaError::LengthTooLarge { len: item_count })?;

        let alloc_i32 = |count: usize| -> Result<CudaPooledDeviceBuffer, CudaError> {
            let bytes = count
                .checked_mul(std::mem::size_of::<i32>())
                .ok_or(CudaError::LengthTooLarge { len: count })?;
            pool.take(bytes)
        };
        let ll_size = low_width * low_height;
        let lh_size = low_width * high_height;
        let hl_size = high_width * low_height;
        let hh_size = high_width * high_height;

        let ll_q = alloc_i32(item_count * ll_size)?;
        let lh_q = alloc_i32(item_count * lh_size)?;
        let hl_q = alloc_i32(item_count * hl_size)?;
        let hh_q = alloc_i32(item_count * hh_size)?;

        let ((), quantize_codeblock_us) = self.time_default_stream_us(|| {
            self.launch_transcode_dwt97_quantize_codeblock_bands(
                &bands,
                Dwt97CodeblockBandBuffers {
                    ll: pooled_device_buffer(&ll_q)?,
                    hl: pooled_device_buffer(&hl_q)?,
                    lh: pooled_device_buffer(&lh_q)?,
                    hh: pooled_device_buffer(&hh_q)?,
                },
                params,
                items,
            )
        })?;

        Ok((
            CudaHtj2k97DeviceCodeblockBands {
                ll: ll_q,
                hl: hl_q,
                lh: lh_q,
                hh: hh_q,
                item_count,
                low_width,
                low_height,
                high_width,
                high_height,
            },
            CudaDwt97BatchStageTimings {
                pack_upload_us,
                idct_row_lift_us,
                column_lift_us,
                quantize_codeblock_us,
                ht_encode_us: 0,
                ht_codeblock_dispatches: 0,
                readback_us: 0,
            },
        ))
    }

    #[allow(
        clippy::similar_names,
        clippy::too_many_arguments,
        clippy::too_many_lines
    )]
    fn j2k_transcode_htj2k97_codeblock_i16_batch_resident_fused_with_pool(
        &self,
        blocks: &[i16],
        item_count: usize,
        block_cols: usize,
        block_rows: usize,
        width: usize,
        height: usize,
        params: CudaHtj2k97QuantizeParams,
        pool: &CudaBufferPool,
    ) -> Result<(CudaHtj2k97DeviceCodeblockBands, CudaDwt97BatchStageTimings), CudaError> {
        if !TRANSCODE_PTX_BUILT_FROM_CUDA {
            return Err(CudaError::InvalidArgument {
                message: "CUDA transcode kernels were not built (nvcc unavailable at build time)"
                    .to_string(),
            });
        }
        let grid = validate_dct_block_grid(
            block_cols,
            block_rows,
            width,
            height,
            item_count,
            blocks.len(),
            "9/7 transcode batch has unsupported grid geometry",
        )?;
        let DctBlockGrid {
            block_count,
            low_width,
            low_height,
            high_width,
            high_height,
            dims,
            ..
        } = grid;
        let items =
            u32::try_from(item_count).map_err(|_| CudaError::LengthTooLarge { len: item_count })?;
        let blocks_per_item = checked_i32(block_count)?;
        let low_height_i32 = checked_i32(low_height)?;
        let high_height_i32 = checked_i32(high_height)?;
        let cb_w = checked_i32(params.cb_width)?;
        let cb_h = checked_i32(params.cb_height)?;

        self.inner.set_current()?;

        let alloc_f32 = |count: usize| -> Result<CudaPooledDeviceBuffer, CudaError> {
            let bytes = count
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or(CudaError::LengthTooLarge { len: count })?;
            pool.take(bytes)
        };
        let alloc_i32 = |count: usize| -> Result<CudaPooledDeviceBuffer, CudaError> {
            let bytes = count
                .checked_mul(std::mem::size_of::<i32>())
                .ok_or(CudaError::LengthTooLarge { len: count })?;
            pool.take(bytes)
        };
        let (buffers, pack_upload_us) = self.time_default_stream_us(|| {
            let spatial = alloc_f32(item_count * width * height)?;
            let row_low = alloc_f32(item_count * height * low_width)?;
            let row_high = alloc_f32(item_count * height * high_width)?;
            let blocks_dev = Dwt97BatchInput::I16(blocks).upload(pool)?;
            Ok((spatial, row_low, row_high, blocks_dev))
        })?;
        let (spatial, row_low, row_high, blocks_dev) = buffers;

        let ll_size = low_width * low_height;
        let lh_size = low_width * high_height;
        let hl_size = high_width * low_height;
        let hh_size = high_width * high_height;

        let ll_q = alloc_i32(item_count * ll_size)?;
        let lh_q = alloc_i32(item_count * lh_size)?;
        let hl_q = alloc_i32(item_count * hl_size)?;
        let hh_q = alloc_i32(item_count * hh_size)?;

        let ((), idct_row_lift_us) = self.time_default_stream_us(|| {
            self.launch_transcode_dwt97_idct_batch_kernel(
                CudaKernel::TranscodeDwt97IdctI16Batch,
                dims,
                blocks_per_item,
                items,
                pooled_device_buffer(&blocks_dev)?,
                pooled_device_buffer(&spatial)?,
            )?;
            self.launch_transcode_dwt97_row_lift_batch(
                dims,
                items,
                pooled_device_buffer(&spatial)?,
                pooled_device_buffer(&row_low)?,
                pooled_device_buffer(&row_high)?,
            )?;
            Ok(())
        })?;

        let ((), column_quantize_us) = self.time_default_stream_us(|| {
            if dims.low_width > 0 {
                self.launch_transcode_dwt97_column_lift_quantize_codeblocks_batch(
                    pooled_device_buffer(&row_low)?,
                    dims.low_width,
                    dims.height,
                    low_height_i32,
                    high_height_i32,
                    items,
                    pooled_device_buffer(&ll_q)?,
                    pooled_device_buffer(&lh_q)?,
                    cb_w,
                    cb_h,
                    params.inv_delta_ll,
                    params.inv_delta_lh,
                )?;
            }
            if dims.high_width > 0 {
                self.launch_transcode_dwt97_column_lift_quantize_codeblocks_batch(
                    pooled_device_buffer(&row_high)?,
                    dims.high_width,
                    dims.height,
                    low_height_i32,
                    high_height_i32,
                    items,
                    pooled_device_buffer(&hl_q)?,
                    pooled_device_buffer(&hh_q)?,
                    cb_w,
                    cb_h,
                    params.inv_delta_hl,
                    params.inv_delta_hh,
                )?;
            }
            Ok(())
        })?;

        Ok((
            CudaHtj2k97DeviceCodeblockBands {
                ll: ll_q,
                hl: hl_q,
                lh: lh_q,
                hh: hh_q,
                item_count,
                low_width,
                low_height,
                high_width,
                high_height,
            },
            CudaDwt97BatchStageTimings {
                pack_upload_us,
                idct_row_lift_us,
                column_lift_us: 0,
                quantize_codeblock_us: column_quantize_us,
                ht_encode_us: 0,
                ht_codeblock_dispatches: 0,
                readback_us: 0,
            },
        ))
    }

    /// Compute a same-geometry batch directly into host-owned prequantized
    /// HTJ2K code-block coefficients: staged 9/7 followed by per-subband
    /// deadzone quantization into code-block-major `i32` layout.
    #[allow(clippy::too_many_arguments)]
    pub fn j2k_transcode_htj2k97_codeblock_batch(
        &self,
        blocks: &[f32],
        item_count: usize,
        block_cols: usize,
        block_rows: usize,
        width: usize,
        height: usize,
        params: CudaHtj2k97QuantizeParams,
    ) -> Result<(CudaHtj2k97CodeblockBands, CudaDwt97BatchStageTimings), CudaError> {
        let pool = self.buffer_pool();
        self.j2k_transcode_htj2k97_codeblock_batch_with_pool(
            blocks, item_count, block_cols, block_rows, width, height, params, &pool,
        )
    }

    /// Compute a same-geometry batch directly into host-owned prequantized
    /// HTJ2K code-block coefficients while reusing transient stage buffers
    /// from `pool`.
    #[allow(clippy::similar_names, clippy::too_many_arguments)]
    pub fn j2k_transcode_htj2k97_codeblock_batch_with_pool(
        &self,
        blocks: &[f32],
        item_count: usize,
        block_cols: usize,
        block_rows: usize,
        width: usize,
        height: usize,
        params: CudaHtj2k97QuantizeParams,
        pool: &CudaBufferPool,
    ) -> Result<(CudaHtj2k97CodeblockBands, CudaDwt97BatchStageTimings), CudaError> {
        let (bands, pack_upload_us, idct_row_lift_us, column_lift_us) = self
            .transcode_dwt97_batch_to_device(
                blocks, item_count, block_cols, block_rows, width, height, pool,
            )?;
        let low_width = bands.low_width;
        let low_height = bands.low_height;
        let high_width = bands.high_width;
        let high_height = bands.high_height;
        let items =
            u32::try_from(item_count).map_err(|_| CudaError::LengthTooLarge { len: item_count })?;

        let alloc_i32 = |count: usize| -> Result<CudaDeviceBuffer, CudaError> {
            let bytes = count
                .checked_mul(std::mem::size_of::<i32>())
                .ok_or(CudaError::LengthTooLarge { len: count })?;
            self.allocate(bytes)
        };
        let ll_size = low_width * low_height;
        let lh_size = low_width * high_height;
        let hl_size = high_width * low_height;
        let hh_size = high_width * high_height;

        let ll_q = alloc_i32(item_count * ll_size)?;
        let lh_q = alloc_i32(item_count * lh_size)?;
        let hl_q = alloc_i32(item_count * hl_size)?;
        let hh_q = alloc_i32(item_count * hh_size)?;

        let ((), quantize_codeblock_us) = self.time_default_stream_us(|| {
            self.launch_transcode_dwt97_quantize_codeblock_bands(
                &bands,
                Dwt97CodeblockBandBuffers {
                    ll: &ll_q,
                    hl: &hl_q,
                    lh: &lh_q,
                    hh: &hh_q,
                },
                params,
                items,
            )
        })?;

        let (codeblocks, readback_us) = self.time_default_stream_us(|| {
            Ok(CudaHtj2k97CodeblockBands {
                ll: Self::download_i32_band(&ll_q, item_count * ll_size)?,
                hl: Self::download_i32_band(&hl_q, item_count * hl_size)?,
                lh: Self::download_i32_band(&lh_q, item_count * lh_size)?,
                hh: Self::download_i32_band(&hh_q, item_count * hh_size)?,
                item_count,
                low_width,
                low_height,
                high_width,
                high_height,
            })
        })?;

        Ok((
            codeblocks,
            CudaDwt97BatchStageTimings {
                pack_upload_us,
                idct_row_lift_us,
                column_lift_us,
                quantize_codeblock_us,
                ht_encode_us: 0,
                ht_codeblock_dispatches: 0,
                readback_us,
            },
        ))
    }

    /// Run the shared staged 9/7 batch pipeline (alloc + upload, batched IDCT +
    /// row lift, batched column lift) and return the device-resident bands plus
    /// the three pre-readback stage timings.
    #[allow(clippy::too_many_lines)]
    #[allow(clippy::too_many_arguments)]
    fn transcode_dwt97_batch_to_device(
        &self,
        blocks: &[f32],
        item_count: usize,
        block_cols: usize,
        block_rows: usize,
        width: usize,
        height: usize,
        pool: &CudaBufferPool,
    ) -> Result<(Dwt97BatchDeviceBands, u128, u128, u128), CudaError> {
        self.transcode_dwt97_batch_input_to_device(
            Dwt97BatchInput::F32(blocks),
            item_count,
            block_cols,
            block_rows,
            width,
            height,
            pool,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn transcode_dwt97_i16_batch_to_device(
        &self,
        blocks: &[i16],
        item_count: usize,
        block_cols: usize,
        block_rows: usize,
        width: usize,
        height: usize,
        pool: &CudaBufferPool,
    ) -> Result<(Dwt97BatchDeviceBands, u128, u128, u128), CudaError> {
        self.transcode_dwt97_batch_input_to_device(
            Dwt97BatchInput::I16(blocks),
            item_count,
            block_cols,
            block_rows,
            width,
            height,
            pool,
        )
    }

    #[allow(clippy::too_many_lines)]
    #[allow(clippy::too_many_arguments)]
    fn transcode_dwt97_batch_input_to_device(
        &self,
        input: Dwt97BatchInput<'_>,
        item_count: usize,
        block_cols: usize,
        block_rows: usize,
        width: usize,
        height: usize,
        pool: &CudaBufferPool,
    ) -> Result<(Dwt97BatchDeviceBands, u128, u128, u128), CudaError> {
        if !TRANSCODE_PTX_BUILT_FROM_CUDA {
            return Err(CudaError::InvalidArgument {
                message: "CUDA transcode kernels were not built (nvcc unavailable at build time)"
                    .to_string(),
            });
        }
        let grid = validate_dct_block_grid(
            block_cols,
            block_rows,
            width,
            height,
            item_count,
            input.len(),
            "9/7 transcode batch has unsupported grid geometry",
        )?;
        let DctBlockGrid {
            block_count,
            low_width,
            low_height,
            high_width,
            high_height,
            dims,
            ..
        } = grid;
        let items =
            u32::try_from(item_count).map_err(|_| CudaError::LengthTooLarge { len: item_count })?;
        let blocks_per_item = checked_i32(block_count)?;
        let low_height_i32 = checked_i32(low_height)?;
        let high_height_i32 = checked_i32(high_height)?;

        self.inner.set_current()?;

        let alloc_f32 = |count: usize| -> Result<CudaPooledDeviceBuffer, CudaError> {
            let bytes = count
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or(CudaError::LengthTooLarge { len: count })?;
            pool.take(bytes)
        };

        // Stage: allocate batch buffers and upload all blocks.
        let (buffers, pack_upload_us) = self.time_default_stream_us(|| {
            let spatial = alloc_f32(item_count * width * height)?;
            let row_low = alloc_f32(item_count * height * low_width)?;
            let row_high = alloc_f32(item_count * height * high_width)?;
            let ll = alloc_f32(item_count * low_width * low_height)?;
            let lh = alloc_f32(item_count * low_width * high_height)?;
            let hl = alloc_f32(item_count * high_width * low_height)?;
            let hh = alloc_f32(item_count * high_width * high_height)?;
            let blocks_dev = input.upload(pool)?;
            Ok((spatial, row_low, row_high, ll, lh, hl, hh, blocks_dev))
        })?;
        let (spatial, row_low, row_high, ll, lh, hl, hh, blocks_dev) = buffers;

        // Stage: batched separable IDCT then horizontal 9/7 row lift.
        let ((), idct_row_lift_us) = self.time_default_stream_us(|| {
            let idct_kernel = match input {
                Dwt97BatchInput::F32(_) => CudaKernel::TranscodeDwt97IdctBatch,
                Dwt97BatchInput::I16(_) => CudaKernel::TranscodeDwt97IdctI16Batch,
            };
            self.launch_transcode_dwt97_idct_batch_kernel(
                idct_kernel,
                dims,
                blocks_per_item,
                items,
                pooled_device_buffer(&blocks_dev)?,
                pooled_device_buffer(&spatial)?,
            )?;
            self.launch_transcode_dwt97_row_lift_batch(
                dims,
                items,
                pooled_device_buffer(&spatial)?,
                pooled_device_buffer(&row_low)?,
                pooled_device_buffer(&row_high)?,
            )?;
            Ok(())
        })?;

        // Stage: batched vertical 9/7 column lift for both low and high rows.
        let ((), column_lift_us) = self.time_default_stream_us(|| {
            if dims.low_width > 0 {
                self.launch_transcode_dwt97_column_lift_batch(
                    pooled_device_buffer(&row_low)?,
                    dims.low_width,
                    dims.height,
                    low_height_i32,
                    high_height_i32,
                    items,
                    pooled_device_buffer(&ll)?,
                    pooled_device_buffer(&lh)?,
                )?;
            }
            if dims.high_width > 0 {
                self.launch_transcode_dwt97_column_lift_batch(
                    pooled_device_buffer(&row_high)?,
                    dims.high_width,
                    dims.height,
                    low_height_i32,
                    high_height_i32,
                    items,
                    pooled_device_buffer(&hl)?,
                    pooled_device_buffer(&hh)?,
                )?;
            }
            Ok(())
        })?;

        Ok((
            Dwt97BatchDeviceBands {
                ll,
                lh,
                hl,
                hh,
                low_width,
                low_height,
                high_width,
                high_height,
            },
            pack_upload_us,
            idct_row_lift_us,
            column_lift_us,
        ))
    }

    fn launch_transcode_dwt97_idct_batch_kernel(
        &self,
        kernel: CudaKernel,
        dims: Reversible53Dims,
        blocks_per_item: i32,
        items: u32,
        blocks: &CudaDeviceBuffer,
        spatial: &CudaDeviceBuffer,
    ) -> Result<(), CudaError> {
        let function = self.inner.kernel_function(kernel)?;
        let mut blocks_ptr = blocks.device_ptr();
        let mut block_cols = dims.block_cols;
        let mut width = dims.width;
        let mut height = dims.height;
        let mut blocks_per_item = blocks_per_item;
        let mut spatial_ptr = spatial.device_ptr();
        let mut params = cuda_kernel_params!(
            blocks_ptr,
            block_cols,
            width,
            height,
            blocks_per_item,
            spatial_ptr
        );
        let grid_w = u32::try_from(dims.width).map_err(|_| CudaError::LengthTooLarge { len: 0 })?;
        let grid_h =
            u32::try_from(dims.height).map_err(|_| CudaError::LengthTooLarge { len: 0 })?;
        let base = j2k_dwt53_launch_geometry(grid_w, grid_h)
            .ok_or(CudaError::LengthTooLarge { len: 0 })?;
        let geometry = with_grid_z(base, items);
        self.launch_kernel_async(function, geometry, &mut params)
    }

    fn launch_transcode_dwt97_row_lift_batch(
        &self,
        dims: Reversible53Dims,
        items: u32,
        spatial: &CudaDeviceBuffer,
        row_low: &CudaDeviceBuffer,
        row_high: &CudaDeviceBuffer,
    ) -> Result<(), CudaError> {
        if dims.width <= DWT97_ROW_LIFT_MAX_WIDTH {
            return self.launch_transcode_dwt97_row_lift_batch_coop(
                dims, items, spatial, row_low, row_high,
            );
        }

        let function = self
            .inner
            .kernel_function(CudaKernel::TranscodeDwt97RowLiftBatch)?;
        let mut spatial_ptr = spatial.device_ptr();
        let mut width = dims.width;
        let mut height = dims.height;
        let mut low_width = dims.low_width;
        let mut high_width = dims.high_width;
        let mut low_ptr = row_low.device_ptr();
        let mut high_ptr = row_high.device_ptr();
        let mut params = cuda_kernel_params!(
            spatial_ptr,
            width,
            height,
            low_width,
            high_width,
            low_ptr,
            high_ptr
        );
        let rows =
            usize::try_from(dims.height).map_err(|_| CudaError::LengthTooLarge { len: 0 })?;
        let base = copy_u8_launch_geometry(rows).ok_or(CudaError::LengthTooLarge { len: rows })?;
        let geometry = with_grid_y(base, items);
        self.launch_kernel_async(function, geometry, &mut params)
    }

    fn launch_transcode_dwt97_row_lift_batch_coop(
        &self,
        dims: Reversible53Dims,
        items: u32,
        spatial: &CudaDeviceBuffer,
        row_low: &CudaDeviceBuffer,
        row_high: &CudaDeviceBuffer,
    ) -> Result<(), CudaError> {
        let function = self
            .inner
            .kernel_function(CudaKernel::TranscodeDwt97RowLiftBatchCoop)?;
        let mut spatial_ptr = spatial.device_ptr();
        let mut width = dims.width;
        let mut height = dims.height;
        let mut low_width = dims.low_width;
        let mut high_width = dims.high_width;
        let mut low_ptr = row_low.device_ptr();
        let mut high_ptr = row_high.device_ptr();
        let mut params = cuda_kernel_params!(
            spatial_ptr,
            width,
            height,
            low_width,
            high_width,
            low_ptr,
            high_ptr
        );
        let rows =
            usize::try_from(dims.height).map_err(|_| CudaError::LengthTooLarge { len: 0 })?;
        let rows_per_block = DWT97_ROW_LIFT_COOP_ROWS_PER_BLOCK as usize;
        let grid_x = c_uint::try_from(rows.div_ceil(rows_per_block))
            .map_err(|_| CudaError::LengthTooLarge { len: rows })?;
        let geometry = kernels::CudaLaunchGeometry {
            grid: (grid_x, items, 1),
            block: (
                DWT97_ROW_LIFT_COOP_THREADS_X,
                DWT97_ROW_LIFT_COOP_ROWS_PER_BLOCK,
                1,
            ),
        };
        self.launch_kernel_async(function, geometry, &mut params)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_transcode_dwt97_column_lift_batch(
        &self,
        rows_buffer: &CudaDeviceBuffer,
        band_width: i32,
        height: i32,
        low_height: i32,
        high_height: i32,
        items: u32,
        low_out: &CudaDeviceBuffer,
        high_out: &CudaDeviceBuffer,
    ) -> Result<(), CudaError> {
        let columns =
            usize::try_from(band_width).map_err(|_| CudaError::LengthTooLarge { len: 0 })?;
        if columns == 0 {
            return Ok(());
        }
        let function = self
            .inner
            .kernel_function(CudaKernel::TranscodeDwt97ColumnLiftBatch)?;
        let mut rows_ptr = rows_buffer.device_ptr();
        let mut band = band_width;
        let mut rows = height;
        let mut low_h = low_height;
        let mut high_h = high_height;
        let mut low_ptr = low_out.device_ptr();
        let mut high_ptr = high_out.device_ptr();
        let mut params =
            cuda_kernel_params!(rows_ptr, band, rows, low_h, high_h, low_ptr, high_ptr);
        let base =
            copy_u8_launch_geometry(columns).ok_or(CudaError::LengthTooLarge { len: columns })?;
        let geometry = with_grid_y(base, items);
        self.launch_kernel_async(function, geometry, &mut params)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_transcode_dwt97_column_lift_quantize_codeblocks_batch(
        &self,
        rows_buffer: &CudaDeviceBuffer,
        band_width: i32,
        height: i32,
        low_height: i32,
        high_height: i32,
        items: u32,
        low_out: &CudaDeviceBuffer,
        high_out: &CudaDeviceBuffer,
        cb_width: i32,
        cb_height: i32,
        inv_delta_low: f32,
        inv_delta_high: f32,
    ) -> Result<(), CudaError> {
        let columns =
            usize::try_from(band_width).map_err(|_| CudaError::LengthTooLarge { len: 0 })?;
        if columns == 0 {
            return Ok(());
        }
        let function = self
            .inner
            .kernel_function(CudaKernel::TranscodeDwt97ColumnLiftQuantizeCodeblocksBatch)?;
        let mut rows_ptr = rows_buffer.device_ptr();
        let mut band = band_width;
        let mut rows = height;
        let mut low_h = low_height;
        let mut high_h = high_height;
        let mut low_ptr = low_out.device_ptr();
        let mut high_ptr = high_out.device_ptr();
        let mut cb_w = cb_width;
        let mut cb_h = cb_height;
        let mut inv_low = inv_delta_low;
        let mut inv_high = inv_delta_high;
        let mut params = cuda_kernel_params!(
            rows_ptr, band, rows, low_h, high_h, low_ptr, high_ptr, cb_w, cb_h, inv_low, inv_high
        );
        let base =
            copy_u8_launch_geometry(columns).ok_or(CudaError::LengthTooLarge { len: columns })?;
        let geometry = with_grid_y(base, items);
        self.launch_kernel_async(function, geometry, &mut params)
    }

    fn launch_transcode_dwt97_quantize_codeblock_bands(
        &self,
        bands: &Dwt97BatchDeviceBands,
        outputs: Dwt97CodeblockBandBuffers<'_>,
        params: CudaHtj2k97QuantizeParams,
        items: u32,
    ) -> Result<(), CudaError> {
        let to_i32 = |value: usize| -> Result<i32, CudaError> {
            i32::try_from(value).map_err(|_| CudaError::LengthTooLarge { len: value })
        };
        let low_width = to_i32(bands.low_width)?;
        let low_height = to_i32(bands.low_height)?;
        let high_width = to_i32(bands.high_width)?;
        let high_height = to_i32(bands.high_height)?;
        let cb_width = to_i32(params.cb_width)?;
        let cb_height = to_i32(params.cb_height)?;

        self.launch_transcode_dwt97_quantize_codeblocks(
            pooled_device_buffer(&bands.ll)?,
            outputs.ll,
            low_width,
            low_height,
            cb_width,
            cb_height,
            params.inv_delta_ll,
            items,
        )?;
        self.launch_transcode_dwt97_quantize_codeblocks(
            pooled_device_buffer(&bands.hl)?,
            outputs.hl,
            high_width,
            low_height,
            cb_width,
            cb_height,
            params.inv_delta_hl,
            items,
        )?;
        self.launch_transcode_dwt97_quantize_codeblocks(
            pooled_device_buffer(&bands.lh)?,
            outputs.lh,
            low_width,
            high_height,
            cb_width,
            cb_height,
            params.inv_delta_lh,
            items,
        )?;
        self.launch_transcode_dwt97_quantize_codeblocks(
            pooled_device_buffer(&bands.hh)?,
            outputs.hh,
            high_width,
            high_height,
            cb_width,
            cb_height,
            params.inv_delta_hh,
            items,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_transcode_dwt97_quantize_codeblocks(
        &self,
        band: &CudaDeviceBuffer,
        output: &CudaDeviceBuffer,
        width: i32,
        height: i32,
        cb_width: i32,
        cb_height: i32,
        inv_delta: f32,
        items: u32,
    ) -> Result<(), CudaError> {
        if width <= 0 || height <= 0 {
            return Ok(());
        }
        let function = self
            .inner
            .kernel_function(CudaKernel::TranscodeDwt97QuantizeCodeblocks)?;
        let mut band_ptr = band.device_ptr();
        let mut output_ptr = output.device_ptr();
        let mut width = width;
        let mut height = height;
        let mut cb_width = cb_width;
        let mut cb_height = cb_height;
        let mut inv_delta = inv_delta;
        let mut params = cuda_kernel_params!(
            band_ptr, output_ptr, width, height, cb_width, cb_height, inv_delta
        );
        let grid_w = u32::try_from(width).map_err(|_| CudaError::LengthTooLarge { len: 0 })?;
        let grid_h = u32::try_from(height).map_err(|_| CudaError::LengthTooLarge { len: 0 })?;
        let base = j2k_dwt53_launch_geometry(grid_w, grid_h)
            .ok_or(CudaError::LengthTooLarge { len: 0 })?;
        let geometry = with_grid_z(base, items);
        self.launch_kernel_async(function, geometry, &mut params)
    }
}
