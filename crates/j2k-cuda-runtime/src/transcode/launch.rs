// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    types::{
        Dwt97ColumnLiftBatchLaunch, Dwt97ColumnLiftQuantizeCodeblocksBatchLaunch,
        Dwt97ColumnLiftQuantizeCodeblocksParams, Dwt97QuantizeCodeblocksLaunch,
    },
    CudaHtj2k97QuantizeParams, Dwt97BatchDeviceBands, Dwt97CodeblockBandBuffers, Reversible53Dims,
};
use crate::{
    build_flags::{
        DWT97_ROW_LIFT_COOP_ROWS_PER_BLOCK, DWT97_ROW_LIFT_COOP_THREADS_X, DWT97_ROW_LIFT_MAX_WIDTH,
    },
    context::CudaContext,
    driver::CuFunction,
    error::CudaError,
    execution::cuda_kernel_param,
    kernels::{
        self, copy_u8_launch_geometry, j2k_dwt53_launch_geometry, with_grid_y, with_grid_z,
        CudaKernel,
    },
    memory::{pooled_device_buffer, CudaDeviceBuffer},
};
use std::os::raw::c_uint;

impl CudaContext {
    pub(super) fn launch_transcode_reversible53_idct(
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

    pub(super) fn launch_transcode_reversible53_vertical(
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

    pub(super) fn launch_transcode_reversible53_horizontal(
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

    pub(super) fn transcode_kernel_function(
        &self,
        kernel: CudaKernel,
    ) -> Result<CuFunction, CudaError> {
        self.inner.cuda_oxide_transcode_kernel_function(kernel)
    }
    pub(super) fn launch_transcode_dwt97_idct(
        &self,
        dims: Reversible53Dims,
        blocks: &CudaDeviceBuffer,
        spatial: &CudaDeviceBuffer,
    ) -> Result<(), CudaError> {
        let function = self.transcode_kernel_function(CudaKernel::TranscodeDwt97Idct)?;
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

    pub(super) fn launch_transcode_dwt97_row_lift(
        &self,
        dims: Reversible53Dims,
        spatial: &CudaDeviceBuffer,
        row_low: &CudaDeviceBuffer,
        row_high: &CudaDeviceBuffer,
    ) -> Result<(), CudaError> {
        let function = self.transcode_kernel_function(CudaKernel::TranscodeDwt97RowLift)?;
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

    pub(super) fn launch_transcode_dwt97_column_lift(
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
        let function = self.transcode_kernel_function(CudaKernel::TranscodeDwt97ColumnLift)?;
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
    pub(super) fn launch_transcode_dwt97_idct_batch_kernel(
        &self,
        kernel: CudaKernel,
        dims: Reversible53Dims,
        blocks_per_item: i32,
        items: u32,
        blocks: &CudaDeviceBuffer,
        spatial: &CudaDeviceBuffer,
    ) -> Result<(), CudaError> {
        let function = self.transcode_kernel_function(kernel)?;
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

    pub(super) fn launch_transcode_dwt97_row_lift_batch(
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

        let function = self.transcode_kernel_function(CudaKernel::TranscodeDwt97RowLiftBatch)?;
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

    pub(super) fn launch_transcode_dwt97_row_lift_batch_coop(
        &self,
        dims: Reversible53Dims,
        items: u32,
        spatial: &CudaDeviceBuffer,
        row_low: &CudaDeviceBuffer,
        row_high: &CudaDeviceBuffer,
    ) -> Result<(), CudaError> {
        let function =
            self.transcode_kernel_function(CudaKernel::TranscodeDwt97RowLiftBatchCoop)?;
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

    pub(super) fn launch_transcode_dwt97_column_lift_batch(
        &self,
        request: &Dwt97ColumnLiftBatchLaunch<'_>,
    ) -> Result<(), CudaError> {
        let columns = usize::try_from(request.band_width)
            .map_err(|_| CudaError::LengthTooLarge { len: 0 })?;
        if columns == 0 {
            return Ok(());
        }
        let function = self.transcode_kernel_function(CudaKernel::TranscodeDwt97ColumnLiftBatch)?;
        let mut rows_ptr = request.rows_buffer.device_ptr();
        let mut band = request.band_width;
        let mut rows = request.height;
        let mut low_h = request.low_height;
        let mut high_h = request.high_height;
        let mut low_ptr = request.low_out.device_ptr();
        let mut high_ptr = request.high_out.device_ptr();
        let mut params =
            cuda_kernel_params!(rows_ptr, band, rows, low_h, high_h, low_ptr, high_ptr);
        let base =
            copy_u8_launch_geometry(columns).ok_or(CudaError::LengthTooLarge { len: columns })?;
        let geometry = with_grid_y(base, request.items);
        self.launch_kernel_async(function, geometry, &mut params)
    }

    pub(super) fn launch_transcode_dwt97_column_lift_quantize_codeblocks_batch(
        &self,
        request: &Dwt97ColumnLiftQuantizeCodeblocksBatchLaunch<'_>,
    ) -> Result<(), CudaError> {
        let columns = usize::try_from(request.column.band_width)
            .map_err(|_| CudaError::LengthTooLarge { len: 0 })?;
        if columns == 0 {
            return Ok(());
        }
        let function = self.transcode_kernel_function(
            CudaKernel::TranscodeDwt97ColumnLiftQuantizeCodeblocksBatch,
        )?;
        let mut rows_ptr = request.column.rows_buffer.device_ptr();
        let mut band = request.column.band_width;
        let mut rows = request.column.height;
        let mut low_h = request.column.low_height;
        let mut high_h = request.column.high_height;
        let mut low_ptr = request.column.low_out.device_ptr();
        let mut high_ptr = request.column.high_out.device_ptr();
        let mut kernel_params_value = Dwt97ColumnLiftQuantizeCodeblocksParams {
            cb_width: request.cb_width,
            cb_height: request.cb_height,
            inv_delta_low: request.inv_delta_low,
            inv_delta_high: request.inv_delta_high,
        };
        let mut params = cuda_kernel_params!(
            rows_ptr,
            band,
            rows,
            low_h,
            high_h,
            low_ptr,
            high_ptr,
            kernel_params_value
        );
        let base =
            copy_u8_launch_geometry(columns).ok_or(CudaError::LengthTooLarge { len: columns })?;
        let geometry = with_grid_y(base, request.column.items);
        self.launch_kernel_async(function, geometry, &mut params)
    }

    pub(super) fn launch_transcode_dwt97_quantize_codeblock_bands(
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

        self.launch_transcode_dwt97_quantize_codeblocks(&Dwt97QuantizeCodeblocksLaunch {
            band: pooled_device_buffer(&bands.ll)?,
            output: outputs.ll,
            width: low_width,
            height: low_height,
            cb_width,
            cb_height,
            inv_delta: params.inv_delta_ll,
            items,
        })?;
        self.launch_transcode_dwt97_quantize_codeblocks(&Dwt97QuantizeCodeblocksLaunch {
            band: pooled_device_buffer(&bands.hl)?,
            output: outputs.hl,
            width: high_width,
            height: low_height,
            cb_width,
            cb_height,
            inv_delta: params.inv_delta_hl,
            items,
        })?;
        self.launch_transcode_dwt97_quantize_codeblocks(&Dwt97QuantizeCodeblocksLaunch {
            band: pooled_device_buffer(&bands.lh)?,
            output: outputs.lh,
            width: low_width,
            height: high_height,
            cb_width,
            cb_height,
            inv_delta: params.inv_delta_lh,
            items,
        })?;
        self.launch_transcode_dwt97_quantize_codeblocks(&Dwt97QuantizeCodeblocksLaunch {
            band: pooled_device_buffer(&bands.hh)?,
            output: outputs.hh,
            width: high_width,
            height: high_height,
            cb_width,
            cb_height,
            inv_delta: params.inv_delta_hh,
            items,
        })?;
        Ok(())
    }

    pub(super) fn launch_transcode_dwt97_quantize_codeblocks(
        &self,
        request: &Dwt97QuantizeCodeblocksLaunch<'_>,
    ) -> Result<(), CudaError> {
        if request.width <= 0 || request.height <= 0 {
            return Ok(());
        }
        let function =
            self.transcode_kernel_function(CudaKernel::TranscodeDwt97QuantizeCodeblocks)?;
        let mut band_ptr = request.band.device_ptr();
        let mut output_ptr = request.output.device_ptr();
        let mut width = request.width;
        let mut height = request.height;
        let mut cb_width = request.cb_width;
        let mut cb_height = request.cb_height;
        let mut inv_delta = request.inv_delta;
        let mut params = cuda_kernel_params!(
            band_ptr, output_ptr, width, height, cb_width, cb_height, inv_delta
        );
        let grid_w = u32::try_from(width).map_err(|_| CudaError::LengthTooLarge { len: 0 })?;
        let grid_h = u32::try_from(height).map_err(|_| CudaError::LengthTooLarge { len: 0 })?;
        let base = j2k_dwt53_launch_geometry(grid_w, grid_h)
            .ok_or(CudaError::LengthTooLarge { len: 0 })?;
        let geometry = with_grid_z(base, request.items);
        self.launch_kernel_async(function, geometry, &mut params)
    }
}
