// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    driver::CuDevicePtr,
    error::CudaError,
    execution::{cuda_kernel_param, CudaLaunchMode},
    kernels::{
        j2k_dwt53_launch_geometry, j2k_forward_rct_launch_geometry,
        j2k_idwt_multi_1d_launch_geometry, j2k_idwt_multi_coop_axis_launch_geometry,
        j2k_idwt_tiled_horizontal_launch_geometry, CudaKernel, CudaLaunchGeometry,
    },
    memory::CudaDeviceBuffer,
};

use super::idwt::launch_geometry::{
    idwt_vertical_97_multi_launch_geometry, idwt_vertical_strip_launch_geometry,
};
mod batch;
mod profiling;

/// Widths from which the generic horizontal pass uses the tiled kernel.
const TILED_HORIZONTAL_MIN_WIDTH: usize = 128;
/// Heights the generic vertical pass handles with the strip kernels.
const VERTICAL_STRIP_HEIGHTS: core::ops::RangeInclusive<usize> = 128..=1024;
/// Heights above this use the tall strip kernel.
const TALL_STRIP_MIN_EXCLUSIVE_HEIGHT: usize = 512;

/// Generic-mode horizontal kernel and geometry. Launch validation and launch
/// both use it, so validation checks exactly the geometry that launches.
pub(in crate::j2k_decode) fn generic_horizontal_idwt_route(
    max_width: usize,
    max_rows: usize,
    tiled_horizontal_blocks: usize,
    job_count: usize,
) -> (CudaKernel, Option<CudaLaunchGeometry>) {
    if max_width >= TILED_HORIZONTAL_MIN_WIDTH {
        (
            CudaKernel::J2kIdwtInterleaveHorizontalTiledMulti,
            j2k_idwt_tiled_horizontal_launch_geometry(tiled_horizontal_blocks, job_count),
        )
    } else {
        (
            CudaKernel::J2kIdwtInterleaveHorizontalMulti,
            j2k_idwt_multi_1d_launch_geometry(max_rows, job_count),
        )
    }
}

/// Generic-mode vertical kernel and geometry, shared by validation and launch.
pub(in crate::j2k_decode) fn generic_vertical_idwt_route(
    max_columns: usize,
    max_height: usize,
    job_count: usize,
) -> (CudaKernel, Option<CudaLaunchGeometry>) {
    if VERTICAL_STRIP_HEIGHTS.contains(&max_height) {
        let kernel = if max_height > TALL_STRIP_MIN_EXCLUSIVE_HEIGHT {
            CudaKernel::J2kIdwtVerticalTallStripMulti
        } else {
            CudaKernel::J2kIdwtVerticalStripMulti
        };
        (
            kernel,
            idwt_vertical_strip_launch_geometry(max_columns, max_height, job_count),
        )
    } else {
        (
            CudaKernel::J2kIdwtVerticalMulti,
            j2k_idwt_multi_1d_launch_geometry(max_columns, job_count),
        )
    }
}

impl crate::J2kCudaEngine<'_> {
    pub(in crate::j2k_decode) fn launch_j2k_idwt_interleave(
        &self,
        bands: [&CudaDeviceBuffer; 4],
        output: &CudaDeviceBuffer,
        job: &CudaDeviceBuffer,
        width: u32,
        height: u32,
        mode: CudaLaunchMode,
    ) -> Result<(), CudaError> {
        let function = Self::j2k_idwt_kernel_function(CudaKernel::J2kIdwtInterleave)?;
        let [ll, hl, lh, hh] = bands;
        let mut low_low_ptr = ll.device_ptr();
        let mut high_low_ptr = hl.device_ptr();
        let mut low_high_ptr = lh.device_ptr();
        let mut high_high_ptr = hh.device_ptr();
        let mut output_ptr = output.device_ptr();
        let mut job_ptr = job.device_ptr();
        let mut params = cuda_kernel_params!(
            low_low_ptr,
            high_low_ptr,
            low_high_ptr,
            high_high_ptr,
            output_ptr,
            job_ptr
        );
        let geometry =
            j2k_dwt53_launch_geometry(width, height).ok_or(CudaError::ImageTooLarge {
                width,
                height,
                channels: 1,
            })?;
        match mode {
            CudaLaunchMode::Sync => self.launch_kernel(function, geometry, &mut params),
            CudaLaunchMode::Async => self.launch_kernel_async(function, geometry, &mut params),
        }
    }

    pub(in crate::j2k_decode) fn launch_j2k_idwt_interleave_horizontal_multi_ptr(
        &self,
        jobs_ptr: CuDevicePtr,
        max_width: usize,
        max_rows: usize,
        tiled_horizontal_blocks: usize,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        let (kernel, geometry) =
            generic_horizontal_idwt_route(max_width, max_rows, tiled_horizontal_blocks, job_count);
        let function = Self::j2k_idwt_kernel_function(kernel)?;
        let mut jobs_ptr = jobs_ptr;
        let mut params = cuda_kernel_params!(jobs_ptr);
        let geometry = geometry.ok_or(CudaError::LengthTooLarge { len: job_count })?;
        if synchronize {
            self.launch_kernel(function, geometry, &mut params)
        } else {
            self.launch_kernel_async(function, geometry, &mut params)
        }
    }

    pub(in crate::j2k_decode) fn launch_j2k_idwt_interleave_horizontal_53_multi_ptr(
        &self,
        jobs_ptr: CuDevicePtr,
        max_width: usize,
        max_rows: usize,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        let mut jobs_ptr = jobs_ptr;
        let mut params = cuda_kernel_params!(jobs_ptr);
        let geometry = j2k_idwt_multi_coop_axis_launch_geometry(max_rows, max_width, job_count)
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;
        self.launch_j2k_idwt_named_kernel(
            CudaKernel::J2kIdwtInterleaveHorizontal53Multi,
            geometry,
            &mut params,
            synchronize,
        )
    }

    pub(in crate::j2k_decode) fn launch_j2k_idwt_interleave_horizontal_97_multi_ptr(
        &self,
        jobs_ptr: CuDevicePtr,
        max_width: usize,
        max_rows: usize,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        let mut jobs_ptr = jobs_ptr;
        let mut params = cuda_kernel_params!(jobs_ptr);
        let geometry = j2k_idwt_multi_coop_axis_launch_geometry(max_rows, max_width, job_count)
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;
        self.launch_j2k_idwt_named_kernel(
            CudaKernel::J2kIdwtInterleaveHorizontal97Multi,
            geometry,
            &mut params,
            synchronize,
        )
    }

    pub(in crate::j2k_decode) fn launch_j2k_idwt_horizontal(
        &self,
        kernel: CudaKernel,
        output: &CudaDeviceBuffer,
        job: &CudaDeviceBuffer,
        rows: usize,
        mode: CudaLaunchMode,
    ) -> Result<(), CudaError> {
        let function = Self::j2k_idwt_kernel_function(kernel)?;
        let mut output_ptr = output.device_ptr();
        let mut job_ptr = job.device_ptr();
        let mut params = cuda_kernel_params!(output_ptr, job_ptr);
        let geometry =
            j2k_forward_rct_launch_geometry(rows).ok_or(CudaError::LengthTooLarge { len: rows })?;
        match mode {
            CudaLaunchMode::Sync => self.launch_kernel(function, geometry, &mut params),
            CudaLaunchMode::Async => self.launch_kernel_async(function, geometry, &mut params),
        }
    }

    pub(in crate::j2k_decode) fn launch_j2k_idwt_vertical(
        &self,
        kernel: CudaKernel,
        output: &CudaDeviceBuffer,
        job: &CudaDeviceBuffer,
        columns: usize,
        mode: CudaLaunchMode,
    ) -> Result<(), CudaError> {
        let function = Self::j2k_idwt_kernel_function(kernel)?;
        let mut output_ptr = output.device_ptr();
        let mut job_ptr = job.device_ptr();
        let mut params = cuda_kernel_params!(output_ptr, job_ptr);
        let geometry = j2k_forward_rct_launch_geometry(columns)
            .ok_or(CudaError::LengthTooLarge { len: columns })?;
        match mode {
            CudaLaunchMode::Sync => self.launch_kernel(function, geometry, &mut params),
            CudaLaunchMode::Async => self.launch_kernel_async(function, geometry, &mut params),
        }
    }

    pub(in crate::j2k_decode) fn launch_j2k_idwt_vertical_multi_ptr(
        &self,
        jobs_ptr: CuDevicePtr,
        max_columns: usize,
        max_height: usize,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        let (kernel, geometry) = generic_vertical_idwt_route(max_columns, max_height, job_count);
        let function = Self::j2k_idwt_kernel_function(kernel)?;
        let mut jobs_ptr = jobs_ptr;
        let mut params = cuda_kernel_params!(jobs_ptr);
        let geometry = geometry.ok_or(CudaError::LengthTooLarge { len: job_count })?;
        if synchronize {
            self.launch_kernel(function, geometry, &mut params)
        } else {
            self.launch_kernel_async(function, geometry, &mut params)
        }
    }

    pub(in crate::j2k_decode) fn launch_j2k_idwt_vertical_53_multi_ptr(
        &self,
        jobs_ptr: CuDevicePtr,
        max_columns: usize,
        max_height: usize,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        let mut jobs_ptr = jobs_ptr;
        let mut params = cuda_kernel_params!(jobs_ptr);
        let geometry = idwt_vertical_strip_launch_geometry(max_columns, max_height, job_count)
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;
        self.launch_j2k_idwt_named_kernel(
            CudaKernel::J2kIdwtVertical53Multi,
            geometry,
            &mut params,
            synchronize,
        )
    }

    pub(in crate::j2k_decode) fn launch_j2k_idwt_vertical_97_multi_ptr(
        &self,
        jobs_ptr: CuDevicePtr,
        max_columns: usize,
        max_height: usize,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        let (kernel, geometry) =
            idwt_vertical_97_multi_launch_geometry(max_columns, max_height, job_count)
                .ok_or(CudaError::LengthTooLarge { len: job_count })?;
        let mut jobs_ptr = jobs_ptr;
        let mut params = cuda_kernel_params!(jobs_ptr);
        self.launch_j2k_idwt_named_kernel(kernel, geometry, &mut params, synchronize)
    }

    fn j2k_idwt_kernel_function(
        kernel: CudaKernel,
    ) -> Result<crate::driver::CuFunction, CudaError> {
        kernel.spec()
    }

    fn launch_j2k_idwt_named_kernel<const N: usize>(
        &self,
        kernel: CudaKernel,
        geometry: CudaLaunchGeometry,
        params: &mut [*mut std::ffi::c_void; N],
        synchronize: bool,
    ) -> Result<(), CudaError> {
        let function = Self::j2k_idwt_kernel_function(kernel)?;
        if synchronize {
            self.launch_kernel(function, geometry, params)
        } else {
            self.launch_kernel_async(function, geometry, params)
        }
    }
}
