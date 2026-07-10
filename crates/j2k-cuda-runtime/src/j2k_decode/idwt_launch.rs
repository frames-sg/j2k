// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    context::CudaContext,
    driver::CuDevicePtr,
    error::CudaError,
    execution::{cuda_kernel_param, CudaLaunchMode},
    kernels::{
        j2k_dwt53_launch_geometry, j2k_forward_rct_launch_geometry,
        j2k_idwt_multi_1d_launch_geometry, j2k_idwt_multi_coop_axis_launch_geometry,
        j2k_idwt_multi_coop_columns_launch_geometry, j2k_idwt_multi_coop_launch_geometry,
        CudaKernel, CudaLaunchGeometry,
    },
    memory::CudaDeviceBuffer,
};

impl CudaContext {
    pub(in crate::j2k_decode) fn launch_j2k_idwt_interleave(
        &self,
        bands: [&CudaDeviceBuffer; 4],
        output: &CudaDeviceBuffer,
        job: &CudaDeviceBuffer,
        width: u32,
        height: u32,
        mode: CudaLaunchMode,
    ) -> Result<(), CudaError> {
        let function = self.j2k_idwt_kernel_function(CudaKernel::J2kIdwtInterleave)?;
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

    pub(in crate::j2k_decode) fn launch_j2k_idwt_interleave_horizontal_multi(
        &self,
        jobs: &CudaDeviceBuffer,
        max_rows: usize,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        self.launch_j2k_idwt_interleave_horizontal_multi_ptr(
            jobs.device_ptr(),
            max_rows,
            job_count,
            synchronize,
        )
    }

    pub(in crate::j2k_decode) fn launch_j2k_idwt_interleave_horizontal_multi_ptr(
        &self,
        jobs_ptr: CuDevicePtr,
        max_rows: usize,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        let function =
            self.j2k_idwt_kernel_function(CudaKernel::J2kIdwtInterleaveHorizontalMulti)?;
        let mut jobs_ptr = jobs_ptr;
        let mut params = cuda_kernel_params!(jobs_ptr);
        let geometry = j2k_idwt_multi_1d_launch_geometry(max_rows, job_count)
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;
        if synchronize {
            self.launch_kernel(function, geometry, &mut params)
        } else {
            self.launch_kernel_async(function, geometry, &mut params)
        }
    }

    pub(in crate::j2k_decode) fn launch_j2k_idwt_interleave_horizontal_53_multi(
        &self,
        jobs: &CudaDeviceBuffer,
        max_rows: usize,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        self.launch_j2k_idwt_interleave_horizontal_53_multi_ptr(
            jobs.device_ptr(),
            max_rows,
            job_count,
            synchronize,
        )
    }

    pub(in crate::j2k_decode) fn launch_j2k_idwt_interleave_horizontal_53_multi_ptr(
        &self,
        jobs_ptr: CuDevicePtr,
        max_rows: usize,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        let mut jobs_ptr = jobs_ptr;
        let mut params = cuda_kernel_params!(jobs_ptr);
        let geometry = j2k_idwt_multi_coop_launch_geometry(max_rows, job_count)
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
        let function = self.j2k_idwt_kernel_function(kernel)?;
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
        let function = self.j2k_idwt_kernel_function(kernel)?;
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

    pub(in crate::j2k_decode) fn launch_j2k_idwt_vertical_multi(
        &self,
        jobs: &CudaDeviceBuffer,
        max_columns: usize,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        self.launch_j2k_idwt_vertical_multi_ptr(
            jobs.device_ptr(),
            max_columns,
            job_count,
            synchronize,
        )
    }

    pub(in crate::j2k_decode) fn launch_j2k_idwt_vertical_multi_ptr(
        &self,
        jobs_ptr: CuDevicePtr,
        max_columns: usize,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        let function = self.j2k_idwt_kernel_function(CudaKernel::J2kIdwtVerticalMulti)?;
        let mut jobs_ptr = jobs_ptr;
        let mut params = cuda_kernel_params!(jobs_ptr);
        let geometry = j2k_idwt_multi_1d_launch_geometry(max_columns, job_count)
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;
        if synchronize {
            self.launch_kernel(function, geometry, &mut params)
        } else {
            self.launch_kernel_async(function, geometry, &mut params)
        }
    }

    pub(in crate::j2k_decode) fn launch_j2k_idwt_vertical_53_multi(
        &self,
        jobs: &CudaDeviceBuffer,
        max_columns: usize,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        self.launch_j2k_idwt_vertical_53_multi_ptr(
            jobs.device_ptr(),
            max_columns,
            job_count,
            synchronize,
        )
    }

    pub(in crate::j2k_decode) fn launch_j2k_idwt_vertical_53_multi_ptr(
        &self,
        jobs_ptr: CuDevicePtr,
        max_columns: usize,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        let mut jobs_ptr = jobs_ptr;
        let mut params = cuda_kernel_params!(jobs_ptr);
        let geometry = j2k_idwt_multi_coop_launch_geometry(max_columns, job_count)
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
        const COLUMNS_PER_BLOCK: usize = 4;
        const MIN_COLS4_JOBS: usize = 64;
        let (kernel, geometry) = if job_count >= MIN_COLS4_JOBS && max_height <= 256 {
            let geometry = j2k_idwt_multi_coop_columns_launch_geometry(
                max_columns,
                max_height,
                job_count,
                COLUMNS_PER_BLOCK,
            )
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;
            (CudaKernel::J2kIdwtVertical97MultiCols4, geometry)
        } else {
            let geometry =
                j2k_idwt_multi_coop_axis_launch_geometry(max_columns, max_height, job_count)
                    .ok_or(CudaError::LengthTooLarge { len: job_count })?;
            (CudaKernel::J2kIdwtVertical97Multi, geometry)
        };
        let mut jobs_ptr = jobs_ptr;
        let mut params = cuda_kernel_params!(jobs_ptr);
        self.launch_j2k_idwt_named_kernel(kernel, geometry, &mut params, synchronize)
    }

    fn j2k_idwt_kernel_function(
        &self,
        kernel: CudaKernel,
    ) -> Result<crate::driver::CuFunction, CudaError> {
        self.inner.cuda_oxide_j2k_idwt_kernel_function(kernel)
    }

    fn launch_j2k_idwt_named_kernel<const N: usize>(
        &self,
        kernel: CudaKernel,
        geometry: CudaLaunchGeometry,
        params: &mut [*mut std::ffi::c_void; N],
        synchronize: bool,
    ) -> Result<(), CudaError> {
        let function = self.j2k_idwt_kernel_function(kernel)?;
        if synchronize {
            self.launch_kernel(function, geometry, params)
        } else {
            self.launch_kernel_async(function, geometry, params)
        }
    }
}
