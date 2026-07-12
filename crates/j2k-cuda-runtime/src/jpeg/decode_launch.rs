// SPDX-License-Identifier: MIT OR Apache-2.0

//! Validated CUDA JPEG decode resource upload and kernel launch.

use super::{
    jpeg_rgb8_kernel, CudaJpeg420Params, CudaJpegDecodeStatus, CudaJpegRgb8DecodePlan,
    CudaJpegRgb8ValidatedPlan,
};
use crate::{
    bytes::{
        cuda_jpeg_decode_statuses_as_bytes, cuda_jpeg_decode_statuses_as_bytes_mut,
        cuda_jpeg_entropy_checkpoints_as_bytes, cuda_jpeg_huffman_table_as_bytes,
        u16_slice_as_bytes,
    },
    context::CudaContext,
    error::CudaError,
    execution::{cuda_kernel_param, CudaExecutionStats},
    kernels::{CudaKernel, CudaLaunchGeometry},
    memory::CudaDeviceBuffer,
};

#[derive(Clone, Copy)]
struct CudaJpegDecodeQuantLaunch<'a> {
    y: &'a CudaDeviceBuffer,
    cb: &'a CudaDeviceBuffer,
    cr: &'a CudaDeviceBuffer,
}

#[derive(Clone, Copy)]
pub(super) struct CudaJpegDecodeHuffmanLaunch<'a> {
    pub(super) y_dc: &'a CudaDeviceBuffer,
    pub(super) y_ac: &'a CudaDeviceBuffer,
    pub(super) cb_dc: &'a CudaDeviceBuffer,
    pub(super) cb_ac: &'a CudaDeviceBuffer,
    pub(super) cr_dc: &'a CudaDeviceBuffer,
    pub(super) cr_ac: &'a CudaDeviceBuffer,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CudaJpegDecodeQuantPtrs {
    y: crate::driver::CuDevicePtr,
    cb: crate::driver::CuDevicePtr,
    cr: crate::driver::CuDevicePtr,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CudaJpegDecodeHuffmanPtrs {
    pub(super) y_dc: crate::driver::CuDevicePtr,
    pub(super) y_ac: crate::driver::CuDevicePtr,
    pub(super) cb_dc: crate::driver::CuDevicePtr,
    pub(super) cb_ac: crate::driver::CuDevicePtr,
    pub(super) cr_dc: crate::driver::CuDevicePtr,
    pub(super) cr_ac: crate::driver::CuDevicePtr,
}

// SAFETY: these `#[repr(C)]` structs contain only CUDA device-pointer scalar
// values and mirror the pointer-only structs consumed by the CUDA Oxide kernels.
unsafe impl crate::execution::CudaKernelParam for CudaJpegDecodeQuantPtrs {}

// SAFETY: these `#[repr(C)]` structs contain only CUDA device-pointer scalar
// values and mirror the pointer-only structs consumed by the CUDA Oxide kernels.
unsafe impl crate::execution::CudaKernelParam for CudaJpegDecodeHuffmanPtrs {}

#[derive(Clone, Copy)]
struct CudaJpegDecodeRgb8Launch<'a> {
    kernel: CudaKernel,
    geometry: CudaLaunchGeometry,
    entropy: &'a CudaDeviceBuffer,
    output: &'a CudaDeviceBuffer,
    params: CudaJpeg420Params,
    quant: CudaJpegDecodeQuantLaunch<'a>,
    huffman: CudaJpegDecodeHuffmanLaunch<'a>,
    checkpoints: &'a CudaDeviceBuffer,
    status: &'a CudaDeviceBuffer,
}

impl CudaContext {
    #[expect(
        clippy::similar_names,
        reason = "Y/Cb/Cr DC/AC names mirror the six distinct JPEG decode table roles"
    )]
    pub(super) fn decode_jpeg_rgb8_owned_validated(
        &self,
        plan: &CudaJpegRgb8DecodePlan<'_>,
        output: &CudaDeviceBuffer,
        validated: CudaJpegRgb8ValidatedPlan,
        mut statuses: Vec<CudaJpegDecodeStatus>,
    ) -> Result<CudaExecutionStats, CudaError> {
        // Both safe decode entrypoints converge here. Initialize the entire
        // validated extent, including pitched row padding, before any kernel
        // can produce a successful output.
        self.memset_d8(output, 0, validated.output_len)?;
        let (kernel, kernel_name) = jpeg_rgb8_kernel(plan.sampling);
        let entropy = self.upload(plan.entropy_bytes)?;
        let y_quant = self.upload(u16_slice_as_bytes(&plan.y_quant))?;
        let cb_quant = self.upload(u16_slice_as_bytes(&plan.cb_quant))?;
        let cr_quant = self.upload(u16_slice_as_bytes(&plan.cr_quant))?;
        let y_dc = self.upload(cuda_jpeg_huffman_table_as_bytes(&plan.y_dc_table))?;
        let y_ac = self.upload(cuda_jpeg_huffman_table_as_bytes(&plan.y_ac_table))?;
        let cb_dc = self.upload(cuda_jpeg_huffman_table_as_bytes(&plan.cb_dc_table))?;
        let cb_ac = self.upload(cuda_jpeg_huffman_table_as_bytes(&plan.cb_ac_table))?;
        let cr_dc = self.upload(cuda_jpeg_huffman_table_as_bytes(&plan.cr_dc_table))?;
        let cr_ac = self.upload(cuda_jpeg_huffman_table_as_bytes(&plan.cr_ac_table))?;
        let checkpoints = self.upload(cuda_jpeg_entropy_checkpoints_as_bytes(
            plan.entropy_checkpoints,
        ))?;
        let status_buffer = self.upload(cuda_jpeg_decode_statuses_as_bytes(&statuses))?;
        let quant = CudaJpegDecodeQuantLaunch {
            y: &y_quant,
            cb: &cb_quant,
            cr: &cr_quant,
        };
        let huffman = CudaJpegDecodeHuffmanLaunch {
            y_dc: &y_dc,
            y_ac: &y_ac,
            cb_dc: &cb_dc,
            cb_ac: &cb_ac,
            cr_dc: &cr_dc,
            cr_ac: &cr_ac,
        };
        self.launch_jpeg_decode_rgb8(CudaJpegDecodeRgb8Launch {
            kernel,
            geometry: validated.geometry,
            entropy: &entropy,
            output,
            params: validated.params,
            quant,
            huffman,
            checkpoints: &checkpoints,
            status: &status_buffer,
        })?;
        status_buffer.copy_to_host(cuda_jpeg_decode_statuses_as_bytes_mut(&mut statuses))?;
        for status in statuses {
            if status.code != 0 {
                return Err(CudaError::KernelStatus {
                    kernel: kernel_name,
                    code: status.code,
                    detail: status.detail,
                });
            }
        }
        Ok(CudaExecutionStats {
            kernel_dispatches: 1,
            copy_kernel_dispatches: 0,
            decode_kernel_dispatches: 1,
            hardware_decode: false,
        })
    }

    fn launch_jpeg_decode_rgb8(
        &self,
        launch: CudaJpegDecodeRgb8Launch<'_>,
    ) -> Result<(), CudaError> {
        let CudaJpegDecodeRgb8Launch {
            kernel,
            geometry,
            entropy,
            output,
            params,
            quant,
            huffman,
            checkpoints,
            status,
        } = launch;
        let function = self.jpeg_rgb8_kernel_function(kernel)?;
        let mut params = params;
        let mut entropy_ptr = entropy.device_ptr();
        let mut output_ptr = output.device_ptr();
        let mut quant_ptrs = CudaJpegDecodeQuantPtrs {
            y: quant.y.device_ptr(),
            cb: quant.cb.device_ptr(),
            cr: quant.cr.device_ptr(),
        };
        let mut huffman_ptrs = CudaJpegDecodeHuffmanPtrs {
            y_dc: huffman.y_dc.device_ptr(),
            y_ac: huffman.y_ac.device_ptr(),
            cb_dc: huffman.cb_dc.device_ptr(),
            cb_ac: huffman.cb_ac.device_ptr(),
            cr_dc: huffman.cr_dc.device_ptr(),
            cr_ac: huffman.cr_ac.device_ptr(),
        };
        let mut checkpoints_ptr = checkpoints.device_ptr();
        let mut status_ptr = status.device_ptr();
        let mut kernel_params = cuda_kernel_params!(
            entropy_ptr,
            output_ptr,
            params,
            quant_ptrs,
            huffman_ptrs,
            checkpoints_ptr,
            status_ptr
        );
        self.launch_kernel(function, geometry, &mut kernel_params)
    }

    fn jpeg_rgb8_kernel_function(
        &self,
        kernel: CudaKernel,
    ) -> Result<crate::driver::CuFunction, CudaError> {
        self.inner.cuda_oxide_jpeg_decode_kernel_function(kernel)
    }
}
