// SPDX-License-Identifier: MIT OR Apache-2.0

//! Non-empty CUDA JPEG entropy diagnostic execution.

use super::{
    decode_launch::{CudaJpegDecodeHuffmanLaunch, CudaJpegDecodeHuffmanPtrs},
    diagnostics_allocation::allocate_diagnostic_workspaces_with_cap,
    jpeg_entropy_overflow_count, validate_jpeg_entropy_chunk_plan, CudaJpegChunkedEntropyPlan,
    CudaJpegChunkedEntropyReport, CudaJpegEntropyChunkParams,
};
use crate::{
    bytes::{
        cuda_jpeg_entropy_overflow_states_as_bytes, cuda_jpeg_entropy_overflow_states_as_bytes_mut,
        cuda_jpeg_entropy_sync_states_as_bytes, cuda_jpeg_entropy_sync_states_as_bytes_mut,
        cuda_jpeg_huffman_table_as_bytes,
    },
    context::CudaContext,
    error::{select_resource_release_error, CudaError},
    execution::{cuda_kernel_param, CudaExecutionStats},
    kernels::{CudaKernel, CudaLaunchGeometry},
    memory::{CudaDeviceBuffer, CudaPinnedUploadOperationGuard, CudaPinnedUploadStagingCheckout},
};

#[derive(Clone, Copy)]
struct CudaJpegEntropySync420Launch<'a> {
    entropy: &'a CudaDeviceBuffer,
    params: CudaJpegEntropyChunkParams,
    huffman: CudaJpegDecodeHuffmanLaunch<'a>,
    states: &'a CudaDeviceBuffer,
}

#[derive(Clone, Copy)]
struct CudaJpegEntropyOverflow420Launch<'a> {
    entropy: &'a CudaDeviceBuffer,
    params: CudaJpegEntropyChunkParams,
    huffman: CudaJpegDecodeHuffmanLaunch<'a>,
    states: &'a CudaDeviceBuffer,
    overflows: &'a CudaDeviceBuffer,
}

impl CudaContext {
    #[expect(
        clippy::similar_names,
        reason = "Y/Cb/Cr DC/AC names mirror the six distinct JPEG diagnostic table roles"
    )]
    pub(super) fn diagnose_jpeg_420_entropy_self_sync_nonempty(
        &self,
        plan: &CudaJpegChunkedEntropyPlan<'_>,
        subsequences: usize,
        external_live_bytes: usize,
        pinned_upload: &CudaPinnedUploadOperationGuard<'_>,
    ) -> Result<CudaJpegChunkedEntropyReport, CudaError> {
        let params = validate_jpeg_entropy_chunk_plan(plan, subsequences)?;
        let checkout = pinned_upload.prepare_upload(plan.entropy_bytes.len())?;
        let retained_page_locked_bytes = match checkout.retained_page_locked_bytes() {
            Ok(bytes) => bytes,
            Err(error) => return Err(recycle_checkout_after_error(checkout, error)),
        };
        let workspaces = allocate_diagnostic_workspaces_with_cap(
            subsequences,
            jpeg_entropy_overflow_count(subsequences),
            external_live_bytes,
            retained_page_locked_bytes,
            j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        );
        let (mut states, mut overflows) = match workspaces {
            Ok(workspaces) => workspaces,
            Err(error) => return Err(recycle_checkout_after_error(checkout, error)),
        };
        if let Err(error) = self.inner.set_current() {
            return Err(recycle_checkout_after_error(checkout, error));
        }
        let entropy = checkout.upload(plan.entropy_bytes)?;
        let y_dc = self.upload(cuda_jpeg_huffman_table_as_bytes(&plan.y_dc_table))?;
        let y_ac = self.upload(cuda_jpeg_huffman_table_as_bytes(&plan.y_ac_table))?;
        let cb_dc = self.upload(cuda_jpeg_huffman_table_as_bytes(&plan.cb_dc_table))?;
        let cb_ac = self.upload(cuda_jpeg_huffman_table_as_bytes(&plan.cb_ac_table))?;
        let cr_dc = self.upload(cuda_jpeg_huffman_table_as_bytes(&plan.cr_dc_table))?;
        let cr_ac = self.upload(cuda_jpeg_huffman_table_as_bytes(&plan.cr_ac_table))?;
        let huffman = CudaJpegDecodeHuffmanLaunch {
            y_dc: &y_dc,
            y_ac: &y_ac,
            cb_dc: &cb_dc,
            cb_ac: &cb_ac,
            cr_dc: &cr_dc,
            cr_ac: &cr_ac,
        };

        let states_buffer = self.upload(cuda_jpeg_entropy_sync_states_as_bytes(&states))?;
        self.launch_jpeg_entropy_sync420(CudaJpegEntropySync420Launch {
            entropy: &entropy,
            params,
            huffman,
            states: &states_buffer,
        })?;
        states_buffer.copy_to_host(cuda_jpeg_entropy_sync_states_as_bytes_mut(&mut states))?;

        if !overflows.is_empty() {
            let overflow_buffer =
                self.upload(cuda_jpeg_entropy_overflow_states_as_bytes(&overflows))?;
            self.launch_jpeg_entropy_overflow420(CudaJpegEntropyOverflow420Launch {
                entropy: &entropy,
                params,
                huffman,
                states: &states_buffer,
                overflows: &overflow_buffer,
            })?;
            overflow_buffer.copy_to_host(cuda_jpeg_entropy_overflow_states_as_bytes_mut(
                &mut overflows,
            ))?;
        }

        Ok(CudaJpegChunkedEntropyReport {
            config: plan.config,
            entropy_bytes: plan.entropy_bytes.len(),
            states,
            overflows,
            execution: CudaExecutionStats {
                kernel_dispatches: 1 + usize::from(subsequences > 1),
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 0,
                hardware_decode: false,
            },
        })
    }

    fn launch_jpeg_entropy_sync420(
        &self,
        launch: CudaJpegEntropySync420Launch<'_>,
    ) -> Result<(), CudaError> {
        let CudaJpegEntropySync420Launch {
            entropy,
            params,
            huffman,
            states,
        } = launch;
        let geometry =
            CudaLaunchGeometry::new((params.subsequence_count.div_ceil(128), 1, 1), (128, 1, 1))
                .ok_or(CudaError::InvalidArgument {
                    message: "JPEG entropy sync launch exceeds static CUDA limits".to_string(),
                })?;
        let function = self.jpeg_entropy_kernel_function(CudaKernel::JpegEntropySync420)?;
        let mut params = params;
        let mut entropy_ptr = entropy.device_ptr();
        let mut huffman_ptrs = huffman_ptrs(huffman);
        let mut states_ptr = states.device_ptr();
        let mut kernel_params = cuda_kernel_params!(entropy_ptr, params, huffman_ptrs, states_ptr);
        self.launch_kernel(function, geometry, &mut kernel_params)
    }

    fn launch_jpeg_entropy_overflow420(
        &self,
        launch: CudaJpegEntropyOverflow420Launch<'_>,
    ) -> Result<(), CudaError> {
        let CudaJpegEntropyOverflow420Launch {
            entropy,
            params,
            huffman,
            states,
            overflows,
        } = launch;
        let geometry = CudaLaunchGeometry::new(
            (
                params.subsequence_count.saturating_sub(1).div_ceil(128),
                1,
                1,
            ),
            (128, 1, 1),
        )
        .ok_or(CudaError::InvalidArgument {
            message: "JPEG entropy overflow launch exceeds static CUDA limits".to_string(),
        })?;
        let function = self.jpeg_entropy_kernel_function(CudaKernel::JpegEntropyOverflow420)?;
        let mut params = params;
        let mut entropy_ptr = entropy.device_ptr();
        let mut huffman_ptrs = huffman_ptrs(huffman);
        let mut states_ptr = states.device_ptr();
        let mut overflows_ptr = overflows.device_ptr();
        let mut kernel_params =
            cuda_kernel_params!(entropy_ptr, params, huffman_ptrs, states_ptr, overflows_ptr);
        self.launch_kernel(function, geometry, &mut kernel_params)
    }

    fn jpeg_entropy_kernel_function(
        &self,
        kernel: CudaKernel,
    ) -> Result<crate::driver::CuFunction, CudaError> {
        self.inner.cuda_oxide_jpeg_decode_kernel_function(kernel)
    }
}

fn huffman_ptrs(huffman: CudaJpegDecodeHuffmanLaunch<'_>) -> CudaJpegDecodeHuffmanPtrs {
    CudaJpegDecodeHuffmanPtrs {
        y_dc: huffman.y_dc.device_ptr(),
        y_ac: huffman.y_ac.device_ptr(),
        cb_dc: huffman.cb_dc.device_ptr(),
        cb_ac: huffman.cb_ac.device_ptr(),
        cr_dc: huffman.cr_dc.device_ptr(),
        cr_ac: huffman.cr_ac.device_ptr(),
    }
}

fn recycle_checkout_after_error(
    checkout: CudaPinnedUploadStagingCheckout<'_, '_>,
    primary_error: CudaError,
) -> CudaError {
    match checkout.recycle() {
        Ok(()) => primary_error,
        Err(release_error) => select_resource_release_error(primary_error, release_error),
    }
}
