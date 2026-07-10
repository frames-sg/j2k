// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "cuda-oxide-jpeg-decode")]
use super::{
    decode::{CudaJpegDecodeHuffmanLaunch, CudaJpegDecodeHuffmanPtrs},
    jpeg_entropy_overflow_count, validate_jpeg_entropy_chunk_plan, CudaJpegEntropyChunkParams,
    CudaJpegEntropyOverflowState, CudaJpegEntropySyncState,
};
use super::{CudaJpegChunkedEntropyPlan, CudaJpegChunkedEntropyReport};
#[cfg(feature = "cuda-oxide-jpeg-decode")]
use crate::{
    bytes::{
        cuda_jpeg_entropy_overflow_states_as_bytes, cuda_jpeg_entropy_overflow_states_as_bytes_mut,
        cuda_jpeg_entropy_sync_states_as_bytes, cuda_jpeg_entropy_sync_states_as_bytes_mut,
        cuda_jpeg_huffman_table_as_bytes,
    },
    execution::cuda_kernel_param,
    kernels::{CudaKernel, CudaLaunchGeometry},
    memory::CudaDeviceBuffer,
};
use crate::{context::CudaContext, error::CudaError, execution::CudaExecutionStats};

#[cfg(feature = "cuda-oxide-jpeg-decode")]
#[derive(Clone, Copy)]
struct CudaJpegEntropySync420Launch<'a> {
    entropy: &'a CudaDeviceBuffer,
    params: CudaJpegEntropyChunkParams,
    huffman: CudaJpegDecodeHuffmanLaunch<'a>,
    states: &'a CudaDeviceBuffer,
}

#[cfg(feature = "cuda-oxide-jpeg-decode")]
#[derive(Clone, Copy)]
struct CudaJpegEntropyOverflow420Launch<'a> {
    entropy: &'a CudaDeviceBuffer,
    params: CudaJpegEntropyChunkParams,
    huffman: CudaJpegDecodeHuffmanLaunch<'a>,
    states: &'a CudaDeviceBuffer,
    overflows: &'a CudaDeviceBuffer,
}

impl CudaContext {
    #[doc(hidden)]
    /// Run experimental 4:2:0 JPEG entropy self-sync diagnostics.
    pub fn diagnose_jpeg_420_entropy_self_sync(
        &self,
        plan: &CudaJpegChunkedEntropyPlan<'_>,
    ) -> Result<CudaJpegChunkedEntropyReport, CudaError> {
        plan.config.validate()?;
        let subsequences = plan
            .config
            .subsequence_count_for_entropy_bytes(plan.entropy_bytes.len())?;
        if subsequences == 0 {
            return Ok(CudaJpegChunkedEntropyReport {
                config: plan.config,
                entropy_bytes: plan.entropy_bytes.len(),
                states: Vec::new(),
                overflows: Vec::new(),
                execution: CudaExecutionStats {
                    kernel_dispatches: 0,
                    copy_kernel_dispatches: 0,
                    decode_kernel_dispatches: 0,
                    hardware_decode: false,
                },
            });
        }

        #[cfg(not(feature = "cuda-oxide-jpeg-decode"))]
        {
            let _ = subsequences;
            Err(CudaError::InvalidArgument {
                message: "CUDA JPEG entropy diagnostic PTX was not built".to_string(),
            })
        }

        #[cfg(feature = "cuda-oxide-jpeg-decode")]
        {
            self.diagnose_jpeg_420_entropy_self_sync_nonempty(plan, subsequences)
        }
    }
    #[cfg(feature = "cuda-oxide-jpeg-decode")]
    #[expect(
        clippy::similar_names,
        reason = "Y/Cb/Cr DC/AC names mirror the six distinct JPEG diagnostic table roles"
    )]
    fn diagnose_jpeg_420_entropy_self_sync_nonempty(
        &self,
        plan: &CudaJpegChunkedEntropyPlan<'_>,
        subsequences: usize,
    ) -> Result<CudaJpegChunkedEntropyReport, CudaError> {
        let params = validate_jpeg_entropy_chunk_plan(plan, subsequences)?;
        self.inner.set_current()?;
        let entropy = self.upload_pinned(plan.entropy_bytes)?;
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

        let mut states = vec![CudaJpegEntropySyncState::default(); subsequences];
        let states_buffer = self.upload(cuda_jpeg_entropy_sync_states_as_bytes(&states))?;
        self.launch_jpeg_entropy_sync420(CudaJpegEntropySync420Launch {
            entropy: &entropy,
            params,
            huffman,
            states: &states_buffer,
        })?;
        states_buffer.copy_to_host(cuda_jpeg_entropy_sync_states_as_bytes_mut(&mut states))?;

        let mut overflows = vec![
            CudaJpegEntropyOverflowState::default();
            jpeg_entropy_overflow_count(subsequences)
        ];
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
    #[cfg(feature = "cuda-oxide-jpeg-decode")]
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
        let function = self.jpeg_entropy_kernel_function(CudaKernel::JpegEntropySync420)?;
        let mut params = params;
        let mut entropy_ptr = entropy.device_ptr();
        let mut huffman_ptrs = CudaJpegDecodeHuffmanPtrs {
            y_dc: huffman.y_dc.device_ptr(),
            y_ac: huffman.y_ac.device_ptr(),
            cb_dc: huffman.cb_dc.device_ptr(),
            cb_ac: huffman.cb_ac.device_ptr(),
            cr_dc: huffman.cr_dc.device_ptr(),
            cr_ac: huffman.cr_ac.device_ptr(),
        };
        let mut states_ptr = states.device_ptr();
        let mut kernel_params = cuda_kernel_params!(entropy_ptr, params, huffman_ptrs, states_ptr);
        let geometry = CudaLaunchGeometry {
            grid: (params.subsequence_count.div_ceil(128), 1, 1),
            block: (128, 1, 1),
        };

        self.launch_kernel(function, geometry, &mut kernel_params)
    }

    #[cfg(feature = "cuda-oxide-jpeg-decode")]
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
        let function = self.jpeg_entropy_kernel_function(CudaKernel::JpegEntropyOverflow420)?;
        let mut params = params;
        let mut entropy_ptr = entropy.device_ptr();
        let mut huffman_ptrs = CudaJpegDecodeHuffmanPtrs {
            y_dc: huffman.y_dc.device_ptr(),
            y_ac: huffman.y_ac.device_ptr(),
            cb_dc: huffman.cb_dc.device_ptr(),
            cb_ac: huffman.cb_ac.device_ptr(),
            cr_dc: huffman.cr_dc.device_ptr(),
            cr_ac: huffman.cr_ac.device_ptr(),
        };
        let mut states_ptr = states.device_ptr();
        let mut overflows_ptr = overflows.device_ptr();
        let mut kernel_params =
            cuda_kernel_params!(entropy_ptr, params, huffman_ptrs, states_ptr, overflows_ptr);
        let geometry = CudaLaunchGeometry {
            grid: (
                (params.subsequence_count.saturating_sub(1)).div_ceil(128),
                1,
                1,
            ),
            block: (128, 1, 1),
        };

        self.launch_kernel(function, geometry, &mut kernel_params)
    }

    #[cfg(feature = "cuda-oxide-jpeg-decode")]
    fn jpeg_entropy_kernel_function(
        &self,
        kernel: CudaKernel,
    ) -> Result<crate::driver::CuFunction, CudaError> {
        self.inner.cuda_oxide_jpeg_decode_kernel_function(kernel)
    }
}
