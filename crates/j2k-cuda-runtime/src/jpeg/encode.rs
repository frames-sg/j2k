// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "cuda-oxide-jpeg-encode")]
use super::{
    types::{
        JPEG_BASELINE_ENCODE_STATUS_INVALID_PARAMS, JPEG_BASELINE_ENCODE_STATUS_MISSING_HUFFMAN,
        JPEG_BASELINE_ENCODE_STATUS_OK, JPEG_BASELINE_ENCODE_STATUS_OVERFLOW,
    },
    CudaJpegBaselineEncodeParams, CudaJpegBaselineEncodeStatus,
};
use super::{CudaJpegBaselineEntropyEncodeBatchJob, CudaJpegBaselineEntropyEncodeJob};
#[cfg(feature = "cuda-oxide-jpeg-encode")]
use crate::{
    bytes::{
        cuda_jpeg_baseline_encode_huffman_table_as_bytes,
        cuda_jpeg_baseline_encode_params_as_bytes, cuda_jpeg_baseline_encode_statuses_as_bytes,
        cuda_jpeg_baseline_encode_statuses_as_bytes_mut,
    },
    execution::cuda_kernel_param,
    kernels::{CudaKernel, CudaLaunchGeometry},
    memory::CudaDeviceBuffer,
};
use crate::{context::CudaContext, error::CudaError};

#[cfg(feature = "cuda-oxide-jpeg-encode")]
struct CudaJpegBaselineQuantLaunch<'a> {
    luma: &'a CudaDeviceBuffer,
    chroma: &'a CudaDeviceBuffer,
}

#[cfg(feature = "cuda-oxide-jpeg-encode")]
struct CudaJpegBaselineHuffmanLaunch<'a> {
    dc_luma: &'a CudaDeviceBuffer,
    ac_luma: &'a CudaDeviceBuffer,
    dc_chroma: &'a CudaDeviceBuffer,
    ac_chroma: &'a CudaDeviceBuffer,
}

#[cfg(feature = "cuda-oxide-jpeg-encode")]
struct CudaJpegBaselineEntropyLaunch<'a> {
    input: &'a CudaDeviceBuffer,
    input_offset: usize,
    entropy: &'a CudaDeviceBuffer,
    status: &'a CudaDeviceBuffer,
    params: CudaJpegBaselineEncodeParams,
    quant: CudaJpegBaselineQuantLaunch<'a>,
    huffman: CudaJpegBaselineHuffmanLaunch<'a>,
}

#[cfg(feature = "cuda-oxide-jpeg-encode")]
struct CudaJpegBaselineEntropyBatchLaunch<'a> {
    input: &'a CudaDeviceBuffer,
    entropy: &'a CudaDeviceBuffer,
    status: &'a CudaDeviceBuffer,
    params: &'a CudaDeviceBuffer,
    quant: CudaJpegBaselineQuantLaunch<'a>,
    huffman: CudaJpegBaselineHuffmanLaunch<'a>,
    tile_count: u32,
}

impl CudaContext {
    /// Encode one CUDA-resident tile into baseline JPEG entropy bytes.
    #[cfg_attr(
        feature = "cuda-oxide-jpeg-encode",
        expect(
            clippy::similar_names,
            reason = "DC/AC luma/chroma names mirror the four distinct JPEG Huffman table roles"
        )
    )]
    #[doc(hidden)]
    pub fn encode_jpeg_baseline_entropy(
        &self,
        job: &CudaJpegBaselineEntropyEncodeJob<'_>,
    ) -> Result<Vec<u8>, CudaError> {
        #[cfg(not(feature = "cuda-oxide-jpeg-encode"))]
        {
            let _ = job;
            Err(CudaError::InvalidArgument {
                message: "CUDA JPEG baseline encode PTX was not built".to_string(),
            })
        }

        #[cfg(feature = "cuda-oxide-jpeg-encode")]
        {
            self.inner.set_current()?;
            let entropy = self.allocate(job.entropy_capacity)?;
            let mut status = [CudaJpegBaselineEncodeStatus::default()];
            let status_buffer =
                self.upload(cuda_jpeg_baseline_encode_statuses_as_bytes(&status))?;
            let q_luma = self.upload(&job.q_luma)?;
            let q_chroma = self.upload(&job.q_chroma)?;
            let huff_dc_luma = self.upload(cuda_jpeg_baseline_encode_huffman_table_as_bytes(
                &job.huff_dc_luma,
            ))?;
            let huff_ac_luma = self.upload(cuda_jpeg_baseline_encode_huffman_table_as_bytes(
                &job.huff_ac_luma,
            ))?;
            let huff_dc_chroma = self.upload(cuda_jpeg_baseline_encode_huffman_table_as_bytes(
                &job.huff_dc_chroma,
            ))?;
            let huff_ac_chroma = self.upload(cuda_jpeg_baseline_encode_huffman_table_as_bytes(
                &job.huff_ac_chroma,
            ))?;
            self.launch_jpeg_encode_baseline_entropy(&CudaJpegBaselineEntropyLaunch {
                input: job.input,
                input_offset: job.input_offset,
                entropy: &entropy,
                status: &status_buffer,
                params: job.params,
                quant: CudaJpegBaselineQuantLaunch {
                    luma: &q_luma,
                    chroma: &q_chroma,
                },
                huffman: CudaJpegBaselineHuffmanLaunch {
                    dc_luma: &huff_dc_luma,
                    ac_luma: &huff_ac_luma,
                    dc_chroma: &huff_dc_chroma,
                    ac_chroma: &huff_ac_chroma,
                },
            })?;
            status_buffer
                .copy_to_host(cuda_jpeg_baseline_encode_statuses_as_bytes_mut(&mut status))?;
            validate_jpeg_encode_status(status[0], "j2k_jpeg_encode_baseline_entropy")?;
            let entropy_len =
                usize::try_from(status[0].entropy_len).map_err(|_| CudaError::LengthTooLarge {
                    len: status[0].entropy_len as usize,
                })?;
            if entropy_len > job.entropy_capacity {
                return Err(CudaError::OutputTooSmall {
                    required: entropy_len,
                    have: job.entropy_capacity,
                });
            }
            let mut out = vec![0u8; entropy_len];
            entropy.copy_range_to_host(0, &mut out)?;
            Ok(out)
        }
    }

    /// Encode same-buffer CUDA-resident tiles into baseline JPEG entropy chunks.
    #[expect(
        clippy::too_many_lines,
        reason = "the batch path keeps upload, launch, status, and bounded copy checks together"
    )]
    #[cfg_attr(
        feature = "cuda-oxide-jpeg-encode",
        expect(
            clippy::similar_names,
            reason = "DC/AC luma/chroma names mirror the four distinct JPEG Huffman table roles"
        )
    )]
    #[doc(hidden)]
    pub fn encode_jpeg_baseline_entropy_batch(
        &self,
        job: &CudaJpegBaselineEntropyEncodeBatchJob<'_>,
    ) -> Result<Vec<Vec<u8>>, CudaError> {
        if job.params.is_empty() {
            return Ok(Vec::new());
        }

        #[cfg(not(feature = "cuda-oxide-jpeg-encode"))]
        {
            let _ = job;
            Err(CudaError::InvalidArgument {
                message: "CUDA JPEG baseline encode PTX was not built".to_string(),
            })
        }

        #[cfg(feature = "cuda-oxide-jpeg-encode")]
        {
            self.inner.set_current()?;
            let tile_count =
                u32::try_from(job.params.len()).map_err(|_| CudaError::LengthTooLarge {
                    len: job.params.len(),
                })?;
            let entropy = self.allocate(job.entropy_capacity)?;
            let mut statuses = vec![CudaJpegBaselineEncodeStatus::default(); job.params.len()];
            let status_buffer =
                self.upload(cuda_jpeg_baseline_encode_statuses_as_bytes(&statuses))?;
            let params_buffer =
                self.upload(cuda_jpeg_baseline_encode_params_as_bytes(&job.params))?;
            let q_luma = self.upload(&job.q_luma)?;
            let q_chroma = self.upload(&job.q_chroma)?;
            let huff_dc_luma = self.upload(cuda_jpeg_baseline_encode_huffman_table_as_bytes(
                &job.huff_dc_luma,
            ))?;
            let huff_ac_luma = self.upload(cuda_jpeg_baseline_encode_huffman_table_as_bytes(
                &job.huff_ac_luma,
            ))?;
            let huff_dc_chroma = self.upload(cuda_jpeg_baseline_encode_huffman_table_as_bytes(
                &job.huff_dc_chroma,
            ))?;
            let huff_ac_chroma = self.upload(cuda_jpeg_baseline_encode_huffman_table_as_bytes(
                &job.huff_ac_chroma,
            ))?;
            self.launch_jpeg_encode_baseline_entropy_batch(&CudaJpegBaselineEntropyBatchLaunch {
                input: job.input,
                entropy: &entropy,
                status: &status_buffer,
                params: &params_buffer,
                quant: CudaJpegBaselineQuantLaunch {
                    luma: &q_luma,
                    chroma: &q_chroma,
                },
                huffman: CudaJpegBaselineHuffmanLaunch {
                    dc_luma: &huff_dc_luma,
                    ac_luma: &huff_ac_luma,
                    dc_chroma: &huff_dc_chroma,
                    ac_chroma: &huff_ac_chroma,
                },
                tile_count,
            })?;
            status_buffer.copy_to_host(cuda_jpeg_baseline_encode_statuses_as_bytes_mut(
                &mut statuses,
            ))?;
            let mut out = Vec::with_capacity(job.params.len());
            for (index, (status, params)) in statuses.iter().copied().zip(&job.params).enumerate() {
                validate_jpeg_encode_status(status, "j2k_jpeg_encode_baseline_entropy_batch")?;
                let entropy_len =
                    usize::try_from(status.entropy_len).map_err(|_| CudaError::LengthTooLarge {
                        len: status.entropy_len as usize,
                    })?;
                let offset = usize::try_from(params.entropy_offset_bytes).map_err(|_| {
                    CudaError::LengthTooLarge {
                        len: params.entropy_offset_bytes as usize,
                    }
                })?;
                let capacity = usize::try_from(params.entropy_capacity).map_err(|_| {
                    CudaError::LengthTooLarge {
                        len: params.entropy_capacity as usize,
                    }
                })?;
                if entropy_len > capacity {
                    return Err(CudaError::OutputTooSmall {
                        required: entropy_len,
                        have: capacity,
                    });
                }
                let end = offset
                    .checked_add(entropy_len)
                    .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
                if end > job.entropy_capacity {
                    return Err(CudaError::OutputTooSmall {
                        required: end,
                        have: job.entropy_capacity,
                    });
                }
                let mut chunk = vec![0u8; entropy_len];
                entropy
                    .copy_range_to_host(offset, &mut chunk)
                    .map_err(|error| {
                        if matches!(error, CudaError::OutputTooSmall { .. }) {
                            CudaError::InvalidArgument {
                                message: format!(
                                "JPEG CUDA encode batch tile {index} entropy range is out of bounds"
                            ),
                            }
                        } else {
                            error
                        }
                    })?;
                out.push(chunk);
            }
            Ok(out)
        }
    }
    #[cfg(feature = "cuda-oxide-jpeg-encode")]
    #[expect(
        clippy::similar_names,
        reason = "DC/AC luma/chroma pointer names preserve CUDA parameter order"
    )]
    fn launch_jpeg_encode_baseline_entropy(
        &self,
        request: &CudaJpegBaselineEntropyLaunch<'_>,
    ) -> Result<(), CudaError> {
        let function = self.jpeg_encode_kernel_function(CudaKernel::JpegEncodeBaselineEntropy)?;
        let input_offset =
            u64::try_from(request.input_offset).map_err(|_| CudaError::LengthTooLarge {
                len: request.input_offset,
            })?;
        let mut input_ptr = request
            .input
            .device_ptr()
            .checked_add(input_offset)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        let mut entropy_ptr = request.entropy.device_ptr();
        let mut status_ptr = request.status.device_ptr();
        let mut params = request.params;
        let mut q_luma_ptr = request.quant.luma.device_ptr();
        let mut q_chroma_ptr = request.quant.chroma.device_ptr();
        let mut huff_dc_luma_ptr = request.huffman.dc_luma.device_ptr();
        let mut huff_ac_luma_ptr = request.huffman.ac_luma.device_ptr();
        let mut huff_dc_chroma_ptr = request.huffman.dc_chroma.device_ptr();
        let mut huff_ac_chroma_ptr = request.huffman.ac_chroma.device_ptr();
        let mut kernel_params = cuda_kernel_params!(
            input_ptr,
            entropy_ptr,
            status_ptr,
            params,
            q_luma_ptr,
            q_chroma_ptr,
            huff_dc_luma_ptr,
            huff_ac_luma_ptr,
            huff_dc_chroma_ptr,
            huff_ac_chroma_ptr
        );
        self.launch_kernel(
            function,
            CudaLaunchGeometry {
                grid: (1, 1, 1),
                block: (1, 1, 1),
            },
            &mut kernel_params,
        )
    }

    #[cfg(feature = "cuda-oxide-jpeg-encode")]
    #[expect(
        clippy::similar_names,
        reason = "DC/AC luma/chroma pointer names preserve CUDA parameter order"
    )]
    fn launch_jpeg_encode_baseline_entropy_batch(
        &self,
        request: &CudaJpegBaselineEntropyBatchLaunch<'_>,
    ) -> Result<(), CudaError> {
        let function =
            self.jpeg_encode_kernel_function(CudaKernel::JpegEncodeBaselineEntropyBatch)?;
        let mut input_ptr = request.input.device_ptr();
        let mut entropy_ptr = request.entropy.device_ptr();
        let mut status_ptr = request.status.device_ptr();
        let mut params_ptr = request.params.device_ptr();
        let mut q_luma_ptr = request.quant.luma.device_ptr();
        let mut q_chroma_ptr = request.quant.chroma.device_ptr();
        let mut huff_dc_luma_ptr = request.huffman.dc_luma.device_ptr();
        let mut huff_ac_luma_ptr = request.huffman.ac_luma.device_ptr();
        let mut huff_dc_chroma_ptr = request.huffman.dc_chroma.device_ptr();
        let mut huff_ac_chroma_ptr = request.huffman.ac_chroma.device_ptr();
        let mut tile_count = request.tile_count;
        let mut kernel_params = cuda_kernel_params!(
            input_ptr,
            entropy_ptr,
            status_ptr,
            params_ptr,
            q_luma_ptr,
            q_chroma_ptr,
            huff_dc_luma_ptr,
            huff_ac_luma_ptr,
            huff_dc_chroma_ptr,
            huff_ac_chroma_ptr,
            tile_count
        );
        self.launch_kernel(
            function,
            CudaLaunchGeometry {
                grid: (tile_count, 1, 1),
                block: (1, 1, 1),
            },
            &mut kernel_params,
        )
    }

    #[cfg(feature = "cuda-oxide-jpeg-encode")]
    fn jpeg_encode_kernel_function(
        &self,
        kernel: CudaKernel,
    ) -> Result<crate::driver::CuFunction, CudaError> {
        self.inner.cuda_oxide_jpeg_encode_kernel_function(kernel)
    }
}

#[cfg(feature = "cuda-oxide-jpeg-encode")]
fn validate_jpeg_encode_status(
    status: CudaJpegBaselineEncodeStatus,
    kernel: &'static str,
) -> Result<(), CudaError> {
    match status.code {
        JPEG_BASELINE_ENCODE_STATUS_OK => Ok(()),
        JPEG_BASELINE_ENCODE_STATUS_OVERFLOW
        | JPEG_BASELINE_ENCODE_STATUS_MISSING_HUFFMAN
        | JPEG_BASELINE_ENCODE_STATUS_INVALID_PARAMS => Err(CudaError::KernelStatus {
            kernel,
            code: status.code,
            detail: status.detail,
        }),
        code => Err(CudaError::KernelStatus {
            kernel,
            code,
            detail: status.detail,
        }),
    }
}
