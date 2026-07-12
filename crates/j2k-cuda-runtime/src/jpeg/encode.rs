// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "cuda-oxide-jpeg-encode")]
use super::{
    encode_launch::{
        validate_jpeg_encode_status, CudaJpegBaselineEntropyLaunch, CudaJpegBaselineHuffmanLaunch,
        CudaJpegBaselineQuantLaunch,
    },
    CudaJpegBaselineEncodeStatus,
};
use super::{
    encode_validation::{
        validate_jpeg_baseline_encode_request, validate_jpeg_encode_batch_launch,
        CudaJpegBaselineEncodeTableRefs,
    },
    validation::validate_jpeg_buffer_context,
    CudaJpegBaselineEncodeParams, CudaJpegBaselineEntropyEncodeBatchJob,
    CudaJpegBaselineEntropyEncodeJob,
};
use crate::allocation::host_element_bytes;
#[cfg(feature = "cuda-oxide-jpeg-encode")]
use crate::{
    allocation::try_vec_filled,
    bytes::{
        cuda_jpeg_baseline_encode_huffman_table_as_bytes,
        cuda_jpeg_baseline_encode_statuses_as_bytes,
        cuda_jpeg_baseline_encode_statuses_as_bytes_mut,
    },
};
use crate::{context::CudaContext, error::CudaError};

#[cfg(feature = "cuda-oxide-jpeg-encode")]
use super::encode_allocation::{
    checked_batch_private_host_bytes, checked_single_private_host_bytes,
};

impl CudaContext {
    /// Encode one CUDA-resident tile into baseline JPEG entropy bytes.
    /// The resident input must belong to this context.
    #[doc(hidden)]
    pub fn encode_jpeg_baseline_entropy(
        &self,
        job: &CudaJpegBaselineEntropyEncodeJob<'_>,
    ) -> Result<Vec<u8>, CudaError> {
        self.encode_jpeg_baseline_entropy_with_external_live(job, 0)
    }

    /// Encode while charging host owners retained by the adapter.
    #[cfg_attr(
        feature = "cuda-oxide-jpeg-encode",
        expect(
            clippy::similar_names,
            reason = "DC/AC luma/chroma names mirror the four distinct JPEG Huffman table roles"
        )
    )]
    #[doc(hidden)]
    pub fn encode_jpeg_baseline_entropy_with_external_live(
        &self,
        job: &CudaJpegBaselineEntropyEncodeJob<'_>,
        external_live_bytes: usize,
    ) -> Result<Vec<u8>, CudaError> {
        validate_jpeg_buffer_context(self, [job.input])?;
        #[cfg(feature = "cuda-oxide-jpeg-encode")]
        checked_single_private_host_bytes(external_live_bytes, job.entropy_capacity)?;
        let validated = validate_jpeg_baseline_encode_request(
            job.input.device_ptr(),
            job.input.byte_len(),
            job.input_offset,
            std::slice::from_ref(&job.params),
            job.entropy_capacity,
            CudaJpegBaselineEncodeTableRefs {
                q_luma: &job.q_luma,
                q_chroma: &job.q_chroma,
                huff_dc_luma: &job.huff_dc_luma,
                huff_ac_luma: &job.huff_ac_luma,
                huff_dc_chroma: &job.huff_dc_chroma,
                huff_ac_chroma: &job.huff_ac_chroma,
            },
            0,
        )?;
        #[cfg(not(feature = "cuda-oxide-jpeg-encode"))]
        {
            let _ = (job, validated, external_live_bytes);
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
                input_ptr: validated.first_tile.input_ptr,
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
            if entropy_len > validated.first_tile.entropy_capacity {
                return Err(CudaError::OutputTooSmall {
                    required: entropy_len,
                    have: validated.first_tile.entropy_capacity,
                });
            }
            let mut out = try_vec_filled(entropy_len, 0u8)?;
            checked_single_private_host_bytes(external_live_bytes, out.capacity())?;
            entropy.copy_range_to_host(validated.first_tile.entropy_offset, &mut out)?;
            Ok(out)
        }
    }

    /// Encode same-buffer CUDA-resident tiles into baseline JPEG entropy chunks.
    /// The resident input must belong to this context when `params` is nonempty;
    /// an empty batch remains a no-op and does not inspect the input buffer.
    #[doc(hidden)]
    pub fn encode_jpeg_baseline_entropy_batch(
        &self,
        job: &CudaJpegBaselineEntropyEncodeBatchJob<'_>,
    ) -> Result<Vec<Vec<u8>>, CudaError> {
        self.encode_jpeg_baseline_entropy_batch_with_external_live(job, 0)
    }

    /// Encode a batch while charging host owners retained by the adapter.
    #[doc(hidden)]
    pub fn encode_jpeg_baseline_entropy_batch_with_external_live(
        &self,
        job: &CudaJpegBaselineEntropyEncodeBatchJob<'_>,
        external_live_bytes: usize,
    ) -> Result<Vec<Vec<u8>>, CudaError> {
        if job.params.is_empty() {
            return Ok(Vec::new());
        }
        validate_jpeg_buffer_context(self, [job.input])?;
        #[cfg(feature = "cuda-oxide-jpeg-encode")]
        checked_batch_private_host_bytes(
            external_live_bytes,
            job.params.capacity(),
            job.params.len(),
            job.params.len(),
            job.params.len(),
            job.entropy_capacity,
        )?;
        let validated = validate_jpeg_baseline_encode_request(
            job.input.device_ptr(),
            job.input.byte_len(),
            0,
            &job.params,
            job.entropy_capacity,
            CudaJpegBaselineEncodeTableRefs {
                q_luma: &job.q_luma,
                q_chroma: &job.q_chroma,
                huff_dc_luma: &job.huff_dc_luma,
                huff_ac_luma: &job.huff_ac_luma,
                huff_dc_chroma: &job.huff_dc_chroma,
                huff_ac_chroma: &job.huff_ac_chroma,
            },
            host_element_bytes::<CudaJpegBaselineEncodeParams>(job.params.capacity()),
        )?;
        let batch_geometry = validate_jpeg_encode_batch_launch(validated.tile_count)?;

        #[cfg(not(feature = "cuda-oxide-jpeg-encode"))]
        {
            let _ = (job, validated, batch_geometry, external_live_bytes);
            Err(CudaError::InvalidArgument {
                message: "CUDA JPEG baseline encode PTX was not built".to_string(),
            })
        }

        #[cfg(feature = "cuda-oxide-jpeg-encode")]
        {
            self.execute_jpeg_baseline_entropy_batch(
                job,
                external_live_bytes,
                validated,
                batch_geometry,
            )
        }
    }
}
