// SPDX-License-Identifier: MIT OR Apache-2.0

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
use crate::{error::CudaError, JpegCudaEngine};

#[cfg(feature = "cuda-oxide-jpeg-encode")]
use super::encode_allocation::{
    checked_batch_private_host_bytes, checked_single_private_host_bytes,
};

impl JpegCudaEngine<'_> {
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
    #[doc(hidden)]
    pub fn encode_jpeg_baseline_entropy_with_external_live(
        &self,
        job: &CudaJpegBaselineEntropyEncodeJob<'_>,
        external_live_bytes: usize,
    ) -> Result<Vec<u8>, CudaError> {
        validate_jpeg_buffer_context(*self, [job.input])?;
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
            self.execute_staged_jpeg_baseline_entropy(job, external_live_bytes, validated)
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
        validate_jpeg_buffer_context(*self, [job.input])?;
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
            self.execute_staged_jpeg_baseline_entropy_batch(
                job,
                external_live_bytes,
                validated,
                batch_geometry,
            )
        }
    }
}
