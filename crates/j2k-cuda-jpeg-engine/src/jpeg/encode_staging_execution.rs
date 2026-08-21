// SPDX-License-Identifier: MIT OR Apache-2.0

//! Device scratch ownership and ordered staged CUDA baseline JPEG submission.

use super::{
    encode_allocation::{checked_batch_private_host_bytes, checked_single_private_host_bytes},
    encode_batch::LaunchedBatch,
    encode_launch::{validate_jpeg_encode_status, CudaJpegBaselineHuffmanLaunch},
    encode_staging::checked_staged_encode_plan,
    encode_staging_launch::{
        CudaJpegBaselineEntropyFromCoeffsLaunch, CudaJpegBaselinePrecomputeLaunch,
    },
    encode_validation::CudaJpegBaselineEncodeValidation,
    CudaJpegBaselineEncodeStatus, CudaJpegBaselineEntropyEncodeBatchJob,
    CudaJpegBaselineEntropyEncodeJob,
};
use crate::{
    allocation::{try_vec_defaulted, try_vec_filled},
    bytes::{
        cuda_jpeg_baseline_encode_huffman_table_as_bytes,
        cuda_jpeg_baseline_encode_params_as_bytes, cuda_jpeg_baseline_encode_statuses_as_bytes,
        cuda_jpeg_baseline_encode_statuses_as_bytes_mut,
    },
    error::CudaError,
    kernels::CudaLaunchGeometry,
    memory::CudaDeviceBuffer,
    JpegCudaEngine,
};

struct UploadedHuffmanTables {
    dc_luma: CudaDeviceBuffer,
    ac_luma: CudaDeviceBuffer,
    dc_chroma: CudaDeviceBuffer,
    ac_chroma: CudaDeviceBuffer,
}

impl JpegCudaEngine<'_> {
    pub(super) fn execute_staged_jpeg_baseline_entropy(
        self,
        job: &CudaJpegBaselineEntropyEncodeJob<'_>,
        external_live_bytes: usize,
        validated: CudaJpegBaselineEncodeValidation,
    ) -> Result<Vec<u8>, CudaError> {
        let mut params = job.params;
        params.input_offset_bytes = 0;
        params.entropy_offset_bytes = 0;
        let plan = checked_staged_encode_plan(std::slice::from_ref(&params))?;
        let entropy = self.allocate(job.entropy_capacity)?;
        let coefficients = self.allocate(plan.coefficient_bytes)?;
        let params_buffer = self.upload(cuda_jpeg_baseline_encode_params_as_bytes(
            std::slice::from_ref(&params),
        ))?;
        let mut statuses = [CudaJpegBaselineEncodeStatus::default()];
        let status_buffer = self.upload(cuda_jpeg_baseline_encode_statuses_as_bytes(&statuses))?;
        let q_luma = self.upload(&job.q_luma)?;
        let q_chroma = self.upload(&job.q_chroma)?;
        let huffman = self.upload_huffman_tables(
            &job.huff_dc_luma,
            &job.huff_ac_luma,
            &job.huff_dc_chroma,
            &job.huff_ac_chroma,
        )?;
        self.launch_staged_jpeg_baseline_encode(
            validated.first_tile.input_ptr,
            &coefficients,
            &entropy,
            &status_buffer,
            &params_buffer,
            &q_luma,
            &q_chroma,
            &huffman,
            plan.precompute_geometry,
            CudaLaunchGeometry::new((1, 1, 1), (1, 1, 1)).ok_or_else(|| {
                CudaError::InvalidArgument {
                    message: "fixed JPEG staged entropy geometry is invalid".to_string(),
                }
            })?,
            1,
            plan.total_mcus,
        )?;
        status_buffer.copy_to_host(cuda_jpeg_baseline_encode_statuses_as_bytes_mut(
            &mut statuses,
        ))?;
        validate_jpeg_encode_status(
            statuses[0],
            "j2k_jpeg_encode_baseline_entropy_from_coeffs_batch",
        )?;
        let entropy_len = usize::try_from(statuses[0].entropy_len)
            .map_err(|_| CudaError::LengthTooLarge { len: usize::MAX })?;
        if entropy_len > validated.first_tile.entropy_capacity {
            return Err(CudaError::OutputTooSmall {
                required: entropy_len,
                have: validated.first_tile.entropy_capacity,
            });
        }
        let mut out = try_vec_filled(entropy_len, 0u8)?;
        checked_single_private_host_bytes(external_live_bytes, out.capacity())?;
        entropy.copy_range_to_host(0, &mut out)?;
        Ok(out)
    }

    pub(super) fn execute_staged_jpeg_baseline_entropy_batch(
        self,
        job: &CudaJpegBaselineEntropyEncodeBatchJob<'_>,
        external_live_bytes: usize,
        validated: CudaJpegBaselineEncodeValidation,
        entropy_geometry: CudaLaunchGeometry,
    ) -> Result<Vec<Vec<u8>>, CudaError> {
        let plan = checked_staged_encode_plan(&job.params)?;
        let entropy = self.allocate(job.entropy_capacity)?;
        let coefficients = self.allocate(plan.coefficient_bytes)?;
        let params_buffer = self.upload(cuda_jpeg_baseline_encode_params_as_bytes(&job.params))?;
        let mut statuses: Vec<CudaJpegBaselineEncodeStatus> = try_vec_defaulted(job.params.len())?;
        checked_batch_private_host_bytes(
            external_live_bytes,
            job.params.capacity(),
            job.params.len(),
            statuses.capacity(),
            job.params.len(),
            job.entropy_capacity,
        )?;
        let status_buffer = self.upload(cuda_jpeg_baseline_encode_statuses_as_bytes(&statuses))?;
        let q_luma = self.upload(&job.q_luma)?;
        let q_chroma = self.upload(&job.q_chroma)?;
        let huffman = self.upload_huffman_tables(
            &job.huff_dc_luma,
            &job.huff_ac_luma,
            &job.huff_dc_chroma,
            &job.huff_ac_chroma,
        )?;
        self.launch_staged_jpeg_baseline_encode(
            job.input.device_ptr(),
            &coefficients,
            &entropy,
            &status_buffer,
            &params_buffer,
            &q_luma,
            &q_chroma,
            &huffman,
            plan.precompute_geometry,
            entropy_geometry,
            validated.tile_count,
            plan.total_mcus,
        )?;
        status_buffer.copy_to_host(cuda_jpeg_baseline_encode_statuses_as_bytes_mut(
            &mut statuses,
        ))?;
        Self::collect_jpeg_baseline_entropy_batch(
            job,
            external_live_bytes,
            LaunchedBatch { entropy, statuses },
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the staged boundary owns two ordered kernels and their shared device buffers"
    )]
    fn launch_staged_jpeg_baseline_encode(
        self,
        input_ptr: u64,
        coefficients: &CudaDeviceBuffer,
        entropy: &CudaDeviceBuffer,
        status: &CudaDeviceBuffer,
        params: &CudaDeviceBuffer,
        q_luma: &CudaDeviceBuffer,
        q_chroma: &CudaDeviceBuffer,
        huffman: &UploadedHuffmanTables,
        precompute_geometry: CudaLaunchGeometry,
        entropy_geometry: CudaLaunchGeometry,
        tile_count: u32,
        total_mcus: u32,
    ) -> Result<(), CudaError> {
        self.launch_jpeg_encode_baseline_precompute_batch(&CudaJpegBaselinePrecomputeLaunch {
            input_ptr,
            coefficients,
            params,
            q_luma,
            q_chroma,
            tile_count,
            total_mcus,
            geometry: precompute_geometry,
        })?;
        self.launch_jpeg_encode_baseline_entropy_from_coeffs_batch(
            &CudaJpegBaselineEntropyFromCoeffsLaunch {
                coefficients,
                entropy,
                status,
                params,
                huffman: CudaJpegBaselineHuffmanLaunch {
                    dc_luma: &huffman.dc_luma,
                    ac_luma: &huffman.ac_luma,
                    dc_chroma: &huffman.dc_chroma,
                    ac_chroma: &huffman.ac_chroma,
                },
                tile_count,
                geometry: entropy_geometry,
            },
        )
    }

    fn upload_huffman_tables(
        self,
        dc_luma: &super::CudaJpegBaselineEncodeHuffmanTable,
        ac_luma: &super::CudaJpegBaselineEncodeHuffmanTable,
        dc_chroma: &super::CudaJpegBaselineEncodeHuffmanTable,
        ac_chroma: &super::CudaJpegBaselineEncodeHuffmanTable,
    ) -> Result<UploadedHuffmanTables, CudaError> {
        Ok(UploadedHuffmanTables {
            dc_luma: self.upload(cuda_jpeg_baseline_encode_huffman_table_as_bytes(dc_luma))?,
            ac_luma: self.upload(cuda_jpeg_baseline_encode_huffman_table_as_bytes(ac_luma))?,
            dc_chroma: self.upload(cuda_jpeg_baseline_encode_huffman_table_as_bytes(dc_chroma))?,
            ac_chroma: self.upload(cuda_jpeg_baseline_encode_huffman_table_as_bytes(ac_chroma))?,
        })
    }
}
