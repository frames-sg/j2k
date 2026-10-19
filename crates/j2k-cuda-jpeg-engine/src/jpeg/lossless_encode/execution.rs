// SPDX-License-Identifier: MIT OR Apache-2.0

//! Device buffers and ordered launches of the lossless encode stages.

use j2k_core::accelerator::GpuAbi;

use super::{
    validate_lossless_encode_job, CudaJpegLosslessEncodeJob, CudaJpegLosslessEncodeOutcome,
    CudaJpegLosslessEncodeParams, CUDA_JPEG_LOSSLESS_CATEGORIES,
};
use crate::{
    allocation::{try_vec_filled, HostPhaseBudget},
    error::CudaError,
    execution::cuda_kernel_param,
    kernels::{CudaKernel, CudaLaunchGeometry},
    memory::CudaDeviceBuffer,
    JpegCudaEngine,
};

/// Threads per block; the classify kernel's block histogram needs at least
/// one thread per category.
const BLOCK_THREADS: u32 = 256;
const STATUS_WORDS: usize = 4;
const STATUS_BAD_INDEX: usize = 0;
const STATUS_BAD_VALUE: usize = 1;
const STATUS_PACKED_BYTES: usize = 2;

impl JpegCudaEngine<'_> {
    pub(super) fn execute_lossless_encode<E: From<CudaError>>(
        self,
        job: &CudaJpegLosslessEncodeJob<'_>,
        external_live_bytes: usize,
        code_table: impl FnOnce(
            &[u32; CUDA_JPEG_LOSSLESS_CATEGORIES],
        ) -> Result<[u32; CUDA_JPEG_LOSSLESS_CATEGORIES], E>,
    ) -> Result<CudaJpegLosslessEncodeOutcome, E> {
        validate_lossless_encode_job(job)?;
        let p = job.params;
        let params = self.upload(CudaJpegLosslessEncodeParams::as_bytes(&p))?;
        let tokens = self.allocate(words(p.sample_count)?)?;
        let histogram = self.upload(u32::slice_as_bytes(&[0; CUDA_JPEG_LOSSLESS_CATEGORIES]))?;
        let status = self.upload(u32::slice_as_bytes(&[u32::MAX, 0, 0, 0]))?;
        let input = job.input.device_ptr();
        self.launch(
            CudaKernel::JpegLosslessEncodeClassify,
            p.sample_count,
            &mut [
                input,
                tokens.device_ptr(),
                histogram.device_ptr(),
                status.device_ptr(),
                params.device_ptr(),
            ],
        )?;
        let mut words_out = [0_u32; STATUS_WORDS];
        status.copy_to_host(u32::slice_as_bytes_mut(&mut words_out))?;
        if words_out[STATUS_BAD_INDEX] != u32::MAX {
            self.launch(
                CudaKernel::JpegLosslessEncodeReadSample,
                1,
                &mut [input, status.device_ptr(), params.device_ptr()],
            )?;
            status.copy_to_host(u32::slice_as_bytes_mut(&mut words_out))?;
            return Ok(CudaJpegLosslessEncodeOutcome::SampleOutOfRange {
                index: words_out[STATUS_BAD_INDEX],
                value: words_out[STATUS_BAD_VALUE],
            });
        }
        let mut counts = [0_u32; CUDA_JPEG_LOSSLESS_CATEGORIES];
        histogram.copy_to_host(u32::slice_as_bytes_mut(&mut counts))?;
        let table = self.upload(u32::slice_as_bytes(&code_table(&counts)?))?;
        Ok(self.pack_lossless_stream(
            &p,
            [&params, &tokens, &status, &table],
            external_live_bytes,
        )?)
    }

    /// Stages 3-5: code lengths, offsets, and the packed stream, read back
    /// with each interval's byte length.
    fn pack_lossless_stream(
        self,
        p: &CudaJpegLosslessEncodeParams,
        [params, tokens, status, table]: [&CudaDeviceBuffer; 4],
        external_live_bytes: usize,
    ) -> Result<CudaJpegLosslessEncodeOutcome, CudaError> {
        let chunks = p
            .chunks_per_row
            .checked_mul(p.height)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        let chunk_bits = self.allocate(words(chunks)?)?;
        let chunk_offsets = self.allocate(words(chunks)?)?;
        let row_bits = self.allocate(words(p.height)?)?;
        let row_offsets = self.allocate(words(p.height)?)?;
        let interval_bits = self.allocate(words(p.intervals)?)?;
        let interval_bases = self.allocate(words(p.intervals)?)?;
        let interval_bytes = self.allocate(words(p.intervals)?)?;
        let packed = self.allocate(words(p.packed_words)?)?;
        // Each launch completes before the next starts.
        self.launch(
            CudaKernel::JpegLosslessEncodeChunkBits,
            chunks,
            &mut [
                tokens.device_ptr(),
                chunk_bits.device_ptr(),
                params.device_ptr(),
                table.device_ptr(),
            ],
        )?;
        self.launch(
            CudaKernel::JpegLosslessEncodeRowOffsets,
            p.height,
            &mut [
                chunk_bits.device_ptr(),
                chunk_offsets.device_ptr(),
                row_bits.device_ptr(),
                params.device_ptr(),
            ],
        )?;
        self.launch(
            CudaKernel::JpegLosslessEncodeIntervalOffsets,
            p.intervals,
            &mut [
                row_bits.device_ptr(),
                row_offsets.device_ptr(),
                interval_bits.device_ptr(),
                params.device_ptr(),
            ],
        )?;
        self.launch(
            CudaKernel::JpegLosslessEncodeIntervalBases,
            1,
            &mut [
                interval_bits.device_ptr(),
                interval_bases.device_ptr(),
                interval_bytes.device_ptr(),
                status.device_ptr(),
                params.device_ptr(),
            ],
        )?;
        self.launch(
            CudaKernel::JpegLosslessEncodeClear,
            p.packed_words,
            &mut [
                packed.device_ptr(),
                status.device_ptr(),
                params.device_ptr(),
            ],
        )?;
        self.launch(
            CudaKernel::JpegLosslessEncodePack,
            chunks,
            &mut [
                tokens.device_ptr(),
                packed.device_ptr(),
                chunk_offsets.device_ptr(),
                row_offsets.device_ptr(),
                interval_bases.device_ptr(),
                interval_bits.device_ptr(),
                params.device_ptr(),
                table.device_ptr(),
            ],
        )?;
        let mut words_out = [0_u32; STATUS_WORDS];
        status.copy_to_host(u32::slice_as_bytes_mut(&mut words_out))?;
        let packed_bytes = words_out[STATUS_PACKED_BYTES] as usize;
        if packed_bytes > packed.byte_len() {
            return Err(CudaError::OutputTooSmall {
                required: packed_bytes,
                have: packed.byte_len(),
            });
        }
        let mut budget = HostPhaseBudget::new("JPEG lossless CUDA encode readback");
        for bytes in [external_live_bytes, packed_bytes, p.intervals as usize * 4] {
            budget.account_bytes(bytes).map_err(CudaError::from)?;
        }
        let mut interval_lengths = try_vec_filled(p.intervals as usize, 0_u32)?;
        interval_bytes.copy_to_host(u32::slice_as_bytes_mut(&mut interval_lengths))?;
        let mut stream = try_vec_filled(packed_bytes, 0_u8)?;
        packed.copy_range_to_host(0, &mut stream)?;
        Ok(CudaJpegLosslessEncodeOutcome::Packed {
            stream,
            interval_lengths,
        })
    }

    /// Launch `kernel` over `threads` threads and wait for it. `pointers`
    /// are device-pointer kernel arguments in parameter order.
    fn launch(
        self,
        kernel: CudaKernel,
        threads: u32,
        pointers: &mut [u64],
    ) -> Result<(), CudaError> {
        let geometry = CudaLaunchGeometry::new(
            (threads.div_ceil(BLOCK_THREADS).max(1), 1, 1),
            (BLOCK_THREADS, 1, 1),
        )
        .ok_or_else(|| CudaError::InvalidArgument {
            message: "JPEG lossless CUDA encode launch geometry exceeds device limits".to_string(),
        })?;
        let mut params: Vec<*mut core::ffi::c_void> =
            pointers.iter_mut().map(cuda_kernel_param).collect();
        self.launch_kernel(kernel, geometry, &mut params)
    }
}

fn words(count: u32) -> Result<usize, CudaError> {
    (count as usize)
        .checked_mul(4)
        .map(|bytes| bytes.max(4))
        .ok_or(CudaError::LengthTooLarge { len: usize::MAX })
}
