use crate::{
    bytes::{
        htj2k_encode_compact_jobs_as_bytes, htj2k_encode_jobs_as_bytes,
        htj2k_encode_multi_input_jobs_as_bytes, htj2k_encode_params_as_bytes,
        htj2k_encode_status_as_bytes, htj2k_encode_status_as_bytes_mut,
        htj2k_encode_statuses_as_bytes_mut, htj2k_encode_statuses_byte_len, u16_slice_as_bytes,
    },
    context::{
        CudaContext, CudaHtj2kCompactEncodedCodeBlock, CudaHtj2kCompactEncodedCodeBlocks,
        HTJ2K_UVLC_ENCODE_TABLE_BYTES,
    },
    error::CudaError,
    execution::{cuda_kernel_param, CudaExecutionStats},
    htj2k_decode::{HTJ2K_STATUS_OK, HTJ2K_STATUS_UNSUPPORTED},
    kernels::{
        htj2k_codeblock_sample_launch_geometry, htj2k_encode_codeblock_launch_geometry, CudaKernel,
    },
    memory::{
        checked_image_words, copy_pooled_bytes_to_vec_uninit, pooled_device_buffer, CudaBufferPool,
        CudaDeviceBuffer,
    },
};

/// Static HTJ2K cleanup encoder lookup tables uploaded for CUDA code-block encode.
#[derive(Clone, Copy, Debug)]
pub struct CudaHtj2kEncodeTables<'a> {
    /// HT cleanup encoder VLC table for first quad row contexts.
    pub vlc_table0: &'a [u16; 2048],
    /// HT cleanup encoder VLC table for subsequent quad row contexts.
    pub vlc_table1: &'a [u16; 2048],
    /// Packed HT cleanup encoder UVLC table rows, six bytes per row.
    pub uvlc_table: &'a [u8],
}

/// Device-resident HTJ2K cleanup encode lookup tables reused across sub-band dispatches.
#[derive(Debug)]
pub struct CudaHtj2kEncodeResources {
    pub(crate) vlc_table0: CudaDeviceBuffer,
    pub(crate) vlc_table1: CudaDeviceBuffer,
    pub(crate) uvlc_table: CudaDeviceBuffer,
}

pub(crate) const HTJ2K_ENCODE_MAX_CODEBLOCK_WIDTH: u32 = 1024;

pub(crate) const HTJ2K_ENCODE_MAX_CODEBLOCK_SAMPLES: usize = 4096;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CudaHtj2kEncodeParams {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) coefficient_stride: u32,
    pub(crate) total_bitplanes: u32,
    pub(crate) output_capacity: u32,
    pub(crate) target_coding_passes: u32,
}

/// One HTJ2K code-block encode job consumed by the CUDA batch encoder.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaHtj2kEncodeCodeBlockJob {
    /// Offset, in i32 coefficients, into the contiguous coefficient buffer.
    pub coefficient_offset: u32,
    /// Code-block width in coefficients.
    pub width: u32,
    /// Code-block height in coefficients.
    pub height: u32,
    /// Total coded bitplanes for this code block's sub-band.
    pub total_bitplanes: u8,
    /// Requested HT coding passes. `1` emits cleanup-only output; `2` emits a
    /// zero `SigProp` segment for exactly representable blocks; `3` emits
    /// `SigProp` bits for newly significant magnitude-3 samples plus `MagRef`
    /// bits for cleanup-significant samples.
    pub target_coding_passes: u8,
}

/// One HTJ2K code-block region consumed from a strided resident coefficient buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaHtj2kEncodeCodeBlockRegionJob {
    /// Offset, in i32 coefficients, to the top-left coefficient of this code block.
    pub coefficient_offset: u32,
    /// Source row stride in i32 coefficients.
    pub coefficient_stride: u32,
    /// Code-block width in coefficients.
    pub width: u32,
    /// Code-block height in coefficients.
    pub height: u32,
    /// Total coded bitplanes for this code block's sub-band.
    pub total_bitplanes: u8,
    /// Requested HT coding passes. `1` emits cleanup-only output; `2` emits a
    /// zero `SigProp` segment for exactly representable blocks; `3` emits
    /// `SigProp` bits for newly significant magnitude-3 samples plus `MagRef`
    /// bits for cleanup-significant samples.
    pub target_coding_passes: u8,
}

/// Resident coefficient buffer and jobs for a multi-input HTJ2K encode batch.
#[derive(Clone, Copy, Debug)]
pub struct CudaHtj2kEncodeResidentTarget<'a> {
    /// Device buffer containing quantized i32 coefficients.
    pub coefficients: &'a CudaDeviceBuffer,
    /// Number of i32 coefficients available in `coefficients`.
    pub coefficient_count: usize,
    /// Code-block jobs that read from `coefficients`.
    pub jobs: &'a [CudaHtj2kEncodeCodeBlockJob],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CudaHtj2kEncodeKernelJob {
    pub(crate) coefficient_offset: u32,
    pub(crate) coefficient_stride: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) total_bitplanes: u32,
    pub(crate) output_offset: u32,
    pub(crate) output_capacity: u32,
    pub(crate) target_coding_passes: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CudaHtj2kEncodeMultiInputKernelJob {
    pub(crate) coefficient_ptr: u64,
    pub(crate) coefficient_offset: u32,
    pub(crate) coefficient_stride: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) total_bitplanes: u32,
    pub(crate) output_offset: u32,
    pub(crate) output_capacity: u32,
    pub(crate) target_coding_passes: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CudaHtj2kEncodeCompactJob {
    pub(crate) source_offset: u32,
    pub(crate) compact_offset: u32,
    pub(crate) data_len: u32,
    pub(crate) reserved: u32,
}

/// Status written by the CUDA HTJ2K code-block cleanup-pass encoder.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaHtj2kEncodeStatus {
    /// Zero on success; nonzero values are kernel-defined failures.
    pub code: u32,
    /// Kernel-defined failure detail.
    pub detail: u32,
    /// Encoded payload byte length.
    pub data_len: u32,
    /// Number of coding passes in the encoded payload.
    pub number_of_coding_passes: u32,
    /// Number of missing most-significant bitplanes.
    pub missing_bit_planes: u32,
    /// Reserved for ABI stability.
    pub reserved0: u32,
    /// Reserved for ABI stability.
    pub reserved1: u32,
    /// Reserved for ABI stability.
    pub reserved2: u32,
}

impl CudaHtj2kEncodeStatus {
    /// Return true when the CUDA kernel reported success.
    pub fn is_ok(self) -> bool {
        self.code == HTJ2K_STATUS_OK
    }
}

/// CUDA event timings for HTJ2K cleanup-pass encode stages.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaHtj2kEncodeStageTimings {
    /// Total HT cleanup-pass encode, compaction, and required result readback time, in microseconds.
    pub ht_encode_us: u128,
    /// HT cleanup-pass encode kernel time, in microseconds.
    pub ht_kernel_us: u128,
    /// Status-buffer device-to-host readback time, in microseconds.
    pub ht_status_readback_us: u128,
    /// Encoded-byte compaction kernel time, in microseconds.
    pub ht_compact_us: u128,
    /// Compacted encoded-byte device-to-host readback time, in microseconds.
    pub ht_output_readback_us: u128,
}

impl CudaHtj2kEncodeStageTimings {
    fn from_parts(
        ht_kernel_us: u128,
        ht_status_readback_us: u128,
        ht_compact_us: u128,
        ht_output_readback_us: u128,
    ) -> Self {
        Self {
            ht_encode_us: ht_kernel_us
                .saturating_add(ht_status_readback_us)
                .saturating_add(ht_compact_us)
                .saturating_add(ht_output_readback_us),
            ht_kernel_us,
            ht_status_readback_us,
            ht_compact_us,
            ht_output_readback_us,
        }
    }
}

/// Host-visible HTJ2K cleanup-pass encode result produced by a CUDA kernel.
#[derive(Debug)]
pub struct CudaHtj2kEncodedCodeBlock {
    pub(crate) data: Vec<u8>,
    pub(crate) status: CudaHtj2kEncodeStatus,
    pub(crate) execution: CudaExecutionStats,
    pub(crate) stage_timings: CudaHtj2kEncodeStageTimings,
}

impl CudaHtj2kEncodedCodeBlock {
    /// Encoded cleanup-pass payload bytes.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// HTJ2K cleanup segment length in bytes.
    pub fn cleanup_length(&self) -> u32 {
        if self.status.number_of_coding_passes <= 1 {
            self.status.data_len
        } else {
            self.status.reserved0
        }
    }

    /// HTJ2K refinement segment length in bytes.
    pub fn refinement_length(&self) -> u32 {
        if self.status.number_of_coding_passes <= 1 {
            0
        } else {
            self.status.reserved1
        }
    }

    /// Number of coding passes in the encoded payload.
    pub fn num_coding_passes(&self) -> u8 {
        u8::try_from(self.status.number_of_coding_passes).unwrap_or(u8::MAX)
    }

    /// Number of missing most-significant bitplanes.
    pub fn num_zero_bitplanes(&self) -> u8 {
        u8::try_from(self.status.missing_bit_planes).unwrap_or(u8::MAX)
    }

    /// Consume this code block and return its encoded payload plus segment
    /// metadata.
    pub fn into_parts(self) -> (Vec<u8>, u32, u32, u8, u8) {
        let cleanup_length = if self.status.number_of_coding_passes <= 1 {
            self.status.data_len
        } else {
            self.status.reserved0
        };
        let refinement_length = if self.status.number_of_coding_passes <= 1 {
            0
        } else {
            self.status.reserved1
        };
        (
            self.data,
            cleanup_length,
            refinement_length,
            u8::try_from(self.status.number_of_coding_passes).unwrap_or(u8::MAX),
            u8::try_from(self.status.missing_bit_planes).unwrap_or(u8::MAX),
        )
    }

    /// Kernel status row downloaded after dispatch.
    pub fn status(&self) -> CudaHtj2kEncodeStatus {
        self.status
    }

    /// CUDA execution counters for the encode dispatch.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }

    /// CUDA event timings for the encode dispatch.
    pub fn stage_timings(&self) -> CudaHtj2kEncodeStageTimings {
        self.stage_timings
    }
}

/// Host-visible HTJ2K cleanup-pass encode batch produced by one CUDA kernel dispatch.
#[derive(Debug)]
pub struct CudaHtj2kEncodedCodeBlocks {
    pub(crate) code_blocks: Vec<CudaHtj2kEncodedCodeBlock>,
    pub(crate) execution: CudaExecutionStats,
    pub(crate) stage_timings: CudaHtj2kEncodeStageTimings,
}

impl CudaHtj2kEncodedCodeBlocks {
    /// Encoded cleanup code-block payloads, in the same order as the submitted jobs.
    pub fn code_blocks(&self) -> &[CudaHtj2kEncodedCodeBlock] {
        &self.code_blocks
    }

    /// Consume the batch and return its per-code-block outputs.
    pub fn into_code_blocks(self) -> Vec<CudaHtj2kEncodedCodeBlock> {
        self.code_blocks
    }

    /// CUDA execution counters for the batch encode dispatch.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }

    /// CUDA event timings for the batch encode dispatch.
    pub fn stage_timings(&self) -> CudaHtj2kEncodeStageTimings {
        self.stage_timings
    }
}

pub(crate) const HTJ2K_ENCODE_OUTPUT_CAPACITY: usize = 24 * 1024;

impl CudaContext {
    /// Upload static HTJ2K cleanup encoder lookup tables once for reuse.
    pub fn upload_htj2k_encode_resources(
        &self,
        tables: CudaHtj2kEncodeTables<'_>,
    ) -> Result<CudaHtj2kEncodeResources, CudaError> {
        if tables.uvlc_table.len() != HTJ2K_UVLC_ENCODE_TABLE_BYTES {
            return Err(CudaError::LengthTooLarge {
                len: tables.uvlc_table.len(),
            });
        }
        self.inner.set_current()?;
        Ok(CudaHtj2kEncodeResources {
            vlc_table0: self.upload(u16_slice_as_bytes(tables.vlc_table0))?,
            vlc_table1: self.upload(u16_slice_as_bytes(tables.vlc_table1))?,
            uvlc_table: self.upload(tables.uvlc_table)?,
        })
    }

    /// Encode one HTJ2K cleanup-pass code block with CUDA.
    pub fn encode_htj2k_codeblock(
        &self,
        coefficients: &[i32],
        width: u32,
        height: u32,
        total_bitplanes: u8,
        tables: CudaHtj2kEncodeTables<'_>,
    ) -> Result<CudaHtj2kEncodedCodeBlock, CudaError> {
        let resources = self.upload_htj2k_encode_resources(tables)?;
        self.encode_htj2k_codeblock_with_resources(
            coefficients,
            width,
            height,
            total_bitplanes,
            &resources,
        )
    }

    /// Encode one HTJ2K cleanup-pass code block with pre-uploaded lookup tables.
    pub fn encode_htj2k_codeblock_with_resources(
        &self,
        coefficients: &[i32],
        width: u32,
        height: u32,
        total_bitplanes: u8,
        resources: &CudaHtj2kEncodeResources,
    ) -> Result<CudaHtj2kEncodedCodeBlock, CudaError> {
        let expected_len = checked_image_words(width, height, 1)?;
        if coefficients.len() != expected_len {
            return Err(CudaError::LengthTooLarge {
                len: coefficients.len(),
            });
        }

        self.inner.set_current()?;
        let coefficient_buffer = self.upload_i32_pinned(coefficients)?;
        let output_buffer = self.allocate(HTJ2K_ENCODE_OUTPUT_CAPACITY)?;
        let params = CudaHtj2kEncodeParams {
            width,
            height,
            coefficient_stride: width,
            total_bitplanes: u32::from(total_bitplanes),
            output_capacity: u32::try_from(HTJ2K_ENCODE_OUTPUT_CAPACITY).map_err(|_| {
                CudaError::LengthTooLarge {
                    len: HTJ2K_ENCODE_OUTPUT_CAPACITY,
                }
            })?,
            target_coding_passes: 1,
        };
        let params_buffer = self.upload(htj2k_encode_params_as_bytes(&params))?;
        let initial_status = CudaHtj2kEncodeStatus {
            code: HTJ2K_STATUS_UNSUPPORTED,
            ..CudaHtj2kEncodeStatus::default()
        };
        let status_buffer = self.upload(htj2k_encode_status_as_bytes(&initial_status))?;

        let ((), ht_encode_us) =
            self.time_default_stream_named_us("j2k.htj2k.encode.codeblock", || {
                self.launch_htj2k_encode_codeblock(
                    &coefficient_buffer,
                    &output_buffer,
                    &params_buffer,
                    &resources.vlc_table0,
                    &resources.vlc_table1,
                    &resources.uvlc_table,
                    &status_buffer,
                )
            })?;
        let (status, status_readback_us) = self.time_default_stream_named_us(
            "j2k.htj2k.encode.codeblock.status_readback",
            || {
                let mut status = CudaHtj2kEncodeStatus::default();
                status_buffer.copy_to_host(htj2k_encode_status_as_bytes_mut(&mut status))?;
                if !status.is_ok() {
                    return Err(CudaError::KernelStatus {
                        kernel: "j2k_htj2k_encode_codeblock",
                        code: status.code,
                        detail: status.detail,
                    });
                }
                Ok(status)
            },
        )?;
        let data_len = usize::try_from(status.data_len)
            .map_err(|_| CudaError::LengthTooLarge { len: usize::MAX })?;
        if data_len > HTJ2K_ENCODE_OUTPUT_CAPACITY {
            return Err(CudaError::LengthTooLarge { len: data_len });
        }
        let (data, output_readback_us) = if data_len == 0 {
            (Vec::new(), 0)
        } else {
            self.time_default_stream_named_us("j2k.htj2k.encode.codeblock.output_readback", || {
                let mut data = vec![0u8; data_len];
                output_buffer.copy_range_to_host(0, &mut data)?;
                Ok(data)
            })?
        };
        let stage_timings = CudaHtj2kEncodeStageTimings::from_parts(
            ht_encode_us,
            status_readback_us,
            0,
            output_readback_us,
        );

        Ok(CudaHtj2kEncodedCodeBlock {
            data,
            status,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 0,
                hardware_decode: false,
            },
            stage_timings,
        })
    }

    /// Encode multiple HTJ2K cleanup-pass code blocks with one CUDA dispatch.
    pub fn encode_htj2k_codeblocks(
        &self,
        coefficients: &[i32],
        jobs: &[CudaHtj2kEncodeCodeBlockJob],
        tables: CudaHtj2kEncodeTables<'_>,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        let resources = self.upload_htj2k_encode_resources(tables)?;
        self.encode_htj2k_codeblocks_with_resources(coefficients, jobs, &resources)
    }

    /// Encode multiple HTJ2K cleanup-pass code blocks with pre-uploaded lookup tables.
    pub fn encode_htj2k_codeblocks_with_resources(
        &self,
        coefficients: &[i32],
        jobs: &[CudaHtj2kEncodeCodeBlockJob],
        resources: &CudaHtj2kEncodeResources,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        if jobs.is_empty() {
            return Ok(CudaHtj2kEncodedCodeBlocks {
                code_blocks: Vec::new(),
                execution: CudaExecutionStats::default(),
                stage_timings: CudaHtj2kEncodeStageTimings::default(),
            });
        }

        self.inner.set_current()?;
        let coefficient_buffer = self.upload_i32_pinned(coefficients)?;
        self.encode_htj2k_codeblocks_device_with_resources(
            &coefficient_buffer,
            coefficients.len(),
            jobs,
            resources,
        )
    }

    /// Encode multiple HTJ2K cleanup-pass code blocks from resident quantized coefficients.
    pub fn encode_htj2k_codeblocks_resident(
        &self,
        coefficients: &CudaDeviceBuffer,
        coefficient_count: usize,
        jobs: &[CudaHtj2kEncodeCodeBlockJob],
        tables: CudaHtj2kEncodeTables<'_>,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        let resources = self.upload_htj2k_encode_resources(tables)?;
        self.encode_htj2k_codeblocks_resident_with_resources(
            coefficients,
            coefficient_count,
            jobs,
            &resources,
        )
    }

    /// Encode multiple cleanup-pass code blocks from resident coefficients with lookup table reuse.
    pub fn encode_htj2k_codeblocks_resident_with_resources(
        &self,
        coefficients: &CudaDeviceBuffer,
        coefficient_count: usize,
        jobs: &[CudaHtj2kEncodeCodeBlockJob],
        resources: &CudaHtj2kEncodeResources,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        let pool = self.buffer_pool();
        self.encode_htj2k_codeblocks_resident_with_resources_and_pool(
            coefficients,
            coefficient_count,
            jobs,
            resources,
            &pool,
        )
    }

    /// Encode multiple cleanup-pass code blocks from resident coefficients with
    /// lookup table reuse and caller-owned transient buffer reuse.
    pub fn encode_htj2k_codeblocks_resident_with_resources_and_pool(
        &self,
        coefficients: &CudaDeviceBuffer,
        coefficient_count: usize,
        jobs: &[CudaHtj2kEncodeCodeBlockJob],
        resources: &CudaHtj2kEncodeResources,
        pool: &CudaBufferPool,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        if jobs.is_empty() {
            return Ok(CudaHtj2kEncodedCodeBlocks {
                code_blocks: Vec::new(),
                execution: CudaExecutionStats::default(),
                stage_timings: CudaHtj2kEncodeStageTimings::default(),
            });
        }
        let available_coefficients = coefficients.typed_view::<i32>()?.len();
        if available_coefficients < coefficient_count {
            return Err(CudaError::OutputTooSmall {
                required: coefficient_count
                    .checked_mul(std::mem::size_of::<i32>())
                    .ok_or(CudaError::LengthTooLarge {
                        len: coefficient_count,
                    })?,
                have: coefficients.byte_len(),
            });
        }

        let kernel_jobs = htj2k_encode_kernel_jobs(jobs, coefficient_count)?;
        self.inner.set_current()?;
        self.encode_htj2k_kernel_jobs_device_with_resources_and_pool(
            coefficients,
            &kernel_jobs,
            resources,
            pool,
        )
    }

    /// Encode multiple cleanup-pass code-block batches from independent
    /// resident coefficient buffers with one CUDA dispatch.
    pub fn encode_htj2k_codeblocks_multi_resident_with_resources_and_pool(
        &self,
        targets: &[CudaHtj2kEncodeResidentTarget<'_>],
        resources: &CudaHtj2kEncodeResources,
        pool: &CudaBufferPool,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        self.encode_htj2k_codeblocks_multi_resident_compact_with_resources_and_pool(
            targets, resources, pool,
        )?
        .into_owned_code_blocks()
    }

    /// Encode multiple cleanup-pass code-block batches from independent resident
    /// coefficient buffers with one CUDA dispatch, returning one compact payload
    /// plus per-block ranges.
    pub fn encode_htj2k_codeblocks_multi_resident_compact_with_resources_and_pool(
        &self,
        targets: &[CudaHtj2kEncodeResidentTarget<'_>],
        resources: &CudaHtj2kEncodeResources,
        pool: &CudaBufferPool,
    ) -> Result<CudaHtj2kCompactEncodedCodeBlocks, CudaError> {
        let kernel_jobs = htj2k_encode_multi_input_kernel_jobs(targets)?;
        if kernel_jobs.is_empty() {
            return Ok(CudaHtj2kCompactEncodedCodeBlocks {
                payload: Vec::new(),
                code_blocks: Vec::new(),
                execution: CudaExecutionStats::default(),
                stage_timings: CudaHtj2kEncodeStageTimings::default(),
            });
        }
        self.inner.set_current()?;
        self.encode_htj2k_multi_input_kernel_jobs_device_compact_with_resources_and_pool(
            &kernel_jobs,
            resources,
            pool,
        )
    }

    /// Encode cleanup-pass code blocks from strided resident coefficient regions.
    pub fn encode_htj2k_codeblock_regions_resident(
        &self,
        coefficients: &CudaDeviceBuffer,
        coefficient_count: usize,
        jobs: &[CudaHtj2kEncodeCodeBlockRegionJob],
        tables: CudaHtj2kEncodeTables<'_>,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        let resources = self.upload_htj2k_encode_resources(tables)?;
        self.encode_htj2k_codeblock_regions_resident_with_resources(
            coefficients,
            coefficient_count,
            jobs,
            &resources,
        )
    }

    /// Encode strided resident code-block regions with pre-uploaded lookup tables.
    pub fn encode_htj2k_codeblock_regions_resident_with_resources(
        &self,
        coefficients: &CudaDeviceBuffer,
        coefficient_count: usize,
        jobs: &[CudaHtj2kEncodeCodeBlockRegionJob],
        resources: &CudaHtj2kEncodeResources,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        let pool = self.buffer_pool();
        self.encode_htj2k_codeblock_regions_resident_with_resources_and_pool(
            coefficients,
            coefficient_count,
            jobs,
            resources,
            &pool,
        )
    }

    /// Encode strided resident code-block regions with pre-uploaded lookup
    /// tables and caller-owned transient buffer reuse.
    pub fn encode_htj2k_codeblock_regions_resident_with_resources_and_pool(
        &self,
        coefficients: &CudaDeviceBuffer,
        coefficient_count: usize,
        jobs: &[CudaHtj2kEncodeCodeBlockRegionJob],
        resources: &CudaHtj2kEncodeResources,
        pool: &CudaBufferPool,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        if jobs.is_empty() {
            return Ok(CudaHtj2kEncodedCodeBlocks {
                code_blocks: Vec::new(),
                execution: CudaExecutionStats::default(),
                stage_timings: CudaHtj2kEncodeStageTimings::default(),
            });
        }
        let available_coefficients = coefficients.typed_view::<i32>()?.len();
        if available_coefficients < coefficient_count {
            return Err(CudaError::OutputTooSmall {
                required: coefficient_count
                    .checked_mul(std::mem::size_of::<i32>())
                    .ok_or(CudaError::LengthTooLarge {
                        len: coefficient_count,
                    })?,
                have: coefficients.byte_len(),
            });
        }

        let kernel_jobs = htj2k_encode_region_kernel_jobs(jobs, coefficient_count)?;
        self.inner.set_current()?;
        self.encode_htj2k_kernel_jobs_device_with_resources_and_pool(
            coefficients,
            &kernel_jobs,
            resources,
            pool,
        )
    }

    fn encode_htj2k_codeblocks_device_with_resources(
        &self,
        coefficient_buffer: &CudaDeviceBuffer,
        coefficient_count: usize,
        jobs: &[CudaHtj2kEncodeCodeBlockJob],
        resources: &CudaHtj2kEncodeResources,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        let kernel_jobs = htj2k_encode_kernel_jobs(jobs, coefficient_count)?;
        self.encode_htj2k_kernel_jobs_device_with_resources(
            coefficient_buffer,
            &kernel_jobs,
            resources,
        )
    }

    #[allow(clippy::too_many_lines)]
    fn encode_htj2k_kernel_jobs_device_with_resources(
        &self,
        coefficient_buffer: &CudaDeviceBuffer,
        kernel_jobs: &[CudaHtj2kEncodeKernelJob],
        resources: &CudaHtj2kEncodeResources,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        let pool = self.buffer_pool();
        self.encode_htj2k_kernel_jobs_device_with_resources_and_pool(
            coefficient_buffer,
            kernel_jobs,
            resources,
            &pool,
        )
    }

    #[allow(clippy::too_many_lines)]
    fn encode_htj2k_kernel_jobs_device_with_resources_and_pool(
        &self,
        coefficient_buffer: &CudaDeviceBuffer,
        kernel_jobs: &[CudaHtj2kEncodeKernelJob],
        resources: &CudaHtj2kEncodeResources,
        pool: &CudaBufferPool,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        let output_bytes = kernel_jobs
            .last()
            .map(|job| {
                (job.output_offset as usize)
                    .checked_add(job.output_capacity as usize)
                    .ok_or(CudaError::LengthTooLarge { len: usize::MAX })
            })
            .transpose()?
            .unwrap_or(0);

        let jobs_buffer = pool.upload(htj2k_encode_jobs_as_bytes(kernel_jobs))?;
        let output_buffer = pool.take(output_bytes)?;
        let status_buffer = pool.take(htj2k_encode_statuses_byte_len(kernel_jobs.len())?)?;

        let ((), ht_encode_us) =
            self.time_default_stream_named_us("j2k.htj2k.encode.codeblocks", || {
                self.launch_htj2k_encode_codeblocks(
                    coefficient_buffer,
                    pooled_device_buffer(&output_buffer)?,
                    pooled_device_buffer(&jobs_buffer)?,
                    &resources.vlc_table0,
                    &resources.vlc_table1,
                    &resources.uvlc_table,
                    pooled_device_buffer(&status_buffer)?,
                    kernel_jobs.len(),
                )
            })?;
        let (statuses, status_readback_us) = self.time_default_stream_named_us(
            "j2k.htj2k.encode.codeblocks.status_readback",
            || {
                let mut statuses = vec![CudaHtj2kEncodeStatus::default(); kernel_jobs.len()];
                status_buffer.copy_to_host(htj2k_encode_statuses_as_bytes_mut(&mut statuses))?;
                if let Some(status) = statuses.iter().copied().find(|status| !status.is_ok()) {
                    return Err(CudaError::KernelStatus {
                        kernel: "j2k_htj2k_encode_codeblocks",
                        code: status.code,
                        detail: status.detail,
                    });
                }
                Ok(statuses)
            },
        )?;

        let (compact_jobs, compact_output_bytes) =
            htj2k_encode_compact_jobs(&statuses, kernel_jobs)?;
        let compact_output_buffer = pool.take(compact_output_bytes)?;
        let compact_dispatched = compact_output_bytes != 0;
        let compact_us = if compact_dispatched {
            let compact_jobs_buffer =
                pool.upload(htj2k_encode_compact_jobs_as_bytes(&compact_jobs))?;
            let ((), compact_us) =
                self.time_default_stream_named_us("j2k.htj2k.encode.codeblocks.compact", || {
                    self.launch_htj2k_compact_codeblocks(
                        pooled_device_buffer(&output_buffer)?,
                        pooled_device_buffer(&compact_output_buffer)?,
                        pooled_device_buffer(&compact_jobs_buffer)?,
                        compact_jobs.len(),
                    )
                })?;
            compact_us
        } else {
            0
        };
        let (output, output_readback_us) = if compact_output_bytes == 0 {
            (Vec::new(), 0)
        } else {
            self.time_default_stream_named_us(
                "j2k.htj2k.encode.codeblocks.output_readback",
                || copy_pooled_bytes_to_vec_uninit(&compact_output_buffer, compact_output_bytes),
            )?
        };

        let mut code_blocks = statuses
            .into_iter()
            .zip(kernel_jobs.iter())
            .zip(compact_jobs.iter())
            .map(|((status, job), compact_job)| {
                let data_len = usize::try_from(status.data_len)
                    .map_err(|_| CudaError::LengthTooLarge { len: usize::MAX })?;
                if data_len > job.output_capacity as usize {
                    return Err(CudaError::LengthTooLarge { len: data_len });
                }
                let start = compact_job.compact_offset as usize;
                let end = start
                    .checked_add(data_len)
                    .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
                if end > compact_output_bytes {
                    return Err(CudaError::LengthTooLarge { len: end });
                }
                let data = output[start..end].to_vec();
                Ok(CudaHtj2kEncodedCodeBlock {
                    data,
                    status,
                    execution: CudaExecutionStats {
                        kernel_dispatches: 1,
                        copy_kernel_dispatches: usize::from(compact_dispatched),
                        decode_kernel_dispatches: 0,
                        hardware_decode: false,
                    },
                    stage_timings: CudaHtj2kEncodeStageTimings::default(),
                })
            })
            .collect::<Result<Vec<_>, CudaError>>()?;
        let stage_timings = CudaHtj2kEncodeStageTimings::from_parts(
            ht_encode_us,
            status_readback_us,
            compact_us,
            output_readback_us,
        );
        for block in &mut code_blocks {
            block.stage_timings = stage_timings;
        }
        let copy_kernel_dispatches =
            usize::from(code_blocks.iter().any(|block| !block.data().is_empty()));

        Ok(CudaHtj2kEncodedCodeBlocks {
            code_blocks,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches,
                decode_kernel_dispatches: 0,
                hardware_decode: false,
            },
            stage_timings,
        })
    }

    #[allow(clippy::too_many_lines)]
    fn encode_htj2k_multi_input_kernel_jobs_device_compact_with_resources_and_pool(
        &self,
        kernel_jobs: &[CudaHtj2kEncodeMultiInputKernelJob],
        resources: &CudaHtj2kEncodeResources,
        pool: &CudaBufferPool,
    ) -> Result<CudaHtj2kCompactEncodedCodeBlocks, CudaError> {
        let output_bytes = kernel_jobs
            .last()
            .map(|job| {
                (job.output_offset as usize)
                    .checked_add(job.output_capacity as usize)
                    .ok_or(CudaError::LengthTooLarge { len: usize::MAX })
            })
            .transpose()?
            .unwrap_or(0);

        let jobs_buffer = pool.upload(htj2k_encode_multi_input_jobs_as_bytes(kernel_jobs))?;
        let output_buffer = pool.take(output_bytes)?;
        let status_buffer = pool.take(htj2k_encode_statuses_byte_len(kernel_jobs.len())?)?;
        let cleanup_only = kernel_jobs.iter().all(|job| job.target_coding_passes == 1);
        let cleanup_only_64 = cleanup_only
            && kernel_jobs
                .iter()
                .all(|job| job.width == 64 && job.height == 64 && job.coefficient_stride == 64);
        let status_kernel = if cleanup_only_64 {
            "j2k_htj2k_encode_codeblocks_multi_input_cleanup_64"
        } else if cleanup_only {
            "j2k_htj2k_encode_codeblocks_multi_input_cleanup"
        } else {
            "j2k_htj2k_encode_codeblocks_multi_input"
        };

        let ((), ht_encode_us) =
            self.time_default_stream_named_us("j2k.htj2k.encode.codeblocks.multi_input", || {
                if cleanup_only_64 {
                    self.launch_htj2k_encode_codeblocks_multi_input_cleanup_64(
                        pooled_device_buffer(&output_buffer)?,
                        pooled_device_buffer(&jobs_buffer)?,
                        &resources.vlc_table0,
                        &resources.vlc_table1,
                        &resources.uvlc_table,
                        pooled_device_buffer(&status_buffer)?,
                        kernel_jobs.len(),
                    )
                } else if cleanup_only {
                    self.launch_htj2k_encode_codeblocks_multi_input_cleanup(
                        pooled_device_buffer(&output_buffer)?,
                        pooled_device_buffer(&jobs_buffer)?,
                        &resources.vlc_table0,
                        &resources.vlc_table1,
                        &resources.uvlc_table,
                        pooled_device_buffer(&status_buffer)?,
                        kernel_jobs.len(),
                    )
                } else {
                    self.launch_htj2k_encode_codeblocks_multi_input(
                        pooled_device_buffer(&output_buffer)?,
                        pooled_device_buffer(&jobs_buffer)?,
                        &resources.vlc_table0,
                        &resources.vlc_table1,
                        &resources.uvlc_table,
                        pooled_device_buffer(&status_buffer)?,
                        kernel_jobs.len(),
                    )
                }
            })?;
        let (statuses, status_readback_us) = self.time_default_stream_named_us(
            "j2k.htj2k.encode.codeblocks.multi_input.status_readback",
            || {
                let mut statuses = vec![CudaHtj2kEncodeStatus::default(); kernel_jobs.len()];
                status_buffer.copy_to_host(htj2k_encode_statuses_as_bytes_mut(&mut statuses))?;
                if let Some(status) = statuses.iter().copied().find(|status| !status.is_ok()) {
                    return Err(CudaError::KernelStatus {
                        kernel: status_kernel,
                        code: status.code,
                        detail: status.detail,
                    });
                }
                Ok(statuses)
            },
        )?;

        let (compact_jobs, compact_output_bytes) =
            htj2k_encode_compact_jobs_multi_input(&statuses, kernel_jobs)?;
        let compact_output_buffer = pool.take(compact_output_bytes)?;
        let compact_dispatched = compact_output_bytes != 0;
        let compact_us = if compact_dispatched {
            let compact_jobs_buffer =
                pool.upload(htj2k_encode_compact_jobs_as_bytes(&compact_jobs))?;
            let ((), compact_us) = self.time_default_stream_named_us(
                "j2k.htj2k.encode.codeblocks.multi_input.compact",
                || {
                    self.launch_htj2k_compact_codeblocks(
                        pooled_device_buffer(&output_buffer)?,
                        pooled_device_buffer(&compact_output_buffer)?,
                        pooled_device_buffer(&compact_jobs_buffer)?,
                        compact_jobs.len(),
                    )
                },
            )?;
            compact_us
        } else {
            0
        };
        let (output, output_readback_us) = if compact_output_bytes == 0 {
            (Vec::new(), 0)
        } else {
            self.time_default_stream_named_us(
                "j2k.htj2k.encode.codeblocks.multi_input.output_readback",
                || copy_pooled_bytes_to_vec_uninit(&compact_output_buffer, compact_output_bytes),
            )?
        };

        let mut code_blocks = statuses
            .into_iter()
            .zip(kernel_jobs.iter())
            .zip(compact_jobs.iter())
            .map(|((status, job), compact_job)| {
                let data_len = usize::try_from(status.data_len)
                    .map_err(|_| CudaError::LengthTooLarge { len: usize::MAX })?;
                if data_len > job.output_capacity as usize {
                    return Err(CudaError::LengthTooLarge { len: data_len });
                }
                let start = compact_job.compact_offset as usize;
                let end = start
                    .checked_add(data_len)
                    .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
                if end > compact_output_bytes {
                    return Err(CudaError::LengthTooLarge { len: end });
                }
                Ok(CudaHtj2kCompactEncodedCodeBlock {
                    payload_range: start..end,
                    status,
                    execution: CudaExecutionStats {
                        kernel_dispatches: 1,
                        copy_kernel_dispatches: usize::from(compact_dispatched),
                        decode_kernel_dispatches: 0,
                        hardware_decode: false,
                    },
                    stage_timings: CudaHtj2kEncodeStageTimings::default(),
                })
            })
            .collect::<Result<Vec<_>, CudaError>>()?;
        let stage_timings = CudaHtj2kEncodeStageTimings::from_parts(
            ht_encode_us,
            status_readback_us,
            compact_us,
            output_readback_us,
        );
        for block in &mut code_blocks {
            block.stage_timings = stage_timings;
        }
        let copy_kernel_dispatches = usize::from(!output.is_empty());

        Ok(CudaHtj2kCompactEncodedCodeBlocks {
            payload: output,
            code_blocks,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches,
                decode_kernel_dispatches: 0,
                hardware_decode: false,
            },
            stage_timings,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_htj2k_encode_codeblock(
        &self,
        coefficients: &CudaDeviceBuffer,
        output: &CudaDeviceBuffer,
        params_buffer: &CudaDeviceBuffer,
        vlc_table0: &CudaDeviceBuffer,
        vlc_table1: &CudaDeviceBuffer,
        uvlc_table: &CudaDeviceBuffer,
        status: &CudaDeviceBuffer,
    ) -> Result<(), CudaError> {
        let function = self
            .inner
            .kernel_function(CudaKernel::Htj2kEncodeCodeblock)?;
        let mut coefficients_ptr = coefficients.device_ptr();
        let mut output_ptr = output.device_ptr();
        let mut params_ptr = params_buffer.device_ptr();
        let mut vlc_table0_ptr = vlc_table0.device_ptr();
        let mut vlc_table1_ptr = vlc_table1.device_ptr();
        let mut uvlc_table_ptr = uvlc_table.device_ptr();
        let mut status_ptr = status.device_ptr();
        let mut params = cuda_kernel_params!(
            coefficients_ptr,
            output_ptr,
            params_ptr,
            vlc_table0_ptr,
            vlc_table1_ptr,
            uvlc_table_ptr,
            status_ptr
        );
        let geometry = htj2k_encode_codeblock_launch_geometry(1)
            .ok_or(CudaError::LengthTooLarge { len: 1 })?;
        self.launch_kernel_async(function, geometry, &mut params)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_htj2k_encode_codeblocks(
        &self,
        coefficients: &CudaDeviceBuffer,
        output: &CudaDeviceBuffer,
        jobs: &CudaDeviceBuffer,
        vlc_table0: &CudaDeviceBuffer,
        vlc_table1: &CudaDeviceBuffer,
        uvlc_table: &CudaDeviceBuffer,
        statuses: &CudaDeviceBuffer,
        job_count: usize,
    ) -> Result<(), CudaError> {
        let function = self
            .inner
            .kernel_function(CudaKernel::Htj2kEncodeCodeblocks)?;
        let mut coefficients_ptr = coefficients.device_ptr();
        let mut output_ptr = output.device_ptr();
        let mut jobs_ptr = jobs.device_ptr();
        let mut vlc_table0_ptr = vlc_table0.device_ptr();
        let mut vlc_table1_ptr = vlc_table1.device_ptr();
        let mut uvlc_table_ptr = uvlc_table.device_ptr();
        let mut statuses_ptr = statuses.device_ptr();
        let mut job_count_u64 =
            u64::try_from(job_count).map_err(|_| CudaError::LengthTooLarge { len: job_count })?;
        let mut params = cuda_kernel_params!(
            coefficients_ptr,
            output_ptr,
            jobs_ptr,
            vlc_table0_ptr,
            vlc_table1_ptr,
            uvlc_table_ptr,
            statuses_ptr,
            job_count_u64
        );
        let geometry = htj2k_encode_codeblock_launch_geometry(job_count)
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;
        self.launch_kernel_async(function, geometry, &mut params)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_htj2k_encode_codeblocks_multi_input(
        &self,
        output: &CudaDeviceBuffer,
        jobs: &CudaDeviceBuffer,
        vlc_table0: &CudaDeviceBuffer,
        vlc_table1: &CudaDeviceBuffer,
        uvlc_table: &CudaDeviceBuffer,
        statuses: &CudaDeviceBuffer,
        job_count: usize,
    ) -> Result<(), CudaError> {
        let function = self
            .inner
            .kernel_function(CudaKernel::Htj2kEncodeCodeblocksMultiInput)?;
        let mut output_ptr = output.device_ptr();
        let mut jobs_ptr = jobs.device_ptr();
        let mut vlc_table0_ptr = vlc_table0.device_ptr();
        let mut vlc_table1_ptr = vlc_table1.device_ptr();
        let mut uvlc_table_ptr = uvlc_table.device_ptr();
        let mut statuses_ptr = statuses.device_ptr();
        let mut job_count_u64 =
            u64::try_from(job_count).map_err(|_| CudaError::LengthTooLarge { len: job_count })?;
        let mut params = cuda_kernel_params!(
            output_ptr,
            jobs_ptr,
            vlc_table0_ptr,
            vlc_table1_ptr,
            uvlc_table_ptr,
            statuses_ptr,
            job_count_u64
        );
        let geometry = htj2k_encode_codeblock_launch_geometry(job_count)
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;
        self.launch_kernel_async(function, geometry, &mut params)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_htj2k_encode_codeblocks_multi_input_cleanup(
        &self,
        output: &CudaDeviceBuffer,
        jobs: &CudaDeviceBuffer,
        vlc_table0: &CudaDeviceBuffer,
        vlc_table1: &CudaDeviceBuffer,
        uvlc_table: &CudaDeviceBuffer,
        statuses: &CudaDeviceBuffer,
        job_count: usize,
    ) -> Result<(), CudaError> {
        let function = self
            .inner
            .kernel_function(CudaKernel::Htj2kEncodeCodeblocksMultiInputCleanup)?;
        let mut output_ptr = output.device_ptr();
        let mut jobs_ptr = jobs.device_ptr();
        let mut vlc_table0_ptr = vlc_table0.device_ptr();
        let mut vlc_table1_ptr = vlc_table1.device_ptr();
        let mut uvlc_table_ptr = uvlc_table.device_ptr();
        let mut statuses_ptr = statuses.device_ptr();
        let mut job_count_u64 =
            u64::try_from(job_count).map_err(|_| CudaError::LengthTooLarge { len: job_count })?;
        let mut params = cuda_kernel_params!(
            output_ptr,
            jobs_ptr,
            vlc_table0_ptr,
            vlc_table1_ptr,
            uvlc_table_ptr,
            statuses_ptr,
            job_count_u64
        );
        let geometry = htj2k_encode_codeblock_launch_geometry(job_count)
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;
        self.launch_kernel_async(function, geometry, &mut params)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_htj2k_encode_codeblocks_multi_input_cleanup_64(
        &self,
        output: &CudaDeviceBuffer,
        jobs: &CudaDeviceBuffer,
        vlc_table0: &CudaDeviceBuffer,
        vlc_table1: &CudaDeviceBuffer,
        uvlc_table: &CudaDeviceBuffer,
        statuses: &CudaDeviceBuffer,
        job_count: usize,
    ) -> Result<(), CudaError> {
        let function = self
            .inner
            .kernel_function(CudaKernel::Htj2kEncodeCodeblocksMultiInputCleanup64)?;
        let mut output_ptr = output.device_ptr();
        let mut jobs_ptr = jobs.device_ptr();
        let mut vlc_table0_ptr = vlc_table0.device_ptr();
        let mut vlc_table1_ptr = vlc_table1.device_ptr();
        let mut uvlc_table_ptr = uvlc_table.device_ptr();
        let mut statuses_ptr = statuses.device_ptr();
        let mut job_count_u64 =
            u64::try_from(job_count).map_err(|_| CudaError::LengthTooLarge { len: job_count })?;
        let mut params = cuda_kernel_params!(
            output_ptr,
            jobs_ptr,
            vlc_table0_ptr,
            vlc_table1_ptr,
            uvlc_table_ptr,
            statuses_ptr,
            job_count_u64
        );
        let geometry = htj2k_encode_codeblock_launch_geometry(job_count)
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;
        self.launch_kernel_async(function, geometry, &mut params)
    }

    fn launch_htj2k_compact_codeblocks(
        &self,
        scratch: &CudaDeviceBuffer,
        compact: &CudaDeviceBuffer,
        jobs: &CudaDeviceBuffer,
        job_count: usize,
    ) -> Result<(), CudaError> {
        let function = self
            .inner
            .kernel_function(CudaKernel::Htj2kCompactCodeblocks)?;
        let mut scratch_ptr = scratch.device_ptr();
        let mut compact_ptr = compact.device_ptr();
        let mut jobs_ptr = jobs.device_ptr();
        let mut job_count_u64 =
            u64::try_from(job_count).map_err(|_| CudaError::LengthTooLarge { len: job_count })?;
        let mut params = cuda_kernel_params!(scratch_ptr, compact_ptr, jobs_ptr, job_count_u64);
        let geometry = htj2k_codeblock_sample_launch_geometry(job_count)
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;
        self.launch_kernel_async(function, geometry, &mut params)
    }
}

pub(crate) fn htj2k_encode_kernel_jobs(
    jobs: &[CudaHtj2kEncodeCodeBlockJob],
    coefficient_words: usize,
) -> Result<Vec<CudaHtj2kEncodeKernelJob>, CudaError> {
    let mut output_offset = 0usize;
    let mut kernel_jobs = Vec::with_capacity(jobs.len());
    for job in jobs {
        validate_htj2k_encode_codeblock_shape(job.width, job.height)?;
        let coefficient_offset = job.coefficient_offset as usize;
        let coefficient_len = checked_image_words(job.width, job.height, 1)?;
        let coefficient_end =
            coefficient_offset
                .checked_add(coefficient_len)
                .ok_or(CudaError::LengthTooLarge {
                    len: coefficient_words,
                })?;
        if coefficient_end > coefficient_words {
            return Err(CudaError::LengthTooLarge {
                len: coefficient_end,
            });
        }

        let output_end = output_offset
            .checked_add(HTJ2K_ENCODE_OUTPUT_CAPACITY)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        if output_end > u32::MAX as usize {
            return Err(CudaError::LengthTooLarge { len: output_end });
        }
        kernel_jobs.push(CudaHtj2kEncodeKernelJob {
            coefficient_offset: job.coefficient_offset,
            coefficient_stride: job.width,
            width: job.width,
            height: job.height,
            total_bitplanes: u32::from(job.total_bitplanes),
            output_offset: u32::try_from(output_offset)
                .map_err(|_| CudaError::LengthTooLarge { len: output_offset })?,
            output_capacity: u32::try_from(HTJ2K_ENCODE_OUTPUT_CAPACITY).map_err(|_| {
                CudaError::LengthTooLarge {
                    len: HTJ2K_ENCODE_OUTPUT_CAPACITY,
                }
            })?,
            target_coding_passes: u32::from(job.target_coding_passes),
        });
        output_offset = output_end;
    }
    Ok(kernel_jobs)
}

pub(crate) fn htj2k_encode_multi_input_kernel_jobs(
    targets: &[CudaHtj2kEncodeResidentTarget<'_>],
) -> Result<Vec<CudaHtj2kEncodeMultiInputKernelJob>, CudaError> {
    let job_count = targets
        .iter()
        .try_fold(0usize, |sum, target| sum.checked_add(target.jobs.len()))
        .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
    let mut output_offset = 0usize;
    let mut kernel_jobs = Vec::with_capacity(job_count);
    for target in targets {
        let available_coefficients = target.coefficients.typed_view::<i32>()?.len();
        if available_coefficients < target.coefficient_count {
            return Err(CudaError::OutputTooSmall {
                required: target
                    .coefficient_count
                    .checked_mul(std::mem::size_of::<i32>())
                    .ok_or(CudaError::LengthTooLarge {
                        len: target.coefficient_count,
                    })?,
                have: target.coefficients.byte_len(),
            });
        }
        for job in target.jobs {
            validate_htj2k_encode_codeblock_shape(job.width, job.height)?;
            let coefficient_offset = job.coefficient_offset as usize;
            let coefficient_len = checked_image_words(job.width, job.height, 1)?;
            let coefficient_end = coefficient_offset.checked_add(coefficient_len).ok_or(
                CudaError::LengthTooLarge {
                    len: target.coefficient_count,
                },
            )?;
            if coefficient_end > target.coefficient_count {
                return Err(CudaError::LengthTooLarge {
                    len: coefficient_end,
                });
            }

            let output_end = output_offset
                .checked_add(HTJ2K_ENCODE_OUTPUT_CAPACITY)
                .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
            if output_end > u32::MAX as usize {
                return Err(CudaError::LengthTooLarge { len: output_end });
            }
            kernel_jobs.push(CudaHtj2kEncodeMultiInputKernelJob {
                coefficient_ptr: target.coefficients.device_ptr(),
                coefficient_offset: job.coefficient_offset,
                coefficient_stride: job.width,
                width: job.width,
                height: job.height,
                total_bitplanes: u32::from(job.total_bitplanes),
                output_offset: u32::try_from(output_offset)
                    .map_err(|_| CudaError::LengthTooLarge { len: output_offset })?,
                output_capacity: u32::try_from(HTJ2K_ENCODE_OUTPUT_CAPACITY).map_err(|_| {
                    CudaError::LengthTooLarge {
                        len: HTJ2K_ENCODE_OUTPUT_CAPACITY,
                    }
                })?,
                target_coding_passes: u32::from(job.target_coding_passes),
            });
            output_offset = output_end;
        }
    }
    Ok(kernel_jobs)
}

pub(crate) fn htj2k_encode_region_kernel_jobs(
    jobs: &[CudaHtj2kEncodeCodeBlockRegionJob],
    coefficient_words: usize,
) -> Result<Vec<CudaHtj2kEncodeKernelJob>, CudaError> {
    let mut output_offset = 0usize;
    let mut kernel_jobs = Vec::with_capacity(jobs.len());
    for job in jobs {
        validate_htj2k_encode_codeblock_shape(job.width, job.height)?;
        if job.width == 0 || job.height == 0 || job.coefficient_stride < job.width {
            return Err(CudaError::LengthTooLarge {
                len: coefficient_words,
            });
        }
        let row_offset = (job.height as usize - 1)
            .checked_mul(job.coefficient_stride as usize)
            .ok_or(CudaError::LengthTooLarge {
                len: coefficient_words,
            })?;
        let coefficient_end = job
            .coefficient_offset
            .try_into()
            .ok()
            .and_then(|offset: usize| offset.checked_add(row_offset))
            .and_then(|offset| offset.checked_add(job.width as usize))
            .ok_or(CudaError::LengthTooLarge {
                len: coefficient_words,
            })?;
        if coefficient_end > coefficient_words {
            return Err(CudaError::LengthTooLarge {
                len: coefficient_end,
            });
        }

        let output_end = output_offset
            .checked_add(HTJ2K_ENCODE_OUTPUT_CAPACITY)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        if output_end > u32::MAX as usize {
            return Err(CudaError::LengthTooLarge { len: output_end });
        }
        kernel_jobs.push(CudaHtj2kEncodeKernelJob {
            coefficient_offset: job.coefficient_offset,
            coefficient_stride: job.coefficient_stride,
            width: job.width,
            height: job.height,
            total_bitplanes: u32::from(job.total_bitplanes),
            output_offset: u32::try_from(output_offset)
                .map_err(|_| CudaError::LengthTooLarge { len: output_offset })?,
            output_capacity: u32::try_from(HTJ2K_ENCODE_OUTPUT_CAPACITY).map_err(|_| {
                CudaError::LengthTooLarge {
                    len: HTJ2K_ENCODE_OUTPUT_CAPACITY,
                }
            })?,
            target_coding_passes: u32::from(job.target_coding_passes),
        });
        output_offset = output_end;
    }
    Ok(kernel_jobs)
}

pub(crate) fn htj2k_encode_compact_jobs(
    statuses: &[CudaHtj2kEncodeStatus],
    kernel_jobs: &[CudaHtj2kEncodeKernelJob],
) -> Result<(Vec<CudaHtj2kEncodeCompactJob>, usize), CudaError> {
    if statuses.len() != kernel_jobs.len() {
        return Err(CudaError::InvalidArgument {
            message: "HTJ2K encode status count does not match job count".to_string(),
        });
    }

    let mut compact_offset = 0usize;
    let mut compact_jobs = Vec::with_capacity(kernel_jobs.len());
    for (status, job) in statuses.iter().zip(kernel_jobs) {
        let data_len = usize::try_from(status.data_len)
            .map_err(|_| CudaError::LengthTooLarge { len: usize::MAX })?;
        if data_len > job.output_capacity as usize {
            return Err(CudaError::LengthTooLarge { len: data_len });
        }
        let source_end = (job.output_offset as usize)
            .checked_add(data_len)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        let job_output_end = (job.output_offset as usize)
            .checked_add(job.output_capacity as usize)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        if source_end > job_output_end {
            return Err(CudaError::LengthTooLarge { len: source_end });
        }
        compact_jobs.push(CudaHtj2kEncodeCompactJob {
            source_offset: job.output_offset,
            compact_offset: u32::try_from(compact_offset).map_err(|_| {
                CudaError::LengthTooLarge {
                    len: compact_offset,
                }
            })?,
            data_len: status.data_len,
            reserved: status.reserved2,
        });
        compact_offset = compact_offset
            .checked_add(data_len)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        if compact_offset > u32::MAX as usize {
            return Err(CudaError::LengthTooLarge {
                len: compact_offset,
            });
        }
    }

    Ok((compact_jobs, compact_offset))
}

pub(crate) fn htj2k_encode_compact_jobs_multi_input(
    statuses: &[CudaHtj2kEncodeStatus],
    kernel_jobs: &[CudaHtj2kEncodeMultiInputKernelJob],
) -> Result<(Vec<CudaHtj2kEncodeCompactJob>, usize), CudaError> {
    if statuses.len() != kernel_jobs.len() {
        return Err(CudaError::InvalidArgument {
            message: "HTJ2K encode status count does not match job count".to_string(),
        });
    }

    let mut compact_offset = 0usize;
    let mut compact_jobs = Vec::with_capacity(kernel_jobs.len());
    for (status, job) in statuses.iter().zip(kernel_jobs) {
        let data_len = usize::try_from(status.data_len)
            .map_err(|_| CudaError::LengthTooLarge { len: usize::MAX })?;
        if data_len > job.output_capacity as usize {
            return Err(CudaError::LengthTooLarge { len: data_len });
        }
        let source_end = (job.output_offset as usize)
            .checked_add(data_len)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        let job_output_end = (job.output_offset as usize)
            .checked_add(job.output_capacity as usize)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        if source_end > job_output_end {
            return Err(CudaError::LengthTooLarge { len: source_end });
        }
        compact_jobs.push(CudaHtj2kEncodeCompactJob {
            source_offset: job.output_offset,
            compact_offset: u32::try_from(compact_offset).map_err(|_| {
                CudaError::LengthTooLarge {
                    len: compact_offset,
                }
            })?,
            data_len: status.data_len,
            reserved: status.reserved2,
        });
        compact_offset = compact_offset
            .checked_add(data_len)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        if compact_offset > u32::MAX as usize {
            return Err(CudaError::LengthTooLarge {
                len: compact_offset,
            });
        }
    }

    Ok((compact_jobs, compact_offset))
}

pub(crate) fn validate_htj2k_encode_codeblock_shape(
    width: u32,
    height: u32,
) -> Result<(), CudaError> {
    let samples = usize::try_from(width)
        .ok()
        .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
        .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
    if width == 0
        || height == 0
        || width > HTJ2K_ENCODE_MAX_CODEBLOCK_WIDTH
        || samples > HTJ2K_ENCODE_MAX_CODEBLOCK_SAMPLES
    {
        return Err(CudaError::InvalidArgument {
            message: "HTJ2K encode code-block dimensions exceed CUDA kernel limits".to_string(),
        });
    }
    Ok(())
}
