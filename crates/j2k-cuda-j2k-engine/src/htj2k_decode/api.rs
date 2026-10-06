// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::{
    bytes::u16_slice_as_bytes,
    error::CudaError,
    execution::CudaExecutionStats,
    memory::{pooled_device_buffer, CudaBufferPool},
};

use super::{
    output_regions::validate_htj2k_output_layout,
    types::{
        CudaHtj2kCleanupTarget, CudaHtj2kCodeBlockJob, CudaHtj2kCoefficientClearKernelBatch,
        CudaHtj2kCoefficientClearKernelTarget, CudaHtj2kDecodeOutput, CudaHtj2kDecodePayload,
        CudaHtj2kDecodeResources, CudaHtj2kDecodeStageTimings, CudaHtj2kDecodeTableResourceInner,
        CudaHtj2kDecodeTableResources, CudaHtj2kDecodeTables, CudaPooledHtj2kDecodeOutput,
        HTJ2K_COEFFICIENT_CLEAR_TARGETS_PER_BATCH,
    },
};

/// Threads per coefficient-clear block; the kernel strides by `blockDim`.
const COEFFICIENT_CLEAR_THREADS: u32 = 256;
/// Upper bound on blocks clearing one target.
const MAX_COEFFICIENT_CLEAR_BLOCKS_PER_TARGET: u32 = 32;

impl crate::J2kCudaEngine<'_> {
    /// Enqueue zero fills for disjoint coefficient planes in batches of one CUDA launch.
    /// Target job slices describe the later cleanup launch and are not inspected
    /// or retained by this operation.
    ///
    /// # Safety
    ///
    /// Every target allocation must remain live until the context completes
    /// the queued work. All reads, writes, and pool reuse must be ordered on
    /// this context's default stream until that completion point; only then
    /// may the pool hand a target to an upload on its separate upload stream.
    #[doc(hidden)]
    pub unsafe fn clear_htj2k_coefficients_multi_enqueue(
        &self,
        targets: &[CudaHtj2kCleanupTarget<'_>],
    ) -> Result<CudaExecutionStats, CudaError> {
        for target in targets {
            let required = target
                .output_words
                .checked_mul(std::mem::size_of::<u32>())
                .ok_or(CudaError::LengthTooLarge {
                    len: target.output_words,
                })?;
            if !target.coefficients.is_owned_by(self.context) {
                return Err(CudaError::InvalidArgument {
                    message: "HTJ2K coefficient clear target belongs to another context"
                        .to_string(),
                });
            }
            if required > target.coefficients.byte_len() {
                return Err(CudaError::OutputTooSmall {
                    required,
                    have: target.coefficients.byte_len(),
                });
            }
        }
        self.context
            .validate_disjoint_device_buffers(targets.iter().map(|target| target.coefficients))?;
        if targets.is_empty() {
            return Ok(CudaExecutionStats::default());
        }
        self.prepare_operation()?;
        let mut dispatches = 0usize;
        for targets in targets.chunks(HTJ2K_COEFFICIENT_CLEAR_TARGETS_PER_BATCH) {
            let mut kernel_targets = [CudaHtj2kCoefficientClearKernelTarget::default();
                HTJ2K_COEFFICIENT_CLEAR_TARGETS_PER_BATCH];
            let mut max_words = 0u64;
            for (output, target) in kernel_targets.iter_mut().zip(targets) {
                let words =
                    u64::try_from(target.output_words).map_err(|_| CudaError::LengthTooLarge {
                        len: target.output_words,
                    })?;
                *output = CudaHtj2kCoefficientClearKernelTarget {
                    output_ptr: target.coefficients.device_ptr(),
                    words,
                };
                max_words = max_words.max(words);
            }
            let blocks_per_target =
                u32::try_from(max_words.div_ceil(u64::from(COEFFICIENT_CLEAR_THREADS)))
                    .map_or(MAX_COEFFICIENT_CLEAR_BLOCKS_PER_TARGET, |blocks| {
                        blocks.clamp(1, MAX_COEFFICIENT_CLEAR_BLOCKS_PER_TARGET)
                    });
            let target_count = u32::try_from(targets.len())
                .map_err(|_| CudaError::LengthTooLarge { len: targets.len() })?;
            let geometry = crate::kernels::CudaLaunchGeometry::new(
                (blocks_per_target, target_count, 1),
                (COEFFICIENT_CLEAR_THREADS, 1, 1),
            )
            .ok_or(CudaError::LengthTooLarge { len: targets.len() })?;
            let batch = CudaHtj2kCoefficientClearKernelBatch {
                targets: kernel_targets,
                target_count,
                blocks_per_target,
            };
            if let Err(error) = self.launch_htj2k_coefficient_clear(batch, geometry) {
                return self.context.synchronize_then_error(error);
            }
            dispatches = dispatches.saturating_add(1);
        }
        Ok(CudaExecutionStats::new(dispatches, 0, 0, false))
    }

    fn decode_empty_htj2k_codeblocks(
        &self,
        jobs: &[CudaHtj2kCodeBlockJob],
        output_words: usize,
    ) -> Result<CudaHtj2kDecodeOutput, CudaError> {
        self.prepare_operation()?;
        let output_bytes = output_words
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaError::LengthTooLarge { len: output_words })?;
        let coefficients = self.allocate(output_bytes)?;
        if super::planning::htj2k_decode_needs_zero_fill(jobs, output_words)? {
            self.memset_d32(&coefficients, 0, output_words)?;
            self.synchronize()?;
        }
        Ok(CudaHtj2kDecodeOutput {
            coefficients,
            execution: CudaExecutionStats::default(),
            statuses: Vec::new(),
            stage_timings: CudaHtj2kDecodeStageTimings::default(),
        })
    }

    /// Decode HTJ2K code blocks into a device-resident f32 coefficient plane.
    #[doc(hidden)]
    pub fn decode_htj2k_codeblocks(
        &self,
        payload: &[u8],
        jobs: &[CudaHtj2kCodeBlockJob],
        tables: CudaHtj2kDecodeTables<'_>,
        output_words: usize,
    ) -> Result<CudaHtj2kDecodeOutput, CudaError> {
        if jobs.is_empty() {
            return self.decode_empty_htj2k_codeblocks(jobs, output_words);
        }
        let resources = self.upload_htj2k_decode_resources(payload, tables)?;
        self.decode_htj2k_codeblocks_with_resources(&resources, jobs, output_words)
    }

    /// Upload HTJ2K decode payload and lookup tables once for reuse by sub-band dispatches.
    fn upload_htj2k_decode_resources(
        &self,
        payload: &[u8],
        tables: CudaHtj2kDecodeTables<'_>,
    ) -> Result<CudaHtj2kDecodeResources, CudaError> {
        let tables = self.upload_htj2k_decode_table_resources(tables)?;
        self.upload_htj2k_decode_resources_with_tables(payload, &tables)
    }

    /// Upload static HTJ2K cleanup decode lookup tables once for reuse.
    #[doc(hidden)]
    pub fn upload_htj2k_decode_table_resources(
        &self,
        tables: CudaHtj2kDecodeTables<'_>,
    ) -> Result<CudaHtj2kDecodeTableResources, CudaError> {
        self.prepare_operation()?;
        Ok(CudaHtj2kDecodeTableResources {
            inner: Arc::new(CudaHtj2kDecodeTableResourceInner {
                vlc_table0: self.upload(u16_slice_as_bytes(tables.vlc_table0))?,
                vlc_table1: self.upload(u16_slice_as_bytes(tables.vlc_table1))?,
                uvlc_table0: self.upload(u16_slice_as_bytes(tables.uvlc_table0))?,
                uvlc_table1: self.upload(u16_slice_as_bytes(tables.uvlc_table1))?,
            }),
        })
    }

    /// Upload an HTJ2K decode payload while reusing already resident cleanup tables.
    #[doc(hidden)]
    pub fn upload_htj2k_decode_resources_with_tables(
        &self,
        payload: &[u8],
        tables: &CudaHtj2kDecodeTableResources,
    ) -> Result<CudaHtj2kDecodeResources, CudaError> {
        if !tables.is_owned_by(self.context) {
            return Err(CudaError::InvalidArgument {
                message: "HTJ2K decode tables must belong to the upload context".to_string(),
            });
        }
        self.prepare_operation()?;
        Ok(CudaHtj2kDecodeResources {
            payload: CudaHtj2kDecodePayload::Owned(self.upload_pinned(payload)?),
            payload_len: payload.len(),
            tables: Some(tables.clone()),
        })
    }

    /// Upload a classic-only J2K payload without HTJ2K lookup tables.
    #[doc(hidden)]
    pub fn upload_j2k_decode_payload(
        &self,
        payload: &[u8],
    ) -> Result<CudaHtj2kDecodeResources, CudaError> {
        self.prepare_operation()?;
        Ok(CudaHtj2kDecodeResources {
            payload: CudaHtj2kDecodePayload::Owned(self.upload_pinned(payload)?),
            payload_len: payload.len(),
            tables: None,
        })
    }

    /// Upload HTJ2K decode payload parts, back to back, into a pooled buffer
    /// while reusing already resident cleanup tables.
    #[doc(hidden)]
    pub fn upload_htj2k_decode_resources_with_tables_and_pool(
        &self,
        payload_parts: &[&[u8]],
        tables: &CudaHtj2kDecodeTableResources,
        pool: &CudaBufferPool,
    ) -> Result<CudaHtj2kDecodeResources, CudaError> {
        if !tables.is_owned_by(self.context) || !pool.is_owned_by(self.context) {
            return Err(CudaError::InvalidArgument {
                message: "HTJ2K decode tables and pool must belong to the upload context"
                    .to_string(),
            });
        }
        self.prepare_operation()?;
        let payload = pool.upload_pinned_parts_enqueue(payload_parts)?;
        Ok(CudaHtj2kDecodeResources {
            payload: CudaHtj2kDecodePayload::Pooled(payload),
            // The upload already rejected an overflowing total.
            payload_len: payload_parts.iter().map(|part| part.len()).sum(),
            tables: Some(tables.clone()),
        })
    }

    /// Upload classic-only J2K payload parts, back to back, into a pooled
    /// buffer without HTJ2K lookup tables.
    #[doc(hidden)]
    pub fn upload_j2k_decode_payload_with_pool(
        &self,
        payload_parts: &[&[u8]],
        pool: &CudaBufferPool,
    ) -> Result<CudaHtj2kDecodeResources, CudaError> {
        if !pool.is_owned_by(self.context) {
            return Err(CudaError::InvalidArgument {
                message: "J2K decode payload pool must belong to the upload context".to_string(),
            });
        }
        self.prepare_operation()?;
        let payload = pool.upload_pinned_parts_enqueue(payload_parts)?;
        Ok(CudaHtj2kDecodeResources {
            payload: CudaHtj2kDecodePayload::Pooled(payload),
            // The upload already rejected an overflowing total.
            payload_len: payload_parts.iter().map(|part| part.len()).sum(),
            tables: None,
        })
    }

    /// Decode HTJ2K code blocks using already resident payload and lookup tables.
    pub(crate) fn decode_htj2k_codeblocks_with_resources(
        &self,
        resources: &CudaHtj2kDecodeResources,
        jobs: &[CudaHtj2kCodeBlockJob],
        output_words: usize,
    ) -> Result<CudaHtj2kDecodeOutput, CudaError> {
        self.decode_htj2k_codeblocks_with_resources_impl(resources, jobs, output_words, true)
    }

    /// Allocate and initialize an HTJ2K coefficient output buffer without
    /// launching entropy cleanup decode. This is used when cleanup work is
    /// batched across multiple output buffers.
    #[doc(hidden)]
    pub fn allocate_htj2k_codeblock_coefficients_with_pool(
        &self,
        jobs: &[CudaHtj2kCodeBlockJob],
        output_words: usize,
        pool: &CudaBufferPool,
    ) -> Result<CudaPooledHtj2kDecodeOutput, CudaError> {
        if !pool.is_owned_by(self.context) {
            return Err(CudaError::InvalidArgument {
                message: "HTJ2K coefficient pool must belong to the allocation context".to_string(),
            });
        }
        let output_layout = validate_htj2k_output_layout(jobs, output_words)?;
        self.prepare_operation()?;
        let coefficients = pool.take(output_layout.output_bytes)?;
        let coefficient_buffer = pooled_device_buffer(&coefficients)?;
        if output_layout.needs_zero_fill {
            self.memset_d32_async(coefficient_buffer, 0, output_words)?;
        }
        Ok(CudaPooledHtj2kDecodeOutput {
            coefficients,
            execution: CudaExecutionStats::default(),
            statuses: Vec::new(),
            stage_timings: CudaHtj2kDecodeStageTimings::default(),
        })
    }

    /// Allocate an HTJ2K coefficient plane and defer any required zero fill.
    #[doc(hidden)]
    pub fn allocate_htj2k_codeblock_coefficients_deferred_clear_with_pool(
        &self,
        jobs: &[CudaHtj2kCodeBlockJob],
        output_words: usize,
        pool: &CudaBufferPool,
    ) -> Result<(CudaPooledHtj2kDecodeOutput, bool), CudaError> {
        if !pool.is_owned_by(self.context) {
            return Err(CudaError::InvalidArgument {
                message: "HTJ2K coefficient pool must belong to the allocation context".to_string(),
            });
        }
        let output_layout = validate_htj2k_output_layout(jobs, output_words)?;
        let coefficients = pool.take(output_layout.output_bytes)?;
        Ok((
            CudaPooledHtj2kDecodeOutput {
                coefficients,
                execution: CudaExecutionStats::default(),
                statuses: Vec::new(),
                stage_timings: CudaHtj2kDecodeStageTimings::default(),
            },
            output_layout.needs_zero_fill,
        ))
    }
}
