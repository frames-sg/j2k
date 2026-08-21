// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    allocation::HostPhaseBudget,
    error::CudaError,
    execution::CudaExecutionStats,
    htj2k_encode::{
        htj2k_encoded_cleanup_length, htj2k_encoded_num_coding_passes,
        htj2k_encoded_num_zero_bitplanes, htj2k_encoded_refinement_length,
        CudaHtj2kEncodeStageTimings, CudaHtj2kEncodeStatus, CudaHtj2kEncodedCodeBlock,
        CudaHtj2kEncodedCodeBlocks,
    },
};

/// Host-visible compact HTJ2K cleanup-pass encode metadata for one code block.
#[doc(hidden)]
#[derive(Debug)]
pub struct CudaHtj2kCompactEncodedCodeBlock {
    pub(crate) payload_range: std::ops::Range<usize>,
    pub(crate) status: CudaHtj2kEncodeStatus,
    pub(crate) execution: CudaExecutionStats,
    pub(crate) stage_timings: CudaHtj2kEncodeStageTimings,
}

impl CudaHtj2kCompactEncodedCodeBlock {
    /// Encoded cleanup-pass payload range in the batch payload.
    pub fn payload_range(&self) -> std::ops::Range<usize> {
        self.payload_range.clone()
    }

    impl_cuda_htj2k_encoded_status_accessors!();

    /// Consume this code block and return its payload range plus segment metadata.
    pub fn into_parts(self) -> (std::ops::Range<usize>, u32, u32, u8, u8) {
        (
            self.payload_range,
            htj2k_encoded_cleanup_length(self.status),
            htj2k_encoded_refinement_length(self.status),
            htj2k_encoded_num_coding_passes(self.status),
            htj2k_encoded_num_zero_bitplanes(self.status),
        )
    }
}

/// Host-visible compact HTJ2K cleanup-pass encode batch produced by one CUDA
/// kernel dispatch.
#[doc(hidden)]
#[derive(Debug)]
pub struct CudaHtj2kCompactEncodedCodeBlocks {
    pub(crate) payload: Vec<u8>,
    pub(crate) code_blocks: Vec<CudaHtj2kCompactEncodedCodeBlock>,
    pub(crate) execution: CudaExecutionStats,
    pub(crate) stage_timings: CudaHtj2kEncodeStageTimings,
}

impl CudaHtj2kCompactEncodedCodeBlocks {
    /// Compact encoded payload shared by all code-block ranges.
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    /// Encoded cleanup code-block metadata, in submitted-job order.
    pub fn code_blocks(&self) -> &[CudaHtj2kCompactEncodedCodeBlock] {
        &self.code_blocks
    }

    /// Consume the batch and return its payload plus per-code-block metadata.
    pub fn into_payload_and_code_blocks(self) -> (Vec<u8>, Vec<CudaHtj2kCompactEncodedCodeBlock>) {
        (self.payload, self.code_blocks)
    }

    /// CUDA execution counters for the batch encode dispatch.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }

    /// CUDA event timings for the batch encode dispatch.
    pub fn stage_timings(&self) -> CudaHtj2kEncodeStageTimings {
        self.stage_timings
    }

    pub(crate) fn into_owned_code_blocks_with_live_host_bytes(
        self,
        live_host_bytes: usize,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        let Self {
            payload,
            code_blocks,
            execution,
            stage_timings,
        } = self;
        let mut host_budget = HostPhaseBudget::with_live_bytes(
            "CUDA compact HTJ2K code-block expansion",
            live_host_bytes,
        )?;
        host_budget.account_vec(&payload)?;
        host_budget.account_vec(&code_blocks)?;
        let mut owned_code_blocks = host_budget.try_vec_with_capacity(code_blocks.len())?;
        for block in code_blocks {
            let CudaHtj2kCompactEncodedCodeBlock {
                payload_range,
                status,
                execution,
                stage_timings,
            } = block;
            if payload_range.start > payload_range.end || payload_range.end > payload.len() {
                return Err(CudaError::LengthTooLarge {
                    len: payload_range.end,
                });
            }
            let data = host_budget.try_vec_from_slice(&payload[payload_range])?;
            owned_code_blocks.push(CudaHtj2kEncodedCodeBlock {
                data,
                status,
                execution,
                stage_timings,
            });
        }

        Ok(CudaHtj2kEncodedCodeBlocks {
            code_blocks: owned_code_blocks,
            execution,
            stage_timings,
        })
    }
}

pub(crate) const HTJ2K_UVLC_ENCODE_TABLE_BYTES: usize = 75 * 6;
