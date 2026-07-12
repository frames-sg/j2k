// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::{EncodeError, EncodedHtJ2kCodeBlock};

struct MalformedFusedHtAccelerator;

impl J2kEncodeStageAccelerator for MalformedFusedHtAccelerator {
    fn encode_ht_subband(
        &mut self,
        job: J2kHtSubbandEncodeJob<'_>,
    ) -> crate::J2kEncodeStageResult<Option<Vec<EncodedHtJ2kCodeBlock>>> {
        let count = usize::try_from(job.width.div_ceil(job.code_block_width))
            .ok()
            .and_then(|columns| {
                usize::try_from(job.height.div_ceil(job.code_block_height))
                    .ok()
                    .and_then(|rows| columns.checked_mul(rows))
            })
            .ok_or_else(|| {
                crate::J2kEncodeStageError::arithmetic_overflow("test fused code-block count")
            })?;
        let output_bytes = count
            .checked_mul(core::mem::size_of::<EncodedHtJ2kCodeBlock>())
            .ok_or_else(|| {
                crate::J2kEncodeStageError::arithmetic_overflow("test fused output allocation size")
            })?;
        let mut outputs = Vec::new();
        outputs.try_reserve_exact(count).map_err(|_| {
            crate::J2kEncodeStageError::host_allocation_failed("test fused output", output_bytes)
        })?;
        for _ in 0..count {
            let mut data = Vec::new();
            data.try_reserve_exact(2).map_err(|_| {
                crate::J2kEncodeStageError::host_allocation_failed("test fused payload", 2)
            })?;
            data.extend([0x5a, 0xa5]);
            outputs.push(EncodedHtJ2kCodeBlock {
                data,
                cleanup_length: 3,
                refinement_length: 0,
                num_coding_passes: 1,
                num_zero_bitplanes: 0,
            });
        }
        Ok(Some(outputs))
    }
}

#[test]
fn malformed_fused_ht_metadata_keeps_the_accelerator_operation_category() {
    let coefficients = [0.0; 16];
    let result = prepare_subband(
        &coefficients,
        4,
        4,
        &QuantStepSize {
            exponent: 8,
            mantissa: 0,
        },
        8,
        2,
        true,
        BlockCodingMode::HighThroughput,
        2,
        2,
        SubBandType::LowLow,
        0,
        &[],
        1,
        1,
        &mut MalformedFusedHtAccelerator,
    );
    let Err(error) = result else {
        panic!("malformed fused HT metadata must fail at the accelerator boundary");
    };

    assert_eq!(
        error,
        EncodeError::Accelerator {
            operation: "fused HT subband encode",
            source: crate::J2kEncodeStageError::internal_invariant(
                "HTJ2K payload segment length mismatch",
            ),
        }
    );
}
