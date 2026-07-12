// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::{EncodeError, J2kForwardDwt53Output, PrecomputedHtj2k53Component};
use alloc::vec;

fn image() -> PrecomputedHtj2k53Image {
    PrecomputedHtj2k53Image {
        width: 1,
        height: 1,
        bit_depth: 8,
        signed: false,
        components: vec![PrecomputedHtj2k53Component {
            x_rsiz: 1,
            y_rsiz: 1,
            dwt: J2kForwardDwt53Output {
                ll: vec![1.0],
                ll_width: 1,
                ll_height: 1,
                levels: Vec::new(),
            },
        }],
    }
}

fn options() -> EncodeOptions {
    EncodeOptions {
        num_decomposition_levels: 0,
        reversible: true,
        use_ht_block_coding: true,
        code_block_width_exp: 2,
        code_block_height_exp: 2,
        ..EncodeOptions::default()
    }
}

struct FailingHtBatchAccelerator;

impl J2kEncodeStageAccelerator for FailingHtBatchAccelerator {
    fn encode_ht_code_blocks(
        &mut self,
        _jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
    ) -> crate::J2kEncodeStageResult<Option<Vec<crate::EncodedHtJ2kCodeBlock>>> {
        Err(crate::J2kEncodeStageError::internal_invariant(
            "injected precomputed 5/3 Tier-1 failure",
        ))
    }
}

#[test]
fn public_precomputed_53_keeps_invalid_input_and_accelerator_categories() {
    let mut invalid = image();
    invalid.width = 0;
    assert_eq!(
        encode_precomputed_htj2k_53(&invalid, &options()),
        Err(EncodeError::InvalidInput {
            what: "invalid dimensions",
        })
    );

    let error = encode_precomputed_htj2k_53_with_accelerator(
        &image(),
        &options(),
        &mut FailingHtBatchAccelerator,
    )
    .expect_err("accelerator failure must remain typed");
    assert_eq!(
        error,
        EncodeError::Accelerator {
            operation: "HT Tier-1 code-block batch encode",
            source: crate::J2kEncodeStageError::internal_invariant(
                "injected precomputed 5/3 Tier-1 failure",
            ),
        }
    );

    let mut invalid_dwt = image();
    invalid_dwt.components[0].dwt.ll.clear();
    assert_eq!(
        encode_precomputed_htj2k_53(&invalid_dwt, &options()),
        Err(EncodeError::InvalidInput {
            what: "accelerated DWT output length mismatch",
        })
    );
}
