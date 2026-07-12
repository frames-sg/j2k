// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::super::NativeEncodePipelineResult;
use super::*;
use crate::{EncodeError, J2kForwardDwt97Output, PrecomputedHtj2k97Component};
use alloc::vec;

fn options() -> EncodeOptions {
    EncodeOptions {
        num_decomposition_levels: 0,
        reversible: false,
        guard_bits: 2,
        use_ht_block_coding: true,
        code_block_width_exp: 2,
        code_block_height_exp: 2,
        ..EncodeOptions::default()
    }
}

fn image(value: f32) -> PrecomputedHtj2k97Image {
    PrecomputedHtj2k97Image {
        width: 1,
        height: 1,
        bit_depth: 8,
        signed: false,
        components: vec![PrecomputedHtj2k97Component {
            x_rsiz: 1,
            y_rsiz: 1,
            dwt: J2kForwardDwt97Output {
                ll: vec![value],
                ll_width: 1,
                ll_height: 1,
                levels: Vec::new(),
            },
        }],
    }
}

#[derive(Default)]
struct BatchCountingAccelerator {
    tier1_batches: usize,
    tier1_jobs: usize,
}

struct FailingBatchAccelerator;

impl J2kEncodeStageAccelerator for FailingBatchAccelerator {
    fn encode_ht_code_blocks(
        &mut self,
        _jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
    ) -> crate::J2kEncodeStageResult<Option<Vec<crate::EncodedHtJ2kCodeBlock>>> {
        Err(crate::J2kEncodeStageError::internal_invariant(
            "injected precomputed 9/7 batch failure",
        ))
    }
}

impl J2kEncodeStageAccelerator for BatchCountingAccelerator {
    fn encode_ht_code_blocks(
        &mut self,
        jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
    ) -> crate::J2kEncodeStageResult<Option<Vec<crate::EncodedHtJ2kCodeBlock>>> {
        self.tier1_batches += 1;
        self.tier1_jobs += jobs.len();
        Ok(None)
    }
}

#[test]
fn owned_precomputed_97_batch_keeps_one_tier1_batch_and_byte_parity() {
    let images = vec![image(1.0), image(2.0)];
    let expected = images
        .iter()
        .map(|image| {
            super::super::encode_precomputed_htj2k_97(image, &options())
                .expect("single precomputed 9/7 encode")
        })
        .collect::<Vec<_>>();
    let mut accelerator = BatchCountingAccelerator::default();

    let actual = encode_precomputed_htj2k_97_batch_owned_with_accelerator(
        images,
        &options(),
        &mut accelerator,
    )
    .expect("owned precomputed 9/7 batch");

    assert_eq!(actual, expected);
    assert_eq!(accelerator.tier1_batches, 1);
    assert_eq!(accelerator.tier1_jobs, 2);
}

#[test]
fn public_precomputed_97_batch_keeps_accelerator_error_category() {
    let error = encode_precomputed_htj2k_97_batch_owned_with_accelerator(
        vec![image(1.0), image(2.0)],
        &options(),
        &mut FailingBatchAccelerator,
    )
    .expect_err("batch accelerator failure must remain typed");

    assert_eq!(
        error,
        EncodeError::Accelerator {
            operation: "HT Tier-1 code-block batch encode",
            source: crate::J2kEncodeStageError::internal_invariant(
                "injected precomputed 9/7 batch failure",
            ),
        }
    );
}

#[test]
fn batch_image_count_mismatch_is_an_internal_invariant() {
    let session = NativeEncodeSession::try_new(NativeEncodeRetainedInput::none())
        .expect("batch mismatch test session");
    let error = super::finalize::packetize_and_finalize_batch(
        Vec::new(),
        vec![Vec::new()],
        &session,
        &mut crate::CpuOnlyJ2kEncodeStageAccelerator,
    )
    .expect_err("mismatched batch owners must fail");

    assert!(matches!(
        error,
        NativeEncodePipelineError::InternalInvariant("encoded image count mismatch")
    ));
}

fn encode_borrowed_batch_at_cap(
    images: &[PrecomputedHtj2k97Image],
    options: &EncodeOptions,
    cap: usize,
) -> NativeEncodePipelineResult<Vec<Vec<u8>>> {
    let retained = precomputed_97_images_retained_bytes(images, images.len())?;
    let session = NativeEncodeSession::try_with_cap(
        NativeEncodeRetainedInput::from_owner_bytes(images, retained),
        cap,
    )?;
    let plans = prepare_batch_plans(images, options, &session)?;
    encode_prepared_batch(
        plans,
        &session,
        &mut crate::CpuOnlyJ2kEncodeStageAccelerator,
    )
}

#[test]
fn aggregate_precomputed_97_batch_accepts_measured_peak_and_rejects_one_byte_less() {
    let images = vec![image(1.0), image(2.0)];
    let options = options();
    let retained =
        precomputed_97_images_retained_bytes(&images, images.len()).expect("batch retained bytes");
    let mut low = retained;
    let mut high = retained + 2_097_152;
    assert!(encode_borrowed_batch_at_cap(&images, &options, high).is_ok());
    while low < high {
        let midpoint = low + (high - low) / 2;
        if encode_borrowed_batch_at_cap(&images, &options, midpoint).is_ok() {
            high = midpoint;
        } else {
            low = midpoint + 1;
        }
    }

    let exact =
        encode_borrowed_batch_at_cap(&images, &options, low).expect("measured exact aggregate cap");
    assert_eq!(exact.len(), 2);
    assert!(matches!(
        encode_borrowed_batch_at_cap(&images, &options, low - 1),
        Err(NativeEncodePipelineError::Typed(
            EncodeError::AllocationTooLarge { .. }
        ))
    ));
}

#[test]
fn batch_preserves_progression_markers_and_tile_part_behavior() {
    let images = vec![image(1.0), image(2.0)];
    let options = EncodeOptions {
        progression_order: super::super::EncodeProgressionOrder::Rlcp,
        write_plt: true,
        write_sop: true,
        write_eph: true,
        tile_part_packet_limit: Some(1),
        ..options()
    };
    let expected = images
        .iter()
        .map(|image| {
            super::super::encode_precomputed_htj2k_97(image, &options)
                .expect("single marker precomputed 9/7 encode")
        })
        .collect::<Vec<_>>();

    let actual = encode_precomputed_htj2k_97_batch_owned_with_accelerator(
        images,
        &options,
        &mut crate::CpuOnlyJ2kEncodeStageAccelerator,
    )
    .expect("marker-bearing precomputed 9/7 batch");

    assert_eq!(actual, expected);
}
