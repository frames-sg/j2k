// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec;

use super::*;
use crate::{
    CpuOnlyJ2kEncodeStageAccelerator, EncodeError, J2kForwardDwt53Output,
    PrecomputedHtj2k53Component,
};

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

fn encode_at_cap(cap: usize) -> crate::EncodeResult<Vec<u8>> {
    encode_precomputed_htj2k_53_with_accelerator_and_max_host_bytes(
        &image(),
        &options(),
        &mut CpuOnlyJ2kEncodeStageAccelerator,
        cap,
    )
}

#[test]
fn lowered_public_cap_accepts_measured_exact_peak_and_rejects_one_less() {
    let mut low = 0usize;
    let mut high = 1_048_576usize;
    assert!(encode_at_cap(high).is_ok());
    while low < high {
        let midpoint = low + (high - low) / 2;
        if encode_at_cap(midpoint).is_ok() {
            high = midpoint;
        } else {
            low = midpoint + 1;
        }
    }

    let exact = encode_at_cap(low).expect("measured exact lowered-cap encode");
    assert!(low > 0);
    assert!(matches!(
        encode_at_cap(low - 1),
        Err(EncodeError::AllocationTooLarge { .. })
    ));
    assert_eq!(
        exact,
        crate::encode_precomputed_htj2k_53(&image(), &options()).expect("default-cap encode")
    );
}

#[test]
fn requested_cap_above_process_ceiling_is_clamped() {
    let above = encode_at_cap(usize::MAX).expect("above-ceiling request is clamped");
    let process = encode_at_cap(crate::DEFAULT_MAX_CODEC_BYTES).expect("process-cap encode");
    assert_eq!(above, process);
}
