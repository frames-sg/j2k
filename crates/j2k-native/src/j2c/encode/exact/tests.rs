// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::validate_htj2k_codestream;
use crate::{EncodeError, EncodeOptions, DEFAULT_MAX_CODEC_BYTES};

fn small_ht_codestream() -> (Vec<u8>, Vec<u8>, EncodeOptions) {
    let pixels = (0..64)
        .map(|index| u8::try_from((index * 29 + 7) & 0xff).expect("sample fits u8"))
        .collect::<Vec<_>>();
    let options = EncodeOptions {
        num_decomposition_levels: 1,
        use_ht_block_coding: true,
        validate_high_throughput_codestream: false,
        ..EncodeOptions::default()
    };
    let codestream = crate::encode(&pixels, 8, 8, 1, 8, false, &options)
        .expect("encode self-validation fixture");
    (pixels, codestream, options)
}

#[test]
fn ht_self_validation_counts_the_retained_codestream_during_decode() {
    let (pixels, codestream, options) = small_ht_codestream();
    validate_htj2k_codestream(
        &codestream,
        codestream.capacity(),
        &pixels,
        8,
        8,
        1,
        8,
        false,
        options.reversible,
    )
    .expect("ordinary retained-output self-validation");

    let error = validate_htj2k_codestream(
        &codestream,
        DEFAULT_MAX_CODEC_BYTES,
        &pixels,
        8,
        8,
        1,
        8,
        false,
        options.reversible,
    )
    .expect_err("retained output at the cap must reject parse/decode metadata");
    assert!(
        matches!(
            &error,
            EncodeError::AllocationTooLarge {
                requested,
                cap: DEFAULT_MAX_CODEC_BYTES,
                ..
            } if *requested > DEFAULT_MAX_CODEC_BYTES
        ),
        "unexpected self-validation cap error: {error:?}"
    );
}

#[test]
fn ht_self_validation_preserves_host_allocation_category() {
    assert_eq!(
        super::map_self_validation_decode_error(crate::DecodeError::HostAllocationFailed {
            what: "synthetic decode owner",
            bytes: 17,
        }),
        EncodeError::HostAllocationFailed {
            what: "synthetic decode owner",
            bytes: 17,
        }
    );
}
