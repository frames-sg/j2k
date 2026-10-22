// SPDX-License-Identifier: MIT OR Apache-2.0

//! Quality decodes that also charge a caller's retained allocations.

use alloc::vec::Vec;

use j2k_native::DEFAULT_MAX_DECODE_BYTES;

use super::super::decoded_psnr;
use crate::encode::lossy::encode_cpu_lossy;
use crate::encode::{J2kBlockCodingMode, J2kLossyEncodeOptions, J2kLossySamples};

fn textured_rgb() -> Vec<u8> {
    let mut state = 0x9e37_79b9_u32;
    (0..128 * 96 * 3)
        .map(|index: u32| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            u8::try_from(((index % 384) / 2 + (state >> 27)) & 0xff).expect("masked sample")
        })
        .collect()
}

#[test]
fn retained_quality_decode_matches_plain_decode_up_to_its_refusal_with_three_threads() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .expect("three-thread validation pool");
    pool.install(retained_quality_decode_matches_plain_decode_up_to_its_refusal);
}

fn retained_quality_decode_matches_plain_decode_up_to_its_refusal() {
    let pixels = textured_rgb();
    let samples = J2kLossySamples::new(&pixels, 128, 96, 3, 8, false).expect("fixture samples");
    for mode in [
        J2kBlockCodingMode::HighThroughput,
        J2kBlockCodingMode::Classic,
    ] {
        let options = J2kLossyEncodeOptions {
            block_coding_mode: mode,
            ..J2kLossyEncodeOptions::default()
        };
        let codestream = encode_cpu_lossy(samples, &options, 1.0).expect("fixture encode");
        let plain = decoded_psnr(samples, &codestream, 0).expect("plain quality decode");
        let fits = |extra| decoded_psnr(samples, &codestream, extra).is_ok();
        let (mut accepted, mut refused) = (0, DEFAULT_MAX_DECODE_BYTES);
        assert!(fits(accepted) && !fits(refused), "{mode:?}");
        while refused - accepted > 1 {
            let middle = accepted + (refused - accepted) / 2;
            if fits(middle) {
                accepted = middle;
            } else {
                refused = middle;
            }
        }
        // Near the edge the decoder takes its smaller-footprint routes (per
        // sub-band HT planning, serial strips, a released coefficient slab).
        // They schedule entropy decoding differently but must decode the same
        // samples, so the quality metric keeps its exact bits.
        for slack in [0, 1, 4_096, 1 << 16, 1 << 20, accepted] {
            let extra = accepted.saturating_sub(slack);
            let psnr = decoded_psnr(samples, &codestream, extra)
                .unwrap_or_else(|error| panic!("{mode:?} extra {extra}: {error:?}"));
            assert_eq!(psnr.to_bits(), plain.to_bits(), "{mode:?} extra {extra}");
        }
    }
}
