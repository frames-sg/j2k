// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{vec, vec::Vec};

use super::cleanup::{
    cleanup_segment_suffix_length, cleanup_symbol_stride, decode_cleanup_symbols,
};
use super::magnitude::decode_magnitude_sign_phase;
use super::pipeline::{decode_impl, prepare_scratch, PHASE_LIMIT_MAGREF};
use super::readers::{ForwardBitReader, ReverseBitReader};
use super::refinement::apply_magnitude_refinement_phase;
use super::segments::{CombinedCodeBlockData, HtCodeBlockSegments};
use super::significance::{
    apply_significance_propagation_phase, build_sigma_from_cleanup_phase, sigma_stride,
    SIGPROP_SPREAD_MASKS,
};
use super::state::{zeroed_u16_scratch, zeroed_u32_scratch, HtBlockDecodeScratch};
use super::stats::{HtBlockDecodeStats, NoHtDecodeStats};
use super::validation::{
    decode_combined_validated_with_scratch, decode_segments_validated_with_scratch,
};
use super::{coefficient_to_f32, coefficient_to_i32, store_coefficient_row};
use super::{
    decode_combined_validated, decode_segments_validated_for_phase,
    decode_segments_validated_with_scratch_for_phase,
};
use crate::error::{DecodeError, DecodingError};
use crate::j2c::ht_block_encode::encode_code_block;

#[test]
fn sigprop_reader_discards_stuffed_msb_even_when_overlap_sets_it() {
    let mut reader = ForwardBitReader::<0>::new(&[0xFF, 0x80, 0x00, 0x00, 0x00]);

    assert_eq!(reader.fetch(), 0x0000_00FF);
}

#[test]
fn magref_reader_discards_stuffed_msb_in_shared_refinement_byte() {
    let mut reader = ReverseBitReader::new_mrp(&[0x00, 0x00, 0x00, 0x00, 0xFF]);

    assert_eq!(reader.fetch(), 0x0000_007F);
}

#[test]
fn test_coefficient_to_i32_shifted_alignment() {
    let aligned = 3u32 << (31 - 5);
    assert_eq!(coefficient_to_i32(aligned, 5), 3);
    assert_eq!(coefficient_to_i32(0x8000_0000 | aligned, 5), -3);
}

#[test]
fn irreversible_coefficient_preserves_the_ht_midpoint_fraction() {
    let midpoint = 3u32 << (31 - 5 - 1);

    assert_eq!(coefficient_to_f32(midpoint, 5).to_bits(), 1.5_f32.to_bits());
    assert_eq!(
        coefficient_to_f32(0x8000_0000 | midpoint, 5).to_bits(),
        (-1.5_f32).to_bits()
    );
}

#[test]
fn test_direct_ht_block_roundtrip_varied_4x4() {
    let original: Vec<i32> = (0..16).map(|i| (i * 3) - 20).collect();
    let total_bitplanes = 6u8;
    let encoded = encode_code_block(&original, 4, 4, total_bitplanes).expect("encode HT block");
    assert_eq!(encoded.num_coding_passes, 1);

    let mut decoded = vec![0u32; original.len()];
    let mut scratch = HtBlockDecodeScratch::default();
    prepare_scratch(&mut scratch, 4, 4).expect("prepare HT scratch");
    let mut observer = NoHtDecodeStats;
    let decoded_ok = decode_impl::<PHASE_LIMIT_MAGREF, _>(
        &encoded.data,
        &[],
        &mut decoded,
        u32::from(encoded.num_zero_bitplanes),
        u32::from(encoded.num_coding_passes),
        4,
        4,
        4,
        false,
        &mut scratch,
        &mut observer,
    );
    assert!(decoded_ok.is_some(), "encoded={:02x?}", encoded.data);

    let decoded_i32: Vec<i32> = decoded
        .into_iter()
        .map(|value| coefficient_to_i32(value, total_bitplanes))
        .collect();
    assert_eq!(decoded_i32, original, "encoded={:02x?}", encoded.data);
}

#[test]
fn test_direct_ht_block_roundtrip_positive_varied_4x4() {
    let original: Vec<i32> = (0..16).map(|i| i * 3).collect();
    let total_bitplanes = 6u8;
    let encoded = encode_code_block(&original, 4, 4, total_bitplanes).expect("encode HT block");
    assert_eq!(encoded.num_coding_passes, 1);

    let mut decoded = vec![0u32; original.len()];
    let mut scratch = HtBlockDecodeScratch::default();
    prepare_scratch(&mut scratch, 4, 4).expect("prepare HT scratch");
    let mut observer = NoHtDecodeStats;
    let decoded_ok = decode_impl::<PHASE_LIMIT_MAGREF, _>(
        &encoded.data,
        &[],
        &mut decoded,
        u32::from(encoded.num_zero_bitplanes),
        u32::from(encoded.num_coding_passes),
        4,
        4,
        4,
        false,
        &mut scratch,
        &mut observer,
    );
    assert!(decoded_ok.is_some(), "encoded={:02x?}", encoded.data);

    let decoded_i32: Vec<i32> = decoded
        .into_iter()
        .map(|value| coefficient_to_i32(value, total_bitplanes))
        .collect();
    assert_eq!(decoded_i32, original, "encoded={:02x?}", encoded.data);
}

#[test]
fn direct_ht_block_roundtrip_31_bit_cleanup_path() {
    let original = vec![
        0,
        1,
        -1,
        255,
        -255,
        65_535,
        -65_535,
        16_777_216,
        -16_777_216,
        (1_i32 << 30) + 17,
        -((1_i32 << 30) + 17),
        (1_i32 << 30) - 1,
        -((1_i32 << 30) - 1),
        1_i32 << 30,
        -(1_i32 << 30),
        i32::MAX,
    ];
    let total_bitplanes = 31u8;
    let encoded = encode_code_block(&original, 4, 4, total_bitplanes).expect("encode HT31 block");
    assert_eq!(encoded.num_coding_passes, 1);
    assert_eq!(encoded.num_zero_bitplanes, 30);

    let mut decoded = vec![0u32; original.len()];
    let mut scratch = HtBlockDecodeScratch::default();
    prepare_scratch(&mut scratch, 4, 4).expect("prepare HT scratch");
    let mut observer = NoHtDecodeStats;
    let decoded_ok = decode_impl::<PHASE_LIMIT_MAGREF, _>(
        &encoded.data,
        &[],
        &mut decoded,
        u32::from(encoded.num_zero_bitplanes),
        u32::from(encoded.num_coding_passes),
        4,
        4,
        4,
        false,
        &mut scratch,
        &mut observer,
    );
    assert!(decoded_ok.is_some(), "encoded={:02x?}", encoded.data);

    let decoded_i32: Vec<i32> = decoded
        .into_iter()
        .map(|value| coefficient_to_i32(value, total_bitplanes))
        .collect();
    assert_eq!(decoded_i32, original, "encoded={:02x?}", encoded.data);
}

#[test]
fn cleanup_and_magnitude_sign_phases_decode_odd_sized_block() {
    let width = 15u32;
    let height = 13u32;
    let original: Vec<i32> = (0..(width * height))
        .map(|i| {
            let value = (i32::try_from(i).expect("test coefficient index fits i32") % 61) - 30;
            if i % 7 == 0 {
                0
            } else {
                value
            }
        })
        .collect();
    let total_bitplanes = 6u8;
    let encoded =
        encode_code_block(&original, width, height, total_bitplanes).expect("encode HT block");
    assert_eq!(encoded.num_coding_passes, 1);

    let lcup = encoded.data.len();
    let scup = cleanup_segment_suffix_length(&encoded.data, lcup).expect("valid cleanup info");
    let sstr = cleanup_symbol_stride(width);
    let quad_rows = height.div_ceil(2) as usize;
    let mut cleanup = vec![0u16; sstr * (quad_rows + 1)];
    decode_cleanup_symbols(&encoded.data, lcup, scup, width, height, sstr, &mut cleanup)
        .expect("decode cleanup symbols");

    let mut decoded = vec![0u32; original.len()];
    let mut v_n_scratch = vec![0u32; width.div_ceil(2) as usize + 2];
    let mut magsgn = Vec::with_capacity(
        super::readers::flat_magsgn_capacity(original.len()).expect("MagSgn capacity"),
    );
    decode_magnitude_sign_phase(
        &encoded.data,
        lcup,
        scup,
        &cleanup,
        &mut decoded,
        u32::from(encoded.num_zero_bitplanes),
        width,
        height,
        width,
        sstr,
        &mut v_n_scratch,
        &mut magsgn,
    )
    .expect("decode magnitude/sign phase");

    let decoded_i32: Vec<i32> = decoded
        .into_iter()
        .map(|value| coefficient_to_i32(value, total_bitplanes))
        .collect();
    assert_eq!(decoded_i32, original, "encoded={:02x?}", encoded.data);
}

#[test]
fn sigma_phase_builds_masks_and_zeroes_edge_sentinels() {
    let width = 7u32;
    let height = 5u32;
    let sstr = cleanup_symbol_stride(width);
    let mstr = sigma_stride(width);
    let sigma_rows = height.div_ceil(4) as usize + 1;
    let mut cleanup = vec![0u16; sstr * (height.div_ceil(2) as usize + 1)];
    cleanup[0] = 0x30;
    cleanup[2] = 0xC0;
    cleanup[sstr] = 0xF0;
    cleanup[sstr + 2] = 0x30;
    cleanup[2 * sstr] = 0xC0;
    cleanup[2 * sstr + 2] = 0xF0;
    let mut sigma = vec![0u16; sigma_rows * mstr];

    build_sigma_from_cleanup_phase(&cleanup, &mut sigma, width, height, sstr, mstr)
        .expect("build sigma");

    let expected_first = u16::try_from(
        ((u32::from(cleanup[0]) & 0x30) >> 4)
            | ((u32::from(cleanup[0]) & 0xC0) >> 2)
            | ((u32::from(cleanup[2]) & 0x30) << 4)
            | ((u32::from(cleanup[2]) & 0xC0) << 6)
            | ((u32::from(cleanup[sstr]) & 0x30) >> 2)
            | (u32::from(cleanup[sstr]) & 0xC0)
            | ((u32::from(cleanup[sstr + 2]) & 0x30) << 6)
            | ((u32::from(cleanup[sstr + 2]) & 0xC0) << 8),
    )
    .expect("sigma test mask fits u16");
    let expected_second = u16::try_from(
        ((u32::from(cleanup[4]) & 0x30) >> 4)
            | ((u32::from(cleanup[4]) & 0xC0) >> 2)
            | ((u32::from(cleanup[6]) & 0x30) << 4)
            | ((u32::from(cleanup[6]) & 0xC0) << 6)
            | ((u32::from(cleanup[sstr + 4]) & 0x30) >> 2)
            | (u32::from(cleanup[sstr + 4]) & 0xC0)
            | ((u32::from(cleanup[sstr + 6]) & 0x30) << 6)
            | ((u32::from(cleanup[sstr + 6]) & 0xC0) << 8),
    )
    .expect("sigma test mask fits u16");
    assert_eq!(sigma[0], expected_first);
    assert_eq!(sigma[1], expected_second);
    assert_eq!(sigma[2], 0);

    let bottom = height.div_ceil(4) as usize * mstr;
    for x in 0..=width.div_ceil(4) as usize {
        assert_eq!(sigma[bottom + x], 0);
    }
}

#[test]
fn refinement_phases_leave_output_unchanged_for_empty_sigma() {
    let width = 7u32;
    let height = 5u32;
    let stride = width;
    let mstr = sigma_stride(width);
    let sigma = vec![0u16; (height.div_ceil(4) as usize + 1) * mstr];
    let mut prev_row_sig = vec![0u16; width.div_ceil(4) as usize + 8];
    let mut decoded = vec![0x1234_5678u32; (stride * height) as usize];
    let expected = decoded.clone();

    apply_significance_propagation_phase(
        &[],
        &sigma,
        &mut decoded,
        width,
        height,
        stride,
        mstr,
        false,
        5,
        &mut prev_row_sig,
    )
    .expect("empty sigma sigprop");
    apply_magnitude_refinement_phase(&[], &sigma, &mut decoded, width, height, stride, mstr, 5)
        .expect("empty sigma magref");

    assert_eq!(decoded, expected);
}

#[test]
fn magnitude_refinement_skips_significance_outside_the_code_block() {
    // A malformed cleanup pass can mark samples of quads that lie across the
    // block edge as significant. Here every sample of an 8 x 4 group is
    // marked in a 5 x 1 block. Refinement must not write outside the block,
    // but those samples still consume their bits, so the in-block samples
    // refine exactly as the first row of an 8 x 4 block does.
    let p = 5;
    let refinement = [0xA5u8, 0x3C, 0x5A, 0xC3];
    let refine = |width: u32, height: u32| {
        let mstr = sigma_stride(width);
        let mut sigma = vec![0u16; (height.div_ceil(4) as usize + 1) * mstr];
        sigma[0] = 0xFFFF;
        sigma[1] = 0xFFFF;
        let mut decoded = vec![1u32 << p; (width * height) as usize];
        apply_magnitude_refinement_phase(
            &refinement,
            &sigma,
            &mut decoded,
            width,
            height,
            width,
            mstr,
            p,
        )
        .expect("magref");
        decoded
    };

    let block = refine(5, 1);
    let larger = refine(8, 4);
    assert_eq!(block, larger[..5]);
    assert_ne!(block, vec![1u32 << p; 5], "refinement data was applied");
}

#[test]
fn fuzzed_code_block_with_significance_below_a_one_row_block_decodes() {
    // From `region_scaled_fuzz`: a 41 x 1 block whose cleanup pass marks
    // samples in rows 1-3 as significant. Its refinement pass used to index
    // past the block.
    let data = [
        0xca, 0x36, 0xf6, 0x9d, 0xab, 0xe3, 0xf0, 0x71, 0xfb, 0x35, 0xfb, 0x5a, 0x36, 0xb5, 0x1e,
        0x45, 0xa7, 0x0f, 0xce, 0xe7, 0x7b, 0xa4, 0x8d, 0xff, 0x1f, 0x64, 0x01, 0x8b, 0x5b, 0x00,
        0x3d, 0x82, 0x0d,
    ];
    let job = crate::HtCodeBlockDecodeJob {
        data: &data,
        cleanup_length: 27,
        refinement_length: 6,
        width: 41,
        height: 1,
        output_stride: 41,
        missing_bit_planes: 7,
        number_of_coding_passes: 3,
        num_bitplanes: 14,
        roi_shift: 0,
        stripe_causal: false,
        strict: true,
        dequantization_step: 1.0,
    };
    let mut output = vec![0.0f32; 41];

    crate::decode_ht_code_block_scalar(job, &mut output).expect("decodes");
}

#[test]
fn sigprop_spread_masks_follow_column_major_scan_order() {
    let row_patterns = [0x33u32, 0x76, 0xEC, 0xC8];

    for bit in 0..16 {
        let expected = row_patterns[bit & 3] << (bit & !3);
        assert_eq!(SIGPROP_SPREAD_MASKS[bit], expected, "bit={bit}");
        assert_eq!(SIGPROP_SPREAD_MASKS[bit] & ((1u32 << bit) - 1), 0);
    }
}

#[test]
fn combined_data_exposes_borrowed_segment_slices() {
    let combined = CombinedCodeBlockData {
        data: vec![0x11, 0x22, 0x33, 0x44, 0x55],
        cleanup_length: 3,
        refinement_length: 2,
    };

    let segments = combined.segments().expect("split combined data");

    assert_eq!(segments.cleanup, &[0x11, 0x22, 0x33]);
    assert_eq!(segments.refinement, &[0x44, 0x55]);
}

#[test]
fn borrowed_segments_decode_matches_owned_combined_decode() {
    let width = 16u32;
    let height = 16u32;
    let original: Vec<i32> = (0..(width * height))
        .map(|i| {
            let value = (i32::try_from(i).expect("test coefficient index fits i32") % 47) - 23;
            if i % 5 == 0 {
                0
            } else {
                value
            }
        })
        .collect();
    let total_bitplanes = 6u8;
    let encoded =
        encode_code_block(&original, width, height, total_bitplanes).expect("encode HT block");
    let combined = CombinedCodeBlockData {
        data: encoded.data.clone(),
        cleanup_length: u32::try_from(encoded.data.len()).expect("test payload length fits u32"),
        refinement_length: 0,
    };
    let segments = HtCodeBlockSegments {
        cleanup: &encoded.data,
        refinement: &[],
    };
    let mut owned_decoded = vec![0u32; original.len()];
    let mut borrowed_decoded = vec![0u32; original.len()];
    let mut scratch = HtBlockDecodeScratch::default();

    decode_combined_validated(
        &combined,
        encoded.num_zero_bitplanes,
        total_bitplanes,
        encoded.num_coding_passes,
        false,
        true,
        &mut owned_decoded,
        width,
        height,
        width,
    )
    .expect("decode owned combined payload");
    decode_segments_validated_with_scratch(
        &segments,
        encoded.num_zero_bitplanes,
        total_bitplanes,
        encoded.num_coding_passes,
        false,
        true,
        &mut borrowed_decoded,
        width,
        height,
        width,
        &mut scratch,
    )
    .expect("decode borrowed payload segments");

    assert_eq!(borrowed_decoded, owned_decoded);
}

#[test]
fn scratch_resize_zeroes_existing_values_when_growing() {
    let mut scratch = HtBlockDecodeScratch::default();
    scratch
        .cleanup
        .try_reserve_exact(8)
        .expect("cleanup scratch");
    scratch.v_n.try_reserve_exact(8).expect("v_n scratch");

    zeroed_u16_scratch(&mut scratch.cleanup, 4)
        .expect("reserved cleanup scratch")
        .fill(7);
    assert_eq!(
        zeroed_u16_scratch(&mut scratch.cleanup, 8).expect("reserved cleanup scratch"),
        &[0; 8]
    );

    zeroed_u32_scratch(&mut scratch.v_n, 4)
        .expect("reserved v_n scratch")
        .fill(9);
    assert_eq!(
        zeroed_u32_scratch(&mut scratch.v_n, 8).expect("reserved v_n scratch"),
        &[0; 8]
    );
}

#[test]
fn undersized_scratch_returns_none_without_allocating() {
    let mut cleanup = Vec::<u16>::new();
    let mut v_n = Vec::<u32>::new();

    assert!(zeroed_u16_scratch(&mut cleanup, 1).is_none());
    assert!(zeroed_u32_scratch(&mut v_n, 1).is_none());
    assert_eq!(cleanup.capacity(), 0);
    assert_eq!(v_n.capacity(), 0);
}

#[test]
fn decode_combined_validated_with_scratch_reuses_zeroed_buffers() {
    let width = 16u32;
    let height = 16u32;
    let original: Vec<i32> = (0..(width * height))
        .map(|i| {
            let value = (i32::try_from(i).expect("test coefficient index fits i32") % 47) - 23;
            if i % 5 == 0 {
                0
            } else {
                value
            }
        })
        .collect();
    let total_bitplanes = 6u8;
    let encoded =
        encode_code_block(&original, width, height, total_bitplanes).expect("encode HT block");
    let combined = CombinedCodeBlockData {
        data: encoded.data.clone(),
        cleanup_length: u32::try_from(encoded.data.len()).expect("test payload length fits u32"),
        refinement_length: 0,
    };
    let mut scratch = HtBlockDecodeScratch::default();
    let mut decoded = vec![0u32; original.len()];

    decode_combined_validated_with_scratch(
        &combined,
        encoded.num_zero_bitplanes,
        total_bitplanes,
        encoded.num_coding_passes,
        false,
        true,
        &mut decoded,
        width,
        height,
        width,
        &mut scratch,
    )
    .expect("decode HT block");

    let first_capacities = scratch.capacities_for_test();
    assert!(first_capacities.cleanup > 0);
    assert!(first_capacities.v_n > 0);

    scratch.poison_for_test();
    decoded.fill(0);

    decode_combined_validated_with_scratch(
        &combined,
        encoded.num_zero_bitplanes,
        total_bitplanes,
        encoded.num_coding_passes,
        false,
        true,
        &mut decoded,
        width,
        height,
        width,
        &mut scratch,
    )
    .expect("decode HT block after scratch poison");

    assert_eq!(scratch.capacities_for_test(), first_capacities);
    let decoded_i32: Vec<i32> = decoded
        .into_iter()
        .map(|value| coefficient_to_i32(value, total_bitplanes))
        .collect();
    assert_eq!(decoded_i32, original, "encoded={:02x?}", encoded.data);
}

#[test]
fn segment_and_strict_validation_errors_remain_exact_and_non_mutating() {
    let Err(error) = HtCodeBlockSegments::from_combined_payload(&[1, 2, 3], 2, 2) else {
        panic!("short combined payload must fail");
    };
    assert_eq!(
        error,
        DecodeError::Decoding(DecodingError::CodeBlockDecodeFailure)
    );

    let segments = HtCodeBlockSegments {
        cleanup: &[],
        refinement: &[],
    };
    let mut decoded = [0xDEAD_BEEFu32];
    let invalid_cases = [
        (0, 32, 1, DecodingError::TooManyBitplanes),
        (2, 1, 1, DecodingError::InvalidBitplaneCount),
        (0, 1, 2, DecodingError::TooManyCodingPasses),
    ];

    for (missing, total, passes, expected) in invalid_cases {
        let error = decode_segments_validated_for_phase::<PHASE_LIMIT_MAGREF>(
            &segments,
            missing,
            total,
            passes,
            false,
            true,
            &mut decoded,
            1,
            1,
            1,
        )
        .expect_err("strict validation must reject invalid metadata");
        assert_eq!(error, DecodeError::Decoding(expected));
        assert_eq!(decoded, [0xDEAD_BEEF]);
    }
}

#[test]
fn decode_stats_preserve_cleanup_counts_and_disabled_timing() {
    let width = 4u32;
    let height = 4u32;
    let total_bitplanes = 6u8;
    let original: Vec<i32> = (0..16).map(|i| (i * 3) - 20).collect();
    let encoded = encode_code_block(&original, width, height, total_bitplanes)
        .expect("encode cleanup-only HT block");
    assert_eq!(encoded.num_coding_passes, 1);
    let segments = HtCodeBlockSegments {
        cleanup: &encoded.data,
        refinement: &[],
    };
    let mut decoded = vec![0u32; original.len()];
    let mut scratch = HtBlockDecodeScratch::default();
    let mut stats = HtBlockDecodeStats::default();

    decode_segments_validated_with_scratch_for_phase::<PHASE_LIMIT_MAGREF>(
        &segments,
        encoded.num_zero_bitplanes,
        total_bitplanes,
        encoded.num_coding_passes,
        false,
        true,
        &mut decoded,
        width,
        height,
        width,
        &mut scratch,
        Some(&mut stats),
        false,
    )
    .expect("decode cleanup-only HT block with stats");

    assert_eq!(stats.blocks, 1);
    assert_eq!(stats.refinement_blocks, 0);
    assert_eq!(stats.cleanup_bytes, encoded.data.len() as u128);
    assert_eq!(stats.refinement_bytes, 0);
    assert_eq!(
        (
            stats.ht_cleanup_us,
            stats.ht_mag_sgn_us,
            stats.ht_sigma_us,
            stats.ht_sigprop_us,
            stats.ht_magref_us,
        ),
        (0, 0, 0, 0, 0)
    );
}

#[test]
#[expect(
    clippy::cast_precision_loss,
    reason = "mirrors the per-sample conversion"
)]
fn coefficient_rows_match_the_per_sample_conversion() {
    let mut coefficients = vec![0, 1, 0x8000_0000, 0x8000_0001, 0x7FFF_FFFF, 0xFFFF_FFFF];
    let mut state = 0x1234_5678_u32;
    for _ in 0..200 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        coefficients.push(state);
    }
    for k_max in [1_u8, 6, 16, 17, 30, 31] {
        for roi_shift in [0_u8, 3] {
            for irreversible_midpoint in [false, true] {
                for step in [1.0_f32, 0.375] {
                    let mut actual = vec![f32::NAN; coefficients.len()];
                    store_coefficient_row(
                        &coefficients,
                        &mut actual,
                        k_max,
                        roi_shift,
                        irreversible_midpoint,
                        step,
                    );
                    for (&coefficient, sample) in coefficients.iter().zip(&actual) {
                        let expected = if irreversible_midpoint {
                            crate::apply_roi_maxshift_inverse_f32(
                                coefficient_to_f32(coefficient, k_max),
                                roi_shift,
                            )
                        } else {
                            crate::apply_roi_maxshift_inverse_i32(
                                coefficient_to_i32(coefficient, k_max),
                                roi_shift,
                            ) as f32
                        } * step;
                        assert_eq!(
                            sample.to_bits(),
                            expected.to_bits(),
                            "coefficient={coefficient:#x} k_max={k_max} roi={roi_shift} \
                             midpoint={irreversible_midpoint} step={step}"
                        );
                    }
                }
            }
        }
    }
}
