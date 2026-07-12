//! Packet-header segment-length math shared between the CPU packetizer and
//! GPU packetization planners.
//!
//! Hidden from the rendered public API: GPU planner crates call these helpers
//! internally so their packet headers stay bit-identical with the CPU
//! encoder's signaling decisions.

mod error;
pub use error::HtSegmentLengthError;

/// Returns whether `value` can be signalled in `bits` bits.
#[must_use]
#[inline]
pub fn value_fits_in_bits(value: u32, bits: u32) -> bool {
    bits >= u32::BITS || value < (1u32 << bits)
}

/// Calculate number of bits needed to encode a segment length.
#[must_use]
#[inline]
pub fn bits_for_length(l_block: u32, num_coding_passes: u8) -> u32 {
    let log2_passes = if num_coding_passes <= 1 {
        0
    } else {
        u32::from(num_coding_passes).ilog2()
    };
    l_block + log2_passes
}

/// Number of bits needed to encode an HT cleanup segment length, folding
/// placeholder passes into the pass count.
#[must_use]
#[inline]
pub fn bits_for_ht_cleanup_length(l_block: u32, raw_num_passes: u8) -> u32 {
    let placeholder_groups = u32::from(raw_num_passes.saturating_sub(1)) / 3;
    let placeholder_passes = placeholder_groups * 3;
    l_block + (placeholder_passes + 1).ilog2()
}

/// Number of bits needed to encode an HT refinement-only packet
/// contribution length after the cleanup segment has already appeared in an
/// earlier quality layer.
#[must_use]
#[inline]
pub fn bits_for_ht_refinement_only_length(l_block: u32, num_coding_passes: u8) -> u32 {
    l_block + u32::from(num_coding_passes > 1)
}

/// Splits an HT code-block packet contribution into its
/// `(cleanup, refinement)` segment lengths, validating them against the
/// contribution payload length.
pub fn ht_segment_lengths(
    num_coding_passes: u8,
    data_len: usize,
    ht_cleanup_length: u32,
    ht_refinement_length: u32,
) -> Result<(u32, u32), HtSegmentLengthError> {
    if num_coding_passes == 0 {
        if data_len == 0 && ht_cleanup_length == 0 && ht_refinement_length == 0 {
            return Ok((0, 0));
        }
        return Err(HtSegmentLengthError::EmptyContributionHasSegments);
    }

    let data_len = u32::try_from(data_len)
        .map_err(|_| HtSegmentLengthError::ContributionLengthExceedsU32 { data_len })?;
    if ht_cleanup_length == 0 && ht_refinement_length != 0 {
        if ht_refinement_length != data_len {
            return Err(HtSegmentLengthError::RefinementOnlyLengthMismatch {
                data_len,
                refinement_length: ht_refinement_length,
            });
        }
        if ht_refinement_length >= 2047 {
            return Err(HtSegmentLengthError::RefinementLengthOutOfRange {
                refinement_length: ht_refinement_length,
            });
        }
        return Ok((0, ht_refinement_length));
    }

    if num_coding_passes == 1 {
        if ht_refinement_length != 0 {
            return Err(HtSegmentLengthError::SinglePassHasRefinement {
                refinement_length: ht_refinement_length,
            });
        }
        let cleanup_length = if ht_cleanup_length == 0 {
            data_len
        } else {
            ht_cleanup_length
        };
        if cleanup_length != data_len {
            return Err(HtSegmentLengthError::SinglePassLengthMismatch {
                data_len,
                cleanup_length,
            });
        }
        return Ok((cleanup_length, 0));
    }

    if ht_cleanup_length == 0 || ht_refinement_length == 0 {
        return Err(HtSegmentLengthError::MultiPassRequiresSegments {
            cleanup_length: ht_cleanup_length,
            refinement_length: ht_refinement_length,
        });
    }
    if ht_cleanup_length.checked_add(ht_refinement_length).ok_or(
        HtSegmentLengthError::MultiPassLengthOverflow {
            cleanup_length: ht_cleanup_length,
            refinement_length: ht_refinement_length,
        },
    )? != data_len
    {
        return Err(HtSegmentLengthError::MultiPassLengthMismatch {
            data_len,
            cleanup_length: ht_cleanup_length,
            refinement_length: ht_refinement_length,
        });
    }
    if !(2..65535).contains(&ht_cleanup_length) {
        return Err(HtSegmentLengthError::CleanupLengthOutOfRange {
            cleanup_length: ht_cleanup_length,
        });
    }
    if ht_refinement_length >= 2047 {
        return Err(HtSegmentLengthError::RefinementLengthOutOfRange {
            refinement_length: ht_refinement_length,
        });
    }

    Ok((ht_cleanup_length, ht_refinement_length))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ht_segment_length_validation_returns_each_semantic_failure() {
        let cases = [
            (
                (0, 1, 1, 0),
                HtSegmentLengthError::EmptyContributionHasSegments,
            ),
            (
                (2, 3, 0, 2),
                HtSegmentLengthError::RefinementOnlyLengthMismatch {
                    data_len: 3,
                    refinement_length: 2,
                },
            ),
            (
                (2, 2047, 0, 2047),
                HtSegmentLengthError::RefinementLengthOutOfRange {
                    refinement_length: 2047,
                },
            ),
            (
                (1, 2, 1, 1),
                HtSegmentLengthError::SinglePassHasRefinement {
                    refinement_length: 1,
                },
            ),
            (
                (1, 3, 2, 0),
                HtSegmentLengthError::SinglePassLengthMismatch {
                    data_len: 3,
                    cleanup_length: 2,
                },
            ),
            (
                (2, 2, 2, 0),
                HtSegmentLengthError::MultiPassRequiresSegments {
                    cleanup_length: 2,
                    refinement_length: 0,
                },
            ),
            (
                (2, 0, u32::MAX, 1),
                HtSegmentLengthError::MultiPassLengthOverflow {
                    cleanup_length: u32::MAX,
                    refinement_length: 1,
                },
            ),
            (
                (2, 5, 2, 2),
                HtSegmentLengthError::MultiPassLengthMismatch {
                    data_len: 5,
                    cleanup_length: 2,
                    refinement_length: 2,
                },
            ),
            (
                (2, 2, 1, 1),
                HtSegmentLengthError::CleanupLengthOutOfRange { cleanup_length: 1 },
            ),
        ];

        for ((passes, data_len, cleanup, refinement), expected) in cases {
            assert_eq!(
                ht_segment_lengths(passes, data_len, cleanup, refinement),
                Err(expected)
            );
        }
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn ht_segment_length_validation_rejects_payloads_above_u32() {
        let data_len = usize::try_from(u64::from(u32::MAX) + 1).expect("64-bit usize");
        assert_eq!(
            ht_segment_lengths(1, data_len, 0, 0),
            Err(HtSegmentLengthError::ContributionLengthExceedsU32 { data_len })
        );
    }

    #[test]
    fn ht_segment_length_validation_accepts_empty_single_and_multi_pass_inputs() {
        assert_eq!(ht_segment_lengths(0, 0, 0, 0), Ok((0, 0)));
        assert_eq!(ht_segment_lengths(1, 5, 0, 0), Ok((5, 0)));
        assert_eq!(ht_segment_lengths(3, 7, 4, 3), Ok((4, 3)));
    }
}
