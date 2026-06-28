//! Packet-header segment-length math shared between the CPU packetizer and
//! GPU packetization planners.
//!
//! Hidden from the rendered public API: GPU planner crates call these helpers
//! internally so their packet headers stay bit-identical with the CPU
//! encoder's signaling decisions.

/// Returns whether `value` can be signalled in `bits` bits.
#[inline]
pub fn value_fits_in_bits(value: u32, bits: u32) -> bool {
    bits >= u32::BITS || value < (1u32 << bits)
}

/// Calculate number of bits needed to encode a segment length.
#[inline]
pub fn bits_for_length(l_block: u32, num_coding_passes: u8) -> u32 {
    let log2_passes = if num_coding_passes <= 1 {
        0
    } else {
        (num_coding_passes as u32).ilog2()
    };
    l_block + log2_passes
}

/// Number of bits needed to encode an HT cleanup segment length, folding
/// placeholder passes into the pass count.
#[inline]
pub fn bits_for_ht_cleanup_length(l_block: u32, raw_num_passes: u8) -> u32 {
    let placeholder_groups = u32::from(raw_num_passes.saturating_sub(1)) / 3;
    let placeholder_passes = placeholder_groups * 3;
    l_block + (placeholder_passes + 1).ilog2()
}

/// Number of bits needed to encode an HT refinement-only packet
/// contribution length after the cleanup segment has already appeared in an
/// earlier quality layer.
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
) -> Result<(u32, u32), &'static str> {
    if num_coding_passes == 0 {
        if data_len == 0 && ht_cleanup_length == 0 && ht_refinement_length == 0 {
            return Ok((0, 0));
        }
        return Err("empty HTJ2K packet contribution must not carry segment bytes");
    }

    let data_len =
        u32::try_from(data_len).map_err(|_| "HTJ2K packet contribution exceeds u32 length")?;
    if ht_cleanup_length == 0 && ht_refinement_length != 0 {
        if ht_refinement_length != data_len {
            return Err("refinement-only HTJ2K packet contribution length mismatch");
        }
        if ht_refinement_length >= 2047 {
            return Err("HTJ2K refinement segment length is out of range");
        }
        return Ok((0, ht_refinement_length));
    }

    if num_coding_passes == 1 {
        if ht_refinement_length != 0 {
            return Err("single-pass HTJ2K packet contribution must not carry refinement bytes");
        }
        let cleanup_length = if ht_cleanup_length == 0 {
            data_len
        } else {
            ht_cleanup_length
        };
        if cleanup_length != data_len {
            return Err("single-pass HTJ2K packet contribution length mismatch");
        }
        return Ok((cleanup_length, 0));
    }

    if ht_cleanup_length == 0 || ht_refinement_length == 0 {
        return Err("multi-pass HTJ2K packet contribution requires cleanup/refinement lengths");
    }
    if ht_cleanup_length
        .checked_add(ht_refinement_length)
        .ok_or("multi-pass HTJ2K packet contribution length overflow")?
        != data_len
    {
        return Err("multi-pass HTJ2K packet contribution length mismatch");
    }
    if !(2..65535).contains(&ht_cleanup_length) {
        return Err("HTJ2K cleanup segment length is out of range");
    }
    if ht_refinement_length >= 2047 {
        return Err("HTJ2K refinement segment length is out of range");
    }

    Ok((ht_cleanup_length, ht_refinement_length))
}
