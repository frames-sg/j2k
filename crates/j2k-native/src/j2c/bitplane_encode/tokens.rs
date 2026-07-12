// SPDX-License-Identifier: MIT OR Apache-2.0

// Token packing for externally generated classic Tier-1 symbols.

use core::mem::size_of;

use super::super::arithmetic_encoder::{ArithmeticEncoder, ArithmeticEncoderContext};
#[cfg(test)]
use super::super::coefficient_view::legacy_coefficient_view_error;
use super::super::encode::allocation::try_untracked_vec;
use crate::writer::BitWriter;
use crate::{EncodeError, EncodeResult, DEFAULT_MAX_CODEC_BYTES};

use super::segments::{reset_contexts, try_push_segment, PendingSegment};
use super::{EncodedCodeBlockSegment, EncodedCodeBlockWithSegments};

mod reader;
use reader::ClassicTier1TokenReader;

pub(crate) type ClassicTier1TokenSegment = crate::J2kTier1TokenSegment;

pub(crate) fn try_pack_classic_selective_bypass_tier1_tokens(
    token_bytes: &[u8],
    token_segments: &[ClassicTier1TokenSegment],
    number_of_coding_passes: u8,
    missing_bit_planes: u8,
) -> EncodeResult<EncodedCodeBlockWithSegments> {
    let allocation = token_allocation(token_segments, number_of_coding_passes)?;
    let mut reader = ClassicTier1TokenReader::new(token_bytes);
    let mut contexts = [ArithmeticEncoderContext::default(); 19];
    reset_contexts(&mut contexts);
    let mut data = try_untracked_vec(
        allocation.payload_bytes.min(256),
        "classic Tier-1 token payload",
    )?;
    let mut segments = try_untracked_vec(
        token_segments.len(),
        "classic Tier-1 token segment metadata",
    )?;

    for segment in token_segments {
        if segment.start_coding_pass >= segment.end_coding_pass {
            return Err(EncodeError::InvalidInput {
                what: "classic Tier-1 token segment pass range is invalid",
            });
        }
        if segment.end_coding_pass > number_of_coding_passes {
            return Err(EncodeError::InvalidInput {
                what: "classic Tier-1 token segment exceeds coding passes",
            });
        }
        let token_bit_offset =
            usize::try_from(segment.token_bit_offset).map_err(|_| EncodeError::InvalidInput {
                what: "classic Tier-1 token bit offset exceeds usize",
            })?;
        let token_bit_count =
            usize::try_from(segment.token_bit_count).map_err(|_| EncodeError::InvalidInput {
                what: "classic Tier-1 token bit count exceeds usize",
            })?;
        reader.seek(token_bit_offset).map_err(token_input_error)?;
        if segment.use_arithmetic {
            if token_bit_count % 6 != 0 {
                return Err(EncodeError::InvalidInput {
                    what: "classic Tier-1 MQ token segment is not aligned to 6-bit symbols",
                });
            }
            let symbol_count = token_bit_count / 6;
            let segment_limit = arithmetic_token_payload_bound(symbol_count)?;
            let mut encoder = ArithmeticEncoder::try_with_byte_limit(segment_limit)?;
            for _ in 0..symbol_count {
                let token = reader.read_bits(6).map_err(token_input_error)?;
                let ctx = (token & 0x1F) as usize;
                if ctx >= contexts.len() {
                    return Err(EncodeError::InvalidInput {
                        what: "classic Tier-1 MQ token context is out of range",
                    });
                }
                let bit = (token >> 5) & 1;
                encoder.encode(bit, &mut contexts[ctx]);
            }
            try_push_segment(
                &mut data,
                &mut segments,
                allocation.payload_bytes,
                token_segments.len(),
                PendingSegment {
                    start_coding_pass: segment.start_coding_pass,
                    end_coding_pass: segment.end_coding_pass,
                    data: encoder.finish_checked()?,
                    distortion_delta: f64::EPSILON,
                    use_arithmetic: true,
                },
            )?;
        } else {
            let segment_limit = raw_token_payload_bound(token_bit_count)?;
            let mut writer = BitWriter::try_with_byte_limit(segment_limit)?;
            for _ in 0..token_bit_count {
                writer.write_bit(reader.read_bits(1).map_err(token_input_error)?);
            }
            try_push_segment(
                &mut data,
                &mut segments,
                allocation.payload_bytes,
                token_segments.len(),
                PendingSegment {
                    start_coding_pass: segment.start_coding_pass,
                    end_coding_pass: segment.end_coding_pass,
                    data: writer.finish_checked()?,
                    distortion_delta: f64::EPSILON,
                    use_arithmetic: false,
                },
            )?;
        }
    }

    Ok(EncodedCodeBlockWithSegments {
        data,
        segments,
        num_coding_passes: number_of_coding_passes,
        num_zero_bitplanes: missing_bit_planes,
    })
}

#[cfg(test)]
pub(crate) fn pack_classic_selective_bypass_tier1_tokens(
    token_bytes: &[u8],
    token_segments: &[ClassicTier1TokenSegment],
    number_of_coding_passes: u8,
    missing_bit_planes: u8,
) -> Result<EncodedCodeBlockWithSegments, &'static str> {
    try_pack_classic_selective_bypass_tier1_tokens(
        token_bytes,
        token_segments,
        number_of_coding_passes,
        missing_bit_planes,
    )
    .map_err(legacy_coefficient_view_error)
}

#[derive(Debug, Clone, Copy)]
struct TokenAllocation {
    payload_bytes: usize,
}

fn token_allocation(
    token_segments: &[ClassicTier1TokenSegment],
    number_of_coding_passes: u8,
) -> EncodeResult<TokenAllocation> {
    if token_segments.len() > usize::from(number_of_coding_passes) {
        return Err(EncodeError::InvalidInput {
            what: "classic Tier-1 token segment count exceeds coding passes",
        });
    }
    let mut payload_bytes = 0usize;
    let mut largest_segment = 0usize;
    let mut previous_end = 0u8;
    for segment in token_segments {
        if segment.start_coding_pass >= segment.end_coding_pass
            || segment.end_coding_pass > number_of_coding_passes
        {
            return Err(EncodeError::InvalidInput {
                what: "classic Tier-1 token segment pass range is invalid",
            });
        }
        if segment.start_coding_pass != previous_end {
            return Err(EncodeError::InvalidInput {
                what: "classic Tier-1 token segments do not provide contiguous pass coverage",
            });
        }
        let expected_arithmetic =
            segment.start_coding_pass <= 9 || segment.start_coding_pass % 3 == 0;
        if segment.use_arithmetic != expected_arithmetic {
            return Err(EncodeError::InvalidInput {
                what: "classic Tier-1 token segment uses the wrong coding mode for its pass",
            });
        }
        if !segment.use_arithmetic {
            let pass_count = segment.end_coding_pass - segment.start_coding_pass;
            if segment.start_coding_pass % 3 != 1
                || pass_count > 2
                || (segment.start_coding_pass..segment.end_coding_pass).any(|pass| pass % 3 == 0)
            {
                return Err(EncodeError::InvalidInput {
                    what: "classic Tier-1 raw token segment crosses a cleanup-pass boundary",
                });
            }
        }
        previous_end = segment.end_coding_pass;
        let token_bits =
            usize::try_from(segment.token_bit_count).map_err(|_| EncodeError::InvalidInput {
                what: "classic Tier-1 token bit count exceeds usize",
            })?;
        let segment_bytes = if segment.use_arithmetic {
            if token_bits % 6 != 0 {
                return Err(EncodeError::InvalidInput {
                    what: "classic Tier-1 MQ token segment is not aligned to 6-bit symbols",
                });
            }
            arithmetic_token_payload_bound(token_bits / 6)?
        } else {
            raw_token_payload_bound(token_bits)?
        };
        payload_bytes =
            payload_bytes
                .checked_add(segment_bytes)
                .ok_or(EncodeError::ArithmeticOverflow {
                    what: "classic Tier-1 token payload",
                })?;
        largest_segment = largest_segment.max(segment_bytes);
    }
    if previous_end != number_of_coding_passes {
        return Err(EncodeError::InvalidInput {
            what: "classic Tier-1 token segments do not cover every coding pass",
        });
    }
    let metadata_bytes = token_segments
        .len()
        .checked_mul(size_of::<EncodedCodeBlockSegment>())
        .ok_or(EncodeError::ArithmeticOverflow {
            what: "classic Tier-1 token segment metadata",
        })?;
    let active_segment_bytes =
        largest_segment
            .checked_add(1)
            .ok_or(EncodeError::ArithmeticOverflow {
                what: "classic Tier-1 token active segment",
            })?;
    let requested = payload_bytes
        .checked_add(metadata_bytes)
        .and_then(|bytes| bytes.checked_add(active_segment_bytes))
        .ok_or(EncodeError::ArithmeticOverflow {
            what: "classic Tier-1 token worker allocation",
        })?;
    if requested > DEFAULT_MAX_CODEC_BYTES {
        return Err(EncodeError::AllocationTooLarge {
            what: "classic Tier-1 token worker allocation",
            requested,
            cap: DEFAULT_MAX_CODEC_BYTES,
        });
    }
    Ok(TokenAllocation { payload_bytes })
}

fn arithmetic_token_payload_bound(symbol_count: usize) -> EncodeResult<usize> {
    symbol_count
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_add(4))
        .ok_or(EncodeError::ArithmeticOverflow {
            what: "classic Tier-1 MQ token payload",
        })
}

fn raw_token_payload_bound(bit_count: usize) -> EncodeResult<usize> {
    bit_count
        .checked_add(6)
        .map(|bits| bits / 7 + 1)
        .ok_or(EncodeError::ArithmeticOverflow {
            what: "classic Tier-1 raw token payload",
        })
}

fn token_input_error(what: &'static str) -> EncodeError {
    EncodeError::InvalidInput { what }
}
