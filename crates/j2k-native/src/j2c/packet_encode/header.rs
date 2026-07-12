// SPDX-License-Identifier: MIT OR Apache-2.0

use super::state::PacketState;
use super::view::{CodeBlockView, ResolutionView, SubbandView};
#[cfg(test)]
use super::CodeBlockPacketData;
use super::PacketMarkerOptions;
use crate::j2c::codestream::markers;
use crate::j2c::codestream_write::BlockCodingMode;
use crate::j2c::encode::allocation::{
    checked_add_bytes, checked_mul_bytes, BudgetedVec, EncodeAllocationLedger,
};
use crate::packet_math::{
    self, bits_for_ht_cleanup_length, bits_for_ht_refinement_only_length, bits_for_length,
    value_fits_in_bits,
};
use crate::writer::{CheckedBitWriter, FallibleBitWriter};
use crate::{EncodeError, EncodeResult};

const MAX_LENGTH_FIELD_BITS: usize = 256;
const MAX_LENGTH_PREFIX_BITS: usize = u32::BITS as usize + 1;
const MAX_PASS_COUNT_BITS: usize = 16;

pub(super) struct FormedHeader<'a> {
    pub(super) bytes: BudgetedVec<'a, u8>,
    pub(super) body_len: usize,
}

pub(super) fn form_packet_header<'a, R: ResolutionView>(
    packet: &R,
    state: &mut PacketState<'a>,
    layer: u8,
    marker_options: PacketMarkerOptions,
    allocations: &'a EncodeAllocationLedger,
) -> EncodeResult<FormedHeader<'a>> {
    validate_state_layout(packet, state)?;
    let planned_header_bytes = planned_header_bytes(packet, state, layer, marker_options)?;
    let mut writer = CheckedBitWriter::try_with_capacity(
        allocations,
        planned_header_bytes,
        "packet header capacity exhausted",
    )?;

    let any_data = packet.subbands().iter().any(|subband| {
        subband
            .code_blocks()
            .iter()
            .any(|code_block| code_block.num_coding_passes() > 0)
    });
    writer.try_write_bit(u32::from(any_data))?;

    let mut body_len = 0usize;
    if any_data {
        for (packet_subband, state_subband) in
            packet.subbands().iter().zip(state.subbands.iter_mut())
        {
            for (index, packet_block) in packet_subband.code_blocks().iter().enumerate() {
                let index_u32 = u32::try_from(index).map_err(|_| EncodeError::InvalidInput {
                    what: "packet state code-block index exceeds u32",
                })?;
                let x = index_u32 % state_subband.num_cbs_x;
                let y = index_u32 / state_subband.num_cbs_x;
                let state_block = state_subband.code_blocks.get_mut(index).ok_or(
                    EncodeError::InternalInvariant {
                        what: "packet code-block state index exceeded validated layout",
                    },
                )?;

                if !state_block.previously_included {
                    state_subband
                        .inclusion_tree
                        .encode(x, y, u32::from(layer) + 1, &mut writer)?;
                    if packet_block.num_coding_passes() == 0 {
                        continue;
                    }
                    state_subband.zero_bitplane_tree.encode(
                        x,
                        y,
                        u32::from(packet_block.num_zero_bitplanes()) + 1,
                        &mut writer,
                    )?;
                } else if packet_block.num_coding_passes() > 0 {
                    writer.try_write_bit(1)?;
                } else {
                    writer.try_write_bit(0)?;
                    continue;
                }

                if packet_block.num_coding_passes() == 0 {
                    continue;
                }
                let data_len = u32::try_from(packet_block.data().len()).map_err(|_| {
                    EncodeError::InvalidInput {
                        what: "code-block payload length exceeds u32",
                    }
                })?;
                match packet_block.block_coding_mode() {
                    BlockCodingMode::Classic => {
                        encode_num_coding_passes(packet_block.num_coding_passes(), &mut writer)?;
                        encode_classic_segment_lengths_with_lblock(
                            packet_block,
                            data_len,
                            &mut state_block.l_block,
                            &mut writer,
                        )?;
                    }
                    BlockCodingMode::HighThroughput => {
                        encode_num_ht_coding_passes(packet_block.num_coding_passes(), &mut writer)?;
                        encode_ht_segment_lengths_with_lblock(
                            packet_block,
                            &mut state_block.l_block,
                            &mut writer,
                        )?;
                    }
                }
                body_len =
                    checked_add_bytes(body_len, packet_block.data().len(), "packet body length")?;
                state_block.previously_included = true;
            }
        }
    }

    let mut bytes = writer.try_finish()?;
    if bytes.last().copied() == Some(0xff) {
        bytes.try_push(0x00)?;
    }
    if marker_options.write_eph {
        bytes.try_push(0xFF)?;
        bytes.try_push(markers::EPH)?;
    }
    Ok(FormedHeader { bytes, body_len })
}

fn validate_state_layout<R: ResolutionView>(
    packet: &R,
    state: &PacketState<'_>,
) -> EncodeResult<()> {
    if state.subbands.len() != packet.subbands().len() {
        return Err(EncodeError::InvalidInput {
            what: "packet descriptor state layout mismatch",
        });
    }
    for (packet_subband, state_subband) in packet.subbands().iter().zip(state.subbands.iter()) {
        if packet_subband.num_cbs_x() != state_subband.num_cbs_x
            || packet_subband.num_cbs_y() != state_subband.num_cbs_y
            || packet_subband.code_blocks().len() != state_subband.code_blocks.len()
        {
            return Err(EncodeError::InvalidInput {
                what: "packet descriptor state layout mismatch",
            });
        }
    }
    Ok(())
}

fn planned_header_bytes<R: ResolutionView>(
    packet: &R,
    state: &PacketState<'_>,
    layer: u8,
    marker_options: PacketMarkerOptions,
) -> EncodeResult<usize> {
    let mut bits = 1usize;
    for (packet_subband, state_subband) in packet.subbands().iter().zip(state.subbands.iter()) {
        let levels = tag_tree_levels(packet_subband.num_cbs_x(), packet_subband.num_cbs_y());
        for (packet_block, state_block) in packet_subband
            .code_blocks()
            .iter()
            .zip(state_subband.code_blocks.iter())
        {
            if packet_block.num_coding_passes() > 164 {
                return Err(EncodeError::InvalidInput {
                    what: "JPEG 2000 packet contribution exceeds 164 coding passes",
                });
            }
            if state_block.previously_included {
                bits = checked_add_bytes(bits, 1, "packet header bit bound")?;
            } else {
                let inclusion_bits = checked_mul_bytes(
                    levels,
                    usize::from(layer) + 2,
                    "packet inclusion tag-tree bit bound",
                )?;
                bits = checked_add_bytes(bits, inclusion_bits, "packet header bit bound")?;
                if packet_block.num_coding_passes() > 0 {
                    let zero_bitplane_bits = checked_mul_bytes(
                        levels,
                        usize::from(packet_block.num_zero_bitplanes()) + 2,
                        "packet zero-bitplane tag-tree bit bound",
                    )?;
                    bits = checked_add_bytes(bits, zero_bitplane_bits, "packet header bit bound")?;
                }
            }

            if packet_block.num_coding_passes() == 0 {
                continue;
            }
            let _data_len = u32::try_from(packet_block.data().len()).map_err(|_| {
                EncodeError::InvalidInput {
                    what: "code-block payload length exceeds u32",
                }
            })?;
            bits = checked_add_bytes(bits, MAX_PASS_COUNT_BITS, "packet header bit bound")?;
            let segment_count = match packet_block.block_coding_mode() {
                BlockCodingMode::Classic => packet_block.classic_segment_lengths().len().max(1),
                BlockCodingMode::HighThroughput => 2,
            };
            let length_bits = checked_add_bytes(
                MAX_LENGTH_PREFIX_BITS,
                checked_mul_bytes(
                    segment_count,
                    MAX_LENGTH_FIELD_BITS,
                    "packet segment-length bit bound",
                )?,
                "packet segment-length bit bound",
            )?;
            bits = checked_add_bytes(bits, length_bits, "packet header bit bound")?;
        }
    }

    let stuffed_bytes = bits.div_ceil(7);
    let tail_bytes = 1usize + usize::from(marker_options.write_eph) * 2;
    checked_add_bytes(stuffed_bytes, tail_bytes, "packet header byte bound")
}

fn tag_tree_levels(mut width: u32, mut height: u32) -> usize {
    let mut levels = 1usize;
    while width > 1 || height > 1 {
        width = width.div_ceil(2);
        height = height.div_ceil(2);
        levels += 1;
    }
    levels
}

pub(super) fn encode_num_coding_passes(
    num_passes: u8,
    writer: &mut impl FallibleBitWriter,
) -> EncodeResult<()> {
    match num_passes {
        1 => writer.try_write_bit(0),
        2 => writer.try_write_bits(0b10, 2),
        3 => writer.try_write_bits(0b1100, 4),
        4 => writer.try_write_bits(0b1101, 4),
        5 => writer.try_write_bits(0b1110, 4),
        6..=36 => {
            writer.try_write_bits(0b1111, 4)?;
            writer.try_write_bits(u32::from(num_passes - 6), 5)
        }
        37..=164 => {
            writer.try_write_bits(0b1_1111_1111, 9)?;
            writer.try_write_bits(u32::from(num_passes - 37), 7)
        }
        _ => Err(EncodeError::InvalidInput {
            what: "JPEG 2000 packet contribution must contain 1..=164 coding passes",
        }),
    }
}

pub(super) fn encode_num_ht_coding_passes(
    num_passes: u8,
    writer: &mut impl FallibleBitWriter,
) -> EncodeResult<()> {
    match num_passes {
        1 => writer.try_write_bit(0),
        2 => writer.try_write_bits(0b10, 2),
        3..=5 => {
            writer.try_write_bits(0b11, 2)?;
            writer.try_write_bits(u32::from(num_passes - 3), 2)
        }
        6..=36 => {
            writer.try_write_bits(0b11, 2)?;
            writer.try_write_bits(0b11, 2)?;
            writer.try_write_bits(u32::from(num_passes - 6), 5)
        }
        37..=164 => {
            writer.try_write_bits(0b11, 2)?;
            writer.try_write_bits(0b11, 2)?;
            writer.try_write_bits(31, 5)?;
            writer.try_write_bits(u32::from(num_passes - 37), 7)
        }
        _ => Err(EncodeError::InvalidInput {
            what: "HTJ2K packet contribution must contain 1..=164 coding passes",
        }),
    }
}

pub(super) fn encode_length(
    length: u32,
    l_block: &mut u32,
    mut num_bits: u32,
    writer: &mut impl FallibleBitWriter,
) -> EncodeResult<()> {
    while !value_fits_in_bits(length, num_bits) {
        writer.try_write_bit(1)?;
        *l_block = l_block
            .checked_add(1)
            .ok_or(EncodeError::ArithmeticOverflow {
                what: "packet length L-block",
            })?;
        num_bits = num_bits
            .checked_add(1)
            .ok_or(EncodeError::ArithmeticOverflow {
                what: "packet length bit count",
            })?;
    }
    writer.try_write_bit(0)?;
    let num_bits = u8::try_from(num_bits).map_err(|_| EncodeError::InvalidInput {
        what: "packet length bit count exceeds u8",
    })?;
    writer.try_write_bits(length, num_bits)
}

#[cfg(test)]
pub(super) fn encode_classic_segment_lengths(
    code_block: &mut CodeBlockPacketData,
    data_len: u32,
    writer: &mut impl FallibleBitWriter,
) -> EncodeResult<()> {
    let mut l_block = code_block.l_block;
    encode_classic_segment_lengths_with_lblock(code_block, data_len, &mut l_block, writer)?;
    code_block.l_block = l_block;
    Ok(())
}

fn encode_classic_segment_lengths_with_lblock<B: CodeBlockView>(
    code_block: &B,
    data_len: u32,
    l_block: &mut u32,
    writer: &mut impl FallibleBitWriter,
) -> EncodeResult<()> {
    if *l_block > u32::from(u8::MAX) {
        return Err(EncodeError::InvalidInput {
            what: "classic packet L-block exceeds u8",
        });
    }
    if code_block.classic_segment_lengths().is_empty() {
        let num_bits = bits_for_length(*l_block, code_block.num_coding_passes());
        return encode_length(data_len, l_block, num_bits, writer);
    }

    if code_block.classic_segment_lengths().len() != usize::from(code_block.num_coding_passes()) {
        return Err(EncodeError::InvalidInput {
            what: "classic pass-terminated contribution segment count mismatch",
        });
    }
    let segment_sum = code_block
        .classic_segment_lengths()
        .iter()
        .try_fold(0u32, |sum, length| sum.checked_add(*length))
        .ok_or(EncodeError::ArithmeticOverflow {
            what: "classic packet contribution segment length sum",
        })?;
    if segment_sum != data_len {
        return Err(EncodeError::InvalidInput {
            what: "classic packet contribution segment length mismatch",
        });
    }

    let mut required_l_block = *l_block;
    while code_block
        .classic_segment_lengths()
        .iter()
        .any(|&length| !value_fits_in_bits(length, bits_for_length(required_l_block, 1)))
    {
        writer.try_write_bit(1)?;
        required_l_block =
            required_l_block
                .checked_add(1)
                .ok_or(EncodeError::ArithmeticOverflow {
                    what: "classic packet L-block",
                })?;
    }
    writer.try_write_bit(0)?;
    *l_block = required_l_block;

    let length_bits =
        u8::try_from(bits_for_length(*l_block, 1)).map_err(|_| EncodeError::InvalidInput {
            what: "classic segment length bit count exceeds u8",
        })?;
    for &segment_len in code_block.classic_segment_lengths() {
        writer.try_write_bits(segment_len, length_bits)?;
    }
    Ok(())
}

fn encode_ht_segment_lengths_with_lblock<B: CodeBlockView>(
    code_block: &B,
    l_block: &mut u32,
    writer: &mut impl FallibleBitWriter,
) -> EncodeResult<()> {
    if *l_block > u32::from(u8::MAX) {
        return Err(EncodeError::InvalidInput {
            what: "HT packet L-block exceeds u8",
        });
    }
    let (cleanup_length, refinement_length) = ht_segment_lengths(code_block)?;
    if cleanup_length == 0 && refinement_length != 0 {
        let mut refinement_bits =
            bits_for_ht_refinement_only_length(*l_block, code_block.num_coding_passes());
        while !value_fits_in_bits(refinement_length, refinement_bits) {
            writer.try_write_bit(1)?;
            *l_block = l_block
                .checked_add(1)
                .ok_or(EncodeError::ArithmeticOverflow {
                    what: "HT packet L-block",
                })?;
            refinement_bits =
                refinement_bits
                    .checked_add(1)
                    .ok_or(EncodeError::ArithmeticOverflow {
                        what: "HT refinement length bit count",
                    })?;
        }
        writer.try_write_bit(0)?;
        let refinement_bits =
            u8::try_from(refinement_bits).map_err(|_| EncodeError::InvalidInput {
                what: "HT refinement length bit count exceeds u8",
            })?;
        writer.try_write_bits(refinement_length, refinement_bits)?;
        return Ok(());
    }

    let mut cleanup_bits = bits_for_ht_cleanup_length(*l_block, code_block.num_coding_passes());
    let refinement_extra_bits = u32::from(code_block.num_coding_passes() > 2);
    while !value_fits_in_bits(cleanup_length, cleanup_bits)
        || (code_block.num_coding_passes() > 1
            && !value_fits_in_bits(
                refinement_length,
                l_block.checked_add(refinement_extra_bits).ok_or(
                    EncodeError::ArithmeticOverflow {
                        what: "HT refinement length bit count",
                    },
                )?,
            ))
    {
        writer.try_write_bit(1)?;
        *l_block = l_block
            .checked_add(1)
            .ok_or(EncodeError::ArithmeticOverflow {
                what: "HT packet L-block",
            })?;
        cleanup_bits = cleanup_bits
            .checked_add(1)
            .ok_or(EncodeError::ArithmeticOverflow {
                what: "HT cleanup length bit count",
            })?;
    }
    writer.try_write_bit(0)?;
    let cleanup_bits = u8::try_from(cleanup_bits).map_err(|_| EncodeError::InvalidInput {
        what: "HT cleanup length bit count exceeds u8",
    })?;
    writer.try_write_bits(cleanup_length, cleanup_bits)?;

    if code_block.num_coding_passes() > 1 {
        let refinement_bits =
            l_block
                .checked_add(refinement_extra_bits)
                .ok_or(EncodeError::ArithmeticOverflow {
                    what: "HT refinement length bit count",
                })?;
        let refinement_bits =
            u8::try_from(refinement_bits).map_err(|_| EncodeError::InvalidInput {
                what: "HT refinement length bit count exceeds u8",
            })?;
        writer.try_write_bits(refinement_length, refinement_bits)?;
    }
    Ok(())
}

pub(super) fn ht_segment_lengths<B: CodeBlockView>(code_block: &B) -> EncodeResult<(u32, u32)> {
    packet_math::ht_segment_lengths(
        code_block.num_coding_passes(),
        code_block.data().len(),
        code_block.ht_cleanup_length(),
        code_block.ht_refinement_length(),
    )
    .map_err(|error| {
        let what = error.reason();
        match error {
            packet_math::HtSegmentLengthError::ContributionLengthExceedsU32 { .. }
            | packet_math::HtSegmentLengthError::MultiPassLengthOverflow { .. } => {
                EncodeError::ArithmeticOverflow { what }
            }
            packet_math::HtSegmentLengthError::EmptyContributionHasSegments
            | packet_math::HtSegmentLengthError::RefinementOnlyLengthMismatch { .. }
            | packet_math::HtSegmentLengthError::RefinementLengthOutOfRange { .. }
            | packet_math::HtSegmentLengthError::SinglePassHasRefinement { .. }
            | packet_math::HtSegmentLengthError::SinglePassLengthMismatch { .. }
            | packet_math::HtSegmentLengthError::MultiPassRequiresSegments { .. }
            | packet_math::HtSegmentLengthError::MultiPassLengthMismatch { .. }
            | packet_math::HtSegmentLengthError::CleanupLengthOutOfRange { .. } => {
                EncodeError::InvalidInput { what }
            }
        }
    })
}
