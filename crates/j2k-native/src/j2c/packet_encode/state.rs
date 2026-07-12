// SPDX-License-Identifier: MIT OR Apache-2.0

use super::view::{CodeBlockView, DescriptorView, ResolutionView, SubbandView};
use crate::j2c::encode::allocation::{BudgetedVec, EncodeAllocationLedger};
use crate::j2c::tag_tree_encode::TagTreeEncoder;
use crate::{EncodeError, EncodeResult};

pub(super) struct PacketCodeBlockState {
    pub(super) previously_included: bool,
    pub(super) l_block: u32,
    first_inclusion_layer: u32,
    zero_bitplane_value: u32,
}

pub(super) struct PacketSubbandState<'a> {
    pub(super) inclusion_tree: TagTreeEncoder<'a>,
    pub(super) zero_bitplane_tree: TagTreeEncoder<'a>,
    pub(super) code_blocks: BudgetedVec<'a, PacketCodeBlockState>,
    pub(super) num_cbs_x: u32,
    pub(super) num_cbs_y: u32,
}

pub(super) struct PacketState<'a> {
    pub(super) subbands: BudgetedVec<'a, PacketSubbandState<'a>>,
}

struct PacketSubbandStateSeed<'a> {
    num_cbs_x: u32,
    num_cbs_y: u32,
    code_blocks: BudgetedVec<'a, PacketCodeBlockState>,
}

struct PacketStateSeed<'a> {
    subbands: BudgetedVec<'a, Option<PacketSubbandStateSeed<'a>>>,
}

pub(super) fn validate_packet_subband_layout<S: SubbandView>(subband: &S) -> EncodeResult<()> {
    let actual_code_blocks =
        u32::try_from(subband.code_blocks().len()).map_err(|_| EncodeError::InvalidInput {
            what: "packet subband code-block count exceeds u32",
        })?;
    if subband.num_cbs_x() == 0 && subband.num_cbs_y() == 0 && actual_code_blocks == 0 {
        return Ok(());
    }
    let expected_code_blocks = subband.num_cbs_x().checked_mul(subband.num_cbs_y()).ok_or(
        EncodeError::ArithmeticOverflow {
            what: "packet subband code-block grid",
        },
    )?;
    if subband.num_cbs_x() == 0
        || subband.num_cbs_y() == 0
        || expected_code_blocks != actual_code_blocks
    {
        return Err(EncodeError::InvalidInput {
            what: "invalid packet subband code-block layout",
        });
    }
    Ok(())
}

fn packet_state_seed<'a, R: ResolutionView>(
    packet: &R,
    allocations: &'a EncodeAllocationLedger,
) -> EncodeResult<PacketStateSeed<'a>> {
    let mut subbands = allocations.try_vec_with_capacity(
        packet.subbands().len(),
        "packet state seed subband capacity exhausted",
    )?;
    for subband in packet.subbands() {
        validate_packet_subband_layout(subband)?;
        let block_count = subband.code_blocks().len();
        let mut code_blocks = allocations
            .try_vec_with_capacity(block_count, "packet code-block seed capacity exhausted")?;
        for code_block in subband.code_blocks() {
            code_blocks.try_push(PacketCodeBlockState {
                first_inclusion_layer: u32::MAX / 2,
                zero_bitplane_value: 0,
                l_block: code_block.l_block(),
                previously_included: code_block.previously_included(),
            })?;
        }
        subbands.try_push(Some(PacketSubbandStateSeed {
            num_cbs_x: subband.num_cbs_x(),
            num_cbs_y: subband.num_cbs_y(),
            code_blocks,
        }))?;
    }
    Ok(PacketStateSeed { subbands })
}

fn validate_packet_state_layout<R: ResolutionView>(
    seed: &PacketStateSeed<'_>,
    packet: &R,
) -> EncodeResult<()> {
    if seed.subbands.len() != packet.subbands().len() {
        return Err(EncodeError::InvalidInput {
            what: "packet descriptor state layout mismatch",
        });
    }
    for (seed_subband, packet_subband) in seed.subbands.iter().zip(packet.subbands()) {
        let seed_subband = seed_subband
            .as_ref()
            .ok_or(EncodeError::InternalInvariant {
                what: "packet state seed subband was consumed before validation",
            })?;
        if seed_subband.num_cbs_x != packet_subband.num_cbs_x()
            || seed_subband.num_cbs_y != packet_subband.num_cbs_y()
            || seed_subband.code_blocks.len() != packet_subband.code_blocks().len()
        {
            return Err(EncodeError::InvalidInput {
                what: "packet descriptor state layout mismatch",
            });
        }
    }
    Ok(())
}

pub(super) fn build_packet_states<'a, R, D>(
    packets: &[R],
    descriptors: &[D],
    allocations: &'a EncodeAllocationLedger,
) -> EncodeResult<BudgetedVec<'a, PacketState<'a>>>
where
    R: ResolutionView,
    D: DescriptorView,
{
    let state_count = packet_state_count(descriptors)?;
    let seeds = build_packet_state_seeds(packets, descriptors, state_count, allocations)?;
    materialize_packet_states(seeds, state_count, allocations)
}

fn packet_state_count<D: DescriptorView>(descriptors: &[D]) -> EncodeResult<usize> {
    descriptors.iter().try_fold(0usize, |count, descriptor| {
        let state_index =
            usize::try_from(descriptor.state_index()).map_err(|_| EncodeError::InvalidInput {
                what: "packet descriptor state index out of range",
            })?;
        if state_index >= descriptors.len() {
            return Err(EncodeError::InvalidInput {
                what: "packet descriptor state index out of range",
            });
        }
        let required = state_index
            .checked_add(1)
            .ok_or(EncodeError::ArithmeticOverflow {
                what: "packet descriptor state count",
            })?;
        Ok(count.max(required))
    })
}

fn build_packet_state_seeds<'a, R, D>(
    packets: &[R],
    descriptors: &[D],
    state_count: usize,
    allocations: &'a EncodeAllocationLedger,
) -> EncodeResult<BudgetedVec<'a, Option<PacketStateSeed<'a>>>>
where
    R: ResolutionView,
    D: DescriptorView,
{
    let mut seeds = allocations
        .try_vec_with_capacity(state_count, "packet state seed table capacity exhausted")?;
    for _ in 0..state_count {
        seeds.try_push(None)?;
    }

    for descriptor in descriptors {
        let packet_index =
            usize::try_from(descriptor.packet_index()).map_err(|_| EncodeError::InvalidInput {
                what: "packet descriptor packet index out of range",
            })?;
        let packet = packets.get(packet_index).ok_or(EncodeError::InvalidInput {
            what: "packet descriptor packet index out of range",
        })?;
        let state_index =
            usize::try_from(descriptor.state_index()).map_err(|_| EncodeError::InvalidInput {
                what: "packet descriptor state index out of range",
            })?;
        let seed_slot = seeds
            .get_mut(state_index)
            .ok_or(EncodeError::InvalidInput {
                what: "packet descriptor state index out of range",
            })?;
        if let Some(existing) = seed_slot.as_ref() {
            validate_packet_state_layout(existing, packet)?;
        } else {
            *seed_slot = Some(packet_state_seed(packet, allocations)?);
        }

        let seed = seed_slot.as_mut().ok_or(EncodeError::InternalInvariant {
            what: "packet descriptor state initialization failed",
        })?;
        for (seed_subband, packet_subband) in seed.subbands.iter_mut().zip(packet.subbands()) {
            let seed_subband = seed_subband
                .as_mut()
                .ok_or(EncodeError::InternalInvariant {
                    what: "packet state seed subband was consumed before descriptor planning",
                })?;
            for (index, code_block) in packet_subband.code_blocks().iter().enumerate() {
                if code_block.num_coding_passes() == 0 {
                    continue;
                }
                let layer = u32::from(descriptor.layer());
                let block_seed = seed_subband.code_blocks.get_mut(index).ok_or(
                    EncodeError::InternalInvariant {
                        what: "packet inclusion seed index exceeded validated layout",
                    },
                )?;
                if layer < block_seed.first_inclusion_layer {
                    block_seed.first_inclusion_layer = layer;
                    block_seed.zero_bitplane_value = u32::from(code_block.num_zero_bitplanes());
                }
            }
        }
    }
    Ok(seeds)
}

fn materialize_packet_states<'a>(
    mut seeds: BudgetedVec<'a, Option<PacketStateSeed<'a>>>,
    state_count: usize,
    allocations: &'a EncodeAllocationLedger,
) -> EncodeResult<BudgetedVec<'a, PacketState<'a>>> {
    let mut states =
        allocations.try_vec_with_capacity(state_count, "packet state table capacity exhausted")?;
    for state_index in 0..state_count {
        let seed = seeds
            .get_mut(state_index)
            .ok_or(EncodeError::InternalInvariant {
                what: "packet state seed index exceeded planned table",
            })?
            .take();
        let Some(mut seed) = seed else {
            states.try_push(PacketState {
                subbands: allocations
                    .try_vec_with_capacity(0, "empty packet state subband capacity exhausted")?,
            })?;
            continue;
        };

        let mut subbands = allocations.try_vec_with_capacity(
            seed.subbands.len(),
            "packet state subband capacity exhausted",
        )?;
        for seed_subband_slot in seed.subbands.iter_mut() {
            let seed_subband = seed_subband_slot
                .take()
                .ok_or(EncodeError::InternalInvariant {
                    what: "packet state seed subband was already consumed",
                })?;
            let PacketSubbandStateSeed {
                num_cbs_x,
                num_cbs_y,
                code_blocks,
            } = seed_subband;
            let mut inclusion_tree = TagTreeEncoder::try_new(num_cbs_x, num_cbs_y, allocations)?;
            let mut zero_bitplane_tree =
                TagTreeEncoder::try_new(num_cbs_x, num_cbs_y, allocations)?;
            for (index, block_state) in code_blocks.iter().enumerate() {
                let index_u32 = u32::try_from(index).map_err(|_| EncodeError::InvalidInput {
                    what: "packet state code-block index exceeds u32",
                })?;
                let x = index_u32 % num_cbs_x;
                let y = index_u32 / num_cbs_x;
                inclusion_tree.set_value(x, y, block_state.first_inclusion_layer)?;
                zero_bitplane_tree.set_value(x, y, block_state.zero_bitplane_value)?;
            }
            subbands.try_push(PacketSubbandState {
                inclusion_tree,
                zero_bitplane_tree,
                code_blocks,
                num_cbs_x,
                num_cbs_y,
            })?;
        }
        states.try_push(PacketState { subbands })?;
        drop(seed);
    }
    drop(seeds);
    Ok(states)
}
