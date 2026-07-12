// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact source/destination high-water accounting for precinct partitioning.

use alloc::vec::Vec;

use super::super::super::allocation::{
    checked_add_bytes, checked_element_bytes, host_allocation_failed,
};
use super::super::super::{
    NativeEncodePipelineResult, NativeEncodeSession, PreparedEncodeCodeBlock,
    PreparedEncodeSubband, PreparedResolutionPacket,
};
use crate::{EncodeError, EncodeResult};

const SPLIT_OWNERS: &str = "precinct-split prepared packet owners";

#[derive(Clone, Copy, Default)]
struct PreparedTreeOwnership {
    structural_bytes: usize,
    payload_bytes: usize,
}

pub(super) struct PrecinctSplitAccounting<'session, 'input> {
    session: &'session NativeEncodeSession<'input>,
    retained_phase_bytes: usize,
    source_structural_bytes: usize,
    payload_bytes: usize,
    destination_structural_bytes: usize,
    peak_phase_bytes: usize,
}

impl<'session, 'input> PrecinctSplitAccounting<'session, 'input> {
    pub(super) fn try_from_source(
        session: &'session NativeEncodeSession<'input>,
        retained_phase_bytes: usize,
        source: &[Vec<PreparedResolutionPacket>],
        source_capacity: usize,
    ) -> EncodeResult<Self> {
        let source = prepared_tree_ownership(source, source_capacity)?;
        let initial_phase_bytes = checked_add_bytes(
            retained_phase_bytes,
            checked_add_bytes(source.structural_bytes, source.payload_bytes, SPLIT_OWNERS)?,
            SPLIT_OWNERS,
        )?;
        session.checked_phase(initial_phase_bytes, SPLIT_OWNERS)?;
        Ok(Self {
            session,
            retained_phase_bytes,
            source_structural_bytes: source.structural_bytes,
            payload_bytes: source.payload_bytes,
            destination_structural_bytes: 0,
            peak_phase_bytes: initial_phase_bytes,
        })
    }

    pub(super) const fn peak_phase_bytes(&self) -> usize {
        self.peak_phase_bytes
    }

    fn check_destination_add(&mut self, bytes: usize, what: &'static str) -> EncodeResult<()> {
        let destination_bytes = checked_add_bytes(self.destination_structural_bytes, bytes, what)?;
        let phase_bytes = checked_add_bytes(
            self.retained_phase_bytes,
            checked_add_bytes(
                self.payload_bytes,
                checked_add_bytes(self.source_structural_bytes, destination_bytes, what)?,
                what,
            )?,
            what,
        )?;
        self.session.checked_phase(phase_bytes, what)?;
        self.peak_phase_bytes = self.peak_phase_bytes.max(phase_bytes);
        Ok(())
    }

    fn commit_destination_capacity<T>(
        &mut self,
        capacity: usize,
        what: &'static str,
    ) -> EncodeResult<()> {
        let bytes = checked_element_bytes::<T>(capacity, what)?;
        self.check_destination_add(bytes, what)?;
        self.destination_structural_bytes =
            checked_add_bytes(self.destination_structural_bytes, bytes, what)?;
        Ok(())
    }

    pub(super) fn release_source_capacity<T>(
        &mut self,
        capacity: usize,
        what: &'static str,
    ) -> EncodeResult<()> {
        let bytes = checked_element_bytes::<T>(capacity, what)?;
        self.source_structural_bytes = self.source_structural_bytes.checked_sub(bytes).ok_or(
            EncodeError::InternalInvariant {
                what: "precinct split source ownership accounting underflowed",
            },
        )?;
        Ok(())
    }

    pub(super) fn finish(
        mut self,
        output: &[Vec<PreparedResolutionPacket>],
        output_capacity: usize,
    ) -> EncodeResult<usize> {
        if self.source_structural_bytes != 0 {
            return Err(EncodeError::InternalInvariant {
                what: "precinct split retained source structural owners",
            });
        }
        let output_ownership = prepared_tree_ownership(output, output_capacity)?;
        if output_ownership.structural_bytes != self.destination_structural_bytes
            || output_ownership.payload_bytes != self.payload_bytes
        {
            return Err(EncodeError::InternalInvariant {
                what: "precinct split output ownership accounting mismatch",
            });
        }
        self.check_destination_add(0, SPLIT_OWNERS)?;
        Ok(self.peak_phase_bytes)
    }
}

pub(super) fn try_destination_vec<T>(
    count: usize,
    accounting: &mut PrecinctSplitAccounting<'_, '_>,
    what: &'static str,
) -> NativeEncodePipelineResult<Vec<T>> {
    let requested_bytes = checked_element_bytes::<T>(count, what)?;
    accounting.check_destination_add(requested_bytes, what)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| host_allocation_failed(what, requested_bytes))?;
    accounting.commit_destination_capacity::<T>(values.capacity(), what)?;
    Ok(values)
}

pub(super) fn try_push_planned<T>(values: &mut Vec<T>, value: T) -> NativeEncodePipelineResult<()> {
    if values.len() == values.capacity() {
        return Err(EncodeError::InternalInvariant {
            what: "precinct split exceeded a fallible allocation plan",
        }
        .into());
    }
    values.push(value);
    Ok(())
}

fn prepared_tree_ownership(
    packets: &[Vec<PreparedResolutionPacket>],
    outer_capacity: usize,
) -> EncodeResult<PreparedTreeOwnership> {
    let mut ownership = PreparedTreeOwnership::default();
    add_structural_capacity::<Vec<PreparedResolutionPacket>>(
        &mut ownership,
        outer_capacity,
        "prepared component packet owners",
    )?;
    for component_packets in packets {
        add_structural_capacity::<PreparedResolutionPacket>(
            &mut ownership,
            component_packets.capacity(),
            "prepared resolution packet owners",
        )?;
        for packet in component_packets {
            add_structural_capacity::<PreparedEncodeSubband>(
                &mut ownership,
                packet.subbands.capacity(),
                "prepared subband owners",
            )?;
            for subband in &packet.subbands {
                add_structural_capacity::<PreparedEncodeCodeBlock>(
                    &mut ownership,
                    subband.code_blocks.capacity(),
                    "prepared code-block owners",
                )?;
                for block in &subband.code_blocks {
                    let coefficient_bytes = match &block.coefficients {
                        super::super::super::PreparedCodeBlockCoefficients::I32(values) => {
                            checked_element_bytes::<i32>(
                                values.capacity(),
                                "prepared i32 code-block coefficients",
                            )?
                        }
                        super::super::super::PreparedCodeBlockCoefficients::I64(values) => {
                            checked_element_bytes::<i64>(
                                values.capacity(),
                                "prepared i64 code-block coefficients",
                            )?
                        }
                        super::super::super::PreparedCodeBlockCoefficients::Empty => 0,
                    };
                    ownership.payload_bytes = checked_add_bytes(
                        ownership.payload_bytes,
                        coefficient_bytes,
                        "prepared code-block coefficients",
                    )?;
                }
                if let Some(blocks) = &subband.preencoded_ht_code_blocks {
                    add_structural_capacity::<crate::EncodedHtJ2kCodeBlock>(
                        &mut ownership,
                        blocks.capacity(),
                        "preencoded HT code-block owners",
                    )?;
                    for block in blocks {
                        ownership.payload_bytes = checked_add_bytes(
                            ownership.payload_bytes,
                            block.data.capacity(),
                            "preencoded HT payload",
                        )?;
                    }
                }
            }
        }
    }
    Ok(ownership)
}

fn add_structural_capacity<T>(
    ownership: &mut PreparedTreeOwnership,
    capacity: usize,
    what: &'static str,
) -> EncodeResult<()> {
    ownership.structural_bytes = checked_add_bytes(
        ownership.structural_bytes,
        checked_element_bytes::<T>(capacity, what)?,
        what,
    )?;
    Ok(())
}
