// SPDX-License-Identifier: MIT OR Apache-2.0

//! Aggregate allocation accounting for main-header marker metadata.

use alloc::vec::Vec;
use core::mem::size_of;

use super::super::{
    CodingStyleComponent, CodingStyleDefault, ComponentInfo, ComponentSizeInfo, PacketLengthMarker,
    PpmMarkerData, PpmPacket, ProgressionChange, QuantizationInfo, StepSize,
};
use crate::error::{DecodeError, Result, ValidationError};
use crate::{try_reserve_decode_elements, DEFAULT_MAX_DECODE_BYTES};

const HEADER_ALLOCATION_WHAT: &str = "native codestream header metadata";

const fn allocation_too_large(requested: usize) -> DecodeError {
    DecodeError::AllocationTooLarge {
        what: HEADER_ALLOCATION_WHAT,
        requested,
        cap: DEFAULT_MAX_DECODE_BYTES,
    }
}

const fn allocation_overflow() -> DecodeError {
    allocation_too_large(usize::MAX)
}

pub(super) fn try_none_vec<T>(
    len: usize,
    budget: &mut HeaderMarkerBudget,
) -> Result<Vec<Option<T>>> {
    let mut values = Vec::new();
    budget.try_reserve_len(&mut values, len)?;
    values.resize_with(len, || None);
    Ok(values)
}

#[derive(Debug)]
pub(super) struct HeaderMarkerBudget {
    owned_bytes: usize,
}

impl Default for HeaderMarkerBudget {
    fn default() -> Self {
        Self {
            owned_bytes: size_of::<super::super::Header<'static>>(),
        }
    }
}

impl HeaderMarkerBudget {
    pub(super) fn with_retained_baseline(retained_baseline_bytes: usize) -> Result<Self> {
        let owned_bytes = retained_baseline_bytes
            .checked_add(size_of::<super::super::Header<'static>>())
            .ok_or_else(allocation_overflow)?;
        if owned_bytes > DEFAULT_MAX_DECODE_BYTES {
            return Err(allocation_too_large(owned_bytes));
        }
        Ok(Self { owned_bytes })
    }

    pub(super) fn remaining_bytes(&self) -> usize {
        DEFAULT_MAX_DECODE_BYTES - self.owned_bytes
    }

    pub(super) fn try_reserve_next<T>(&mut self, values: &mut Vec<T>) -> Result<()> {
        let target_len = values
            .len()
            .checked_add(1)
            .ok_or_else(allocation_overflow)?;
        self.try_reserve_len(values, target_len)
    }

    pub(super) fn try_reserve_len<T>(
        &mut self,
        values: &mut Vec<T>,
        target_len: usize,
    ) -> Result<()> {
        if target_len <= values.capacity() {
            return Ok(());
        }
        let retained_bytes = checked_element_bytes::<T>(values.capacity())?;
        let replacement_bytes = checked_element_bytes::<T>(target_len)?;
        self.ensure_additional(replacement_bytes)?;
        try_reserve_decode_elements(values, target_len)?;
        let actual_bytes = checked_element_bytes::<T>(values.capacity())?;
        let retained_released = self
            .owned_bytes
            .checked_sub(retained_bytes)
            .ok_or(ValidationError::ImageTooLarge)?;
        self.owned_bytes = retained_released
            .checked_add(actual_bytes)
            .ok_or_else(allocation_overflow)?;
        if self.owned_bytes > DEFAULT_MAX_DECODE_BYTES {
            return Err(allocation_too_large(self.owned_bytes));
        }
        Ok(())
    }

    pub(super) fn account_capacity<T>(&mut self, capacity: usize) -> Result<()> {
        let additional = checked_element_bytes::<T>(capacity)?;
        self.ensure_additional(additional)?;
        self.owned_bytes += additional;
        Ok(())
    }

    pub(super) fn account_capacity_overage<T>(
        &mut self,
        planned_count: usize,
        actual_capacity: usize,
    ) -> Result<()> {
        if actual_capacity > planned_count {
            self.account_capacity::<T>(actual_capacity - planned_count)?;
        }
        Ok(())
    }

    pub(super) fn release_capacity<T>(&mut self, capacity: usize) -> Result<()> {
        let released = checked_element_bytes::<T>(capacity)?;
        self.owned_bytes = self
            .owned_bytes
            .checked_sub(released)
            .ok_or(ValidationError::ImageTooLarge)?;
        Ok(())
    }

    pub(super) fn account_elements<T>(&mut self, count: usize) -> Result<()> {
        let additional = count
            .checked_mul(size_of::<T>())
            .ok_or_else(allocation_overflow)?;
        let owned_bytes = self
            .owned_bytes
            .checked_add(additional)
            .ok_or_else(allocation_overflow)?;
        if owned_bytes > DEFAULT_MAX_DECODE_BYTES {
            return Err(allocation_too_large(owned_bytes));
        }
        self.owned_bytes = owned_bytes;
        Ok(())
    }

    fn ensure_additional(&self, additional: usize) -> Result<()> {
        let peak = self
            .owned_bytes
            .checked_add(additional)
            .ok_or_else(allocation_overflow)?;
        if peak > DEFAULT_MAX_DECODE_BYTES {
            return Err(allocation_too_large(peak));
        }
        Ok(())
    }
}

fn checked_element_bytes<T>(count: usize) -> Result<usize> {
    count
        .checked_mul(size_of::<T>())
        .ok_or_else(allocation_overflow)
}

pub(super) fn account_component_metadata_peak(
    component_sizes: &[ComponentSizeInfo],
    coding_overrides: &[Option<CodingStyleComponent>],
    quantization_overrides: &[Option<QuantizationInfo>],
    coding_default: &CodingStyleDefault,
    quantization_default: &QuantizationInfo,
    budget: &mut HeaderMarkerBudget,
) -> Result<()> {
    let component_count = component_sizes.len();
    budget.account_elements::<ComponentInfo>(component_count)?;
    for idx in 0..component_count {
        let coding = coding_overrides[idx]
            .as_ref()
            .unwrap_or(&coding_default.component_parameters);
        let quantization = quantization_overrides[idx]
            .as_ref()
            .unwrap_or(quantization_default);
        budget.account_elements::<(u8, u8)>(coding.parameters.precinct_exponents.len())?;
        budget.account_elements::<StepSize>(quantization.step_sizes.len())?;
    }
    Ok(())
}

pub(super) fn try_extend_progression_changes(
    destination: &mut Vec<ProgressionChange>,
    source: Vec<ProgressionChange>,
    budget: &mut HeaderMarkerBudget,
) -> Result<()> {
    let source_len = source.len();
    let source_capacity = source.capacity();
    let target_len = destination
        .len()
        .checked_add(source_len)
        .ok_or_else(allocation_overflow)?;
    budget.account_capacity::<ProgressionChange>(source_capacity)?;
    budget.try_reserve_len(destination, target_len)?;
    destination.extend(source);
    budget.release_capacity::<ProgressionChange>(source_capacity)?;
    Ok(())
}

pub(super) fn try_flatten_packet_lengths(
    markers: Vec<PacketLengthMarker>,
    budget: &mut HeaderMarkerBudget,
) -> Result<Vec<u32>> {
    let marker_capacity = markers.capacity();
    let source_packet_capacity = markers.iter().try_fold(0_usize, |total, marker| {
        total
            .checked_add(marker.packet_lengths.capacity())
            .ok_or_else(allocation_overflow)
    })?;
    let packet_count = markers.iter().try_fold(0_usize, |total, marker| {
        total
            .checked_add(marker.packet_lengths.len())
            .ok_or_else(allocation_overflow)
    })?;
    let mut packet_lengths = Vec::new();
    budget.try_reserve_len(&mut packet_lengths, packet_count)?;
    for mut marker in markers {
        packet_lengths.append(&mut marker.packet_lengths);
    }
    budget.release_capacity::<PacketLengthMarker>(marker_capacity)?;
    budget.release_capacity::<u32>(source_packet_capacity)?;
    Ok(packet_lengths)
}

pub(super) fn try_flatten_ppm_packets<'a>(
    markers: Vec<PpmMarkerData<'a>>,
    budget: &mut HeaderMarkerBudget,
) -> Result<Vec<PpmPacket<'a>>> {
    let marker_capacity = markers.capacity();
    let source_packet_capacity = markers.iter().try_fold(0_usize, |total, marker| {
        total
            .checked_add(marker.packets.capacity())
            .ok_or_else(allocation_overflow)
    })?;
    let packet_count = markers.iter().try_fold(0_usize, |total, marker| {
        let nonempty_count = marker
            .packets
            .iter()
            .filter(|packet| !packet.data.is_empty())
            .count();
        total
            .checked_add(nonempty_count)
            .ok_or_else(allocation_overflow)
    })?;
    let mut packets = Vec::new();
    budget.try_reserve_len(&mut packets, packet_count)?;
    for marker in markers {
        packets.extend(
            marker
                .packets
                .into_iter()
                .filter(|packet| !packet.data.is_empty()),
        );
    }
    budget.release_capacity::<PpmMarkerData<'_>>(marker_capacity)?;
    budget.release_capacity::<PpmPacket<'_>>(source_packet_capacity)?;
    Ok(packets)
}

#[cfg(test)]
mod tests;
