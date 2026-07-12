// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reversible transform guard-bit and marker-exponent reconciliation.

use super::super::{
    QuantStepSize, Vec, MAX_CLASSIC_REVERSIBLE_MARKER_BITPLANES, MAX_REVERSIBLE_NO_QUANT_EXPONENT,
    MAX_REVERSIBLE_NO_QUANT_GUARD_BITS,
};

pub(in crate::j2c::encode) fn reversible_guard_bits_for_marker_limit(
    bit_depth: u8,
    num_levels: u8,
    requested_guard_bits: u8,
) -> Result<u8, &'static str> {
    if requested_guard_bits > MAX_REVERSIBLE_NO_QUANT_GUARD_BITS {
        return Err("reversible guard bits exceed the Part 1 marker field");
    }
    let max_reversible_gain = if num_levels == 0 { 0 } else { 2 };
    let requested_bitplanes = u16::from(requested_guard_bits)
        .checked_add(u16::from(bit_depth))
        .and_then(|value| value.checked_add(max_reversible_gain))
        .and_then(|value| value.checked_sub(1))
        .ok_or("reversible no-quantization bitplane count underflows")?;
    if requested_bitplanes > MAX_CLASSIC_REVERSIBLE_MARKER_BITPLANES {
        return Err("25-38 bit reversible encode exceeds the current no-quantization guard/exponent signaling limit");
    }
    let min_guard_bits = requested_bitplanes.saturating_sub(MAX_REVERSIBLE_NO_QUANT_EXPONENT - 1);
    let guard_bits = requested_guard_bits
        .max(u8::try_from(min_guard_bits).map_err(|_| "reversible guard bits exceed u8")?);
    if guard_bits > MAX_REVERSIBLE_NO_QUANT_GUARD_BITS {
        return Err("reversible guard bits exceed the Part 1 marker field");
    }
    Ok(guard_bits)
}

pub(in crate::j2c::encode) fn adjust_component_step_sizes_for_guard_delta(
    component_step_sizes: &mut [Vec<QuantStepSize>],
    guard_delta: u8,
) -> Result<(), &'static str> {
    for step_sizes in component_step_sizes {
        adjust_reversible_step_sizes_for_guard_delta(step_sizes, guard_delta)?;
    }
    Ok(())
}

pub(in crate::j2c::encode) fn adjust_reversible_step_sizes_for_guard_delta(
    step_sizes: &mut [QuantStepSize],
    guard_delta: u8,
) -> Result<(), &'static str> {
    let guard_delta = u16::from(guard_delta);
    for step in step_sizes {
        step.exponent = step
            .exponent
            .checked_sub(guard_delta)
            .ok_or("reversible no-quantization exponent underflows guard-bit adjustment")?;
        if step.exponent > MAX_REVERSIBLE_NO_QUANT_EXPONENT {
            return Err("reversible no-quantization exponent exceeds the Part 1 marker field");
        }
    }
    Ok(())
}
