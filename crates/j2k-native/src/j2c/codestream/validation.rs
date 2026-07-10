// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{Header, QuantizationStyle};
use crate::error::{bail, Result, ValidationError};
use crate::math::bit_width_u32;

pub(super) fn skipped_levels_to_reach_target(source: u32, target: u32) -> u8 {
    if source <= target {
        return 0;
    }

    let shrink_ratio = source.div_ceil(target);
    if shrink_ratio <= 1 {
        0
    } else {
        bit_width_u32(shrink_ratio - 1)
    }
}

pub(super) fn validate(header: &Header<'_>) -> Result<()> {
    for info in &header.component_infos {
        let max_resolution_idx = info.coding_style.parameters.num_resolution_levels - 1;
        let quantization_style = info.quantization_info.quantization_style;
        let num_precinct_exponents = info.quantization_info.step_sizes.len();

        if num_precinct_exponents == 0 {
            bail!(ValidationError::MissingPrecinctExponents);
        } else if matches!(
            quantization_style,
            QuantizationStyle::NoQuantization | QuantizationStyle::ScalarExpounded
        ) {
            // See the accesses in the `exponent_mantissa` method. The largest
            // access is 1 + (max_resolution_idx - 1) * 3 + 2.

            if max_resolution_idx == 0 {
                if num_precinct_exponents == 0 {
                    bail!(ValidationError::InsufficientExponents);
                }
            } else if 1 + (max_resolution_idx as usize - 1) * 3 + 2 >= num_precinct_exponents {
                bail!(ValidationError::InsufficientExponents);
            }
        }
    }

    Ok(())
}
