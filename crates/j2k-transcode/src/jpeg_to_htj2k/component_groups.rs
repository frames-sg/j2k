// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{JpegDctComponent, JpegToHtj2kError};
use crate::allocation::{try_vec_filled, try_vec_reserve_len, try_vec_with_capacity};

pub(super) fn same_geometry_component_groups(
    components: &[JpegDctComponent],
) -> Result<Vec<Vec<usize>>, JpegToHtj2kError> {
    let mut assigned = try_vec_filled(components.len(), false)?;
    let mut groups = try_vec_with_capacity(components.len())?;

    for component_index in 0..components.len() {
        if assigned[component_index] {
            continue;
        }
        assigned[component_index] = true;
        let mut group = try_vec_with_capacity(1)?;
        group.push(component_index);
        for candidate_index in component_index + 1..components.len() {
            if !assigned[candidate_index]
                && same_component_geometry(
                    &components[component_index],
                    &components[candidate_index],
                )
            {
                assigned[candidate_index] = true;
                let required_len =
                    group
                        .len()
                        .checked_add(1)
                        .ok_or(JpegToHtj2kError::Validation(
                            "component group length overflow",
                        ))?;
                try_vec_reserve_len(&mut group, required_len)?;
                group.push(candidate_index);
            }
        }
        groups.push(group);
    }

    Ok(groups)
}

fn same_component_geometry(left: &JpegDctComponent, right: &JpegDctComponent) -> bool {
    left.width == right.width
        && left.height == right.height
        && left.block_cols == right.block_cols
        && left.block_rows == right.block_rows
}
