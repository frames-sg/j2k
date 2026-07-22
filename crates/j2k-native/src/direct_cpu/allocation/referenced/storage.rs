// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    fill_without_allocation, normalize_outer_owner, try_reserve_decode_elements,
    DirectComponentBandScratch, DirectComponentPlane, DirectCpuBand, J2kDirectCpuScratch, Result,
    Vec,
};
use super::view::{
    referenced_band_target, referenced_component_band_count, referenced_component_plane_len,
    validate_referenced_shape, ReferencedPlanView,
};

pub(super) fn normalize_referenced_scratch(
    plan: ReferencedPlanView<'_>,
    compressed_payload_bytes: usize,
    retain_classic_workspace: bool,
    retain_ht_workspace: bool,
    scratch: &mut J2kDirectCpuScratch,
) -> Result<()> {
    validate_referenced_shape(plan)?;
    let component_count = plan.component_count();
    normalize_outer_owner(&mut scratch.component_band_sets, component_count);
    normalize_outer_owner(&mut scratch.component_planes, component_count);
    fill_without_allocation(
        &mut scratch.component_band_sets,
        component_count,
        DirectComponentBandScratch::default,
    );
    fill_without_allocation(
        &mut scratch.component_planes,
        component_count,
        DirectComponentPlane::default,
    );

    for component_index in 0..component_count {
        let band_count = referenced_component_band_count(plan, component_index)?;
        if let Some(component) = scratch.component_band_sets.get_mut(component_index) {
            normalize_outer_owner(&mut component.bands, band_count);
            fill_without_allocation(&mut component.bands, band_count, DirectCpuBand::empty);
            for band_index in 0..band_count {
                let target_len = referenced_band_target(plan, component_index, band_index)?;
                if let Some(band) = component.bands.get_mut(band_index) {
                    band.coefficients.clear();
                    if band.coefficients.capacity() < target_len {
                        band.coefficients = Vec::new();
                    }
                }
            }
            component.active_len = 0;
        }

        let plane_len = referenced_component_plane_len(plan, component_index)?;
        if let Some(plane) = scratch.component_planes.get_mut(component_index) {
            plane.samples.clear();
            if plane.samples.capacity() < plane_len {
                plane.samples = Vec::new();
            }
        }
    }
    scratch.compressed_payload.clear();
    if scratch.compressed_payload.capacity() < compressed_payload_bytes {
        scratch.compressed_payload = Vec::new();
    }
    if !retain_classic_workspace {
        scratch.classic_workspace = crate::J2kCodeBlockDecodeWorkspace::default();
    }
    if !retain_ht_workspace {
        scratch.ht_workspace = crate::HtCodeBlockDecodeWorkspace::default();
    }
    scratch.staged_state = None;
    Ok(())
}

pub(super) fn reserve_referenced_scratch(
    plan: ReferencedPlanView<'_>,
    compressed_payload_bytes: usize,
    retained_classic_workspace_dimensions: Option<(u32, u32)>,
    retained_ht_workspace_dimensions: Option<(u32, u32)>,
    scratch: &mut J2kDirectCpuScratch,
) -> Result<()> {
    let component_count = plan.component_count();
    try_reserve_decode_elements(&mut scratch.component_band_sets, component_count)?;
    try_reserve_decode_elements(&mut scratch.component_planes, component_count)?;
    while scratch.component_band_sets.len() < component_count {
        scratch
            .component_band_sets
            .push(DirectComponentBandScratch::default());
    }
    while scratch.component_planes.len() < component_count {
        scratch
            .component_planes
            .push(DirectComponentPlane::default());
    }
    for component_index in 0..component_count {
        let band_count = referenced_component_band_count(plan, component_index)?;
        let component = &mut scratch.component_band_sets[component_index];
        try_reserve_decode_elements(&mut component.bands, band_count)?;
        while component.bands.len() < band_count {
            component.bands.push(DirectCpuBand::empty());
        }
        for band_index in 0..band_count {
            try_reserve_decode_elements(
                &mut component.bands[band_index].coefficients,
                referenced_band_target(plan, component_index, band_index)?,
            )?;
        }
        try_reserve_decode_elements(
            &mut scratch.component_planes[component_index].samples,
            referenced_component_plane_len(plan, component_index)?,
        )?;
    }
    try_reserve_decode_elements(&mut scratch.compressed_payload, compressed_payload_bytes)?;
    if let Some((width, height)) = retained_classic_workspace_dimensions {
        scratch.classic_workspace.prepare(width, height)?;
    }
    if let Some((width, height)) = retained_ht_workspace_dimensions {
        scratch.ht_workspace.prepare(width, height)?;
    }
    Ok(())
}
