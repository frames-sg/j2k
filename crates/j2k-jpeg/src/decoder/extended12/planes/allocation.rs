// SPDX-License-Identifier: MIT OR Apache-2.0

//! Extended-precision plane layout, fallible allocation, and live-cap checks.

use alloc::vec::Vec;

use super::super::super::{
    checked_scratch_len, JpegError, PreparedDecodePlan, PreparedProgressivePlan, SofKind,
};
use crate::allocation::{
    checked_add_allocation_bytes, checked_allocation_bytes, try_reserve_for_len_with_live_budget,
};
use crate::entropy::progressive::ProgressiveDctBlocks;

pub(in crate::decoder::extended12) struct Extended12Plane {
    pub(in crate::decoder::extended12) pixels: Vec<u16>,
    pub(in crate::decoder::extended12) stride: usize,
    pub(in crate::decoder::extended12) width: usize,
}

#[derive(Clone, Copy, Default)]
struct Extended12PlaneSpec {
    len: usize,
    stride: usize,
    width: usize,
}

fn preflight_plane_specs(specs: &[Extended12PlaneSpec]) -> Result<usize, JpegError> {
    let mut total = 0;
    for spec in specs {
        total = checked_add_allocation_bytes(total, checked_allocation_bytes::<u16>(spec.len)?)?;
    }
    Ok(total)
}

fn allocate_plane(
    spec: Extended12PlaneSpec,
    live_bytes: &mut usize,
    cap: usize,
) -> Result<Extended12Plane, JpegError> {
    let mut pixels = Vec::new();
    try_reserve_for_len_with_live_budget(&mut pixels, spec.len, live_bytes, cap)?;
    pixels.resize(spec.len, 0u16);
    Ok(Extended12Plane {
        pixels,
        stride: spec.stride,
        width: spec.width,
    })
}

fn allocate_three_planes(
    specs: [Extended12PlaneSpec; 3],
    initial_live_bytes: usize,
    cap: usize,
) -> Result<[Extended12Plane; 3], JpegError> {
    let planned = preflight_plane_specs(&specs)?;
    ensure_phase_capacity(initial_live_bytes, planned, cap)?;
    let mut live_bytes = initial_live_bytes;
    Ok([
        allocate_plane(specs[0], &mut live_bytes, cap)?,
        allocate_plane(specs[1], &mut live_bytes, cap)?,
        allocate_plane(specs[2], &mut live_bytes, cap)?,
    ])
}

fn allocate_four_planes(
    specs: [Extended12PlaneSpec; 4],
    initial_live_bytes: usize,
    cap: usize,
) -> Result<[Extended12Plane; 4], JpegError> {
    let planned = preflight_plane_specs(&specs)?;
    ensure_phase_capacity(initial_live_bytes, planned, cap)?;
    let mut live_bytes = initial_live_bytes;
    Ok([
        allocate_plane(specs[0], &mut live_bytes, cap)?,
        allocate_plane(specs[1], &mut live_bytes, cap)?,
        allocate_plane(specs[2], &mut live_bytes, cap)?,
        allocate_plane(specs[3], &mut live_bytes, cap)?,
    ])
}

fn ensure_phase_capacity(initial: usize, additional: usize, cap: usize) -> Result<(), JpegError> {
    let requested = initial
        .checked_add(additional)
        .ok_or(JpegError::MemoryCapExceeded {
            requested: usize::MAX,
            cap,
        })?;
    if requested > cap {
        return Err(JpegError::MemoryCapExceeded { requested, cap });
    }
    Ok(())
}

fn plane_capacity_bytes(planes: &[Extended12Plane]) -> Result<usize, JpegError> {
    let mut requested = 0;
    for plane in planes {
        requested = checked_add_allocation_bytes(
            requested,
            checked_allocation_bytes::<u16>(plane.pixels.capacity())?,
        )?;
    }
    Ok(requested)
}

pub(super) fn ensure_progressive_render_capacities(
    dct_blocks: &ProgressiveDctBlocks,
    planes: &[Extended12Plane],
    cap: usize,
) -> Result<(), JpegError> {
    let requested =
        checked_add_allocation_bytes(dct_blocks.capacity_bytes()?, plane_capacity_bytes(planes)?)?;
    if requested > cap {
        return Err(JpegError::MemoryCapExceeded { requested, cap });
    }
    Ok(())
}

pub(in crate::decoder::extended12) fn ensure_progressive12_coefficient_capacities(
    dct_blocks: &ProgressiveDctBlocks,
    cap: usize,
) -> Result<(), JpegError> {
    let requested = dct_blocks.capacity_bytes()?;
    if requested > cap {
        return Err(JpegError::MemoryCapExceeded { requested, cap });
    }
    Ok(())
}

fn sequential_plane_specs<const N: usize>(
    plan: &PreparedDecodePlan,
    sof: SofKind,
) -> Result<[Extended12PlaneSpec; N], JpegError> {
    let mcu_cols = plan
        .dimensions
        .0
        .div_ceil(u32::from(plan.sampling.max_h) * 8);
    let mcu_rows = plan
        .dimensions
        .1
        .div_ceil(u32::from(plan.sampling.max_v) * 8);
    let mut specs = [Extended12PlaneSpec::default(); N];
    for component in &plan.components {
        if component.output_index >= N {
            return Err(JpegError::NotImplemented { sof });
        }
        let stride = checked_scratch_len(&[mcu_cols as usize, usize::from(component.h), 8])?;
        let height = checked_scratch_len(&[mcu_rows as usize, usize::from(component.v), 8])?;
        let len = checked_scratch_len(&[stride, height, core::mem::size_of::<u16>()])?
            / core::mem::size_of::<u16>();
        specs[component.output_index] = Extended12PlaneSpec {
            len,
            stride,
            width: plan
                .dimensions
                .0
                .saturating_mul(u32::from(component.h))
                .div_ceil(u32::from(plan.sampling.max_h)) as usize,
        };
    }
    Ok(specs)
}

pub(super) fn extended12_planes_for_sequential_plan(
    plan: &PreparedDecodePlan,
    sof: SofKind,
) -> Result<[Extended12Plane; 3], JpegError> {
    allocate_three_planes(sequential_plane_specs(plan, sof)?, 0, plan.scratch_bytes)
}

pub(super) fn extended12_four_component_planes_for_sequential_plan(
    plan: &PreparedDecodePlan,
    sof: SofKind,
) -> Result<[Extended12Plane; 4], JpegError> {
    allocate_four_planes(sequential_plane_specs(plan, sof)?, 0, plan.scratch_bytes)
}

fn progressive_plane_specs<const N: usize>(
    plan: &PreparedProgressivePlan,
) -> Result<[Extended12PlaneSpec; N], JpegError> {
    let mut specs = [Extended12PlaneSpec::default(); N];
    for component in &plan.components {
        if component.output_index >= N {
            return Err(JpegError::NotImplemented {
                sof: SofKind::Progressive12,
            });
        }
        let stride = checked_scratch_len(&[component.block_cols as usize, 8])?;
        let height = checked_scratch_len(&[component.block_rows as usize, 8])?;
        let len = checked_scratch_len(&[stride, height, core::mem::size_of::<u16>()])?
            / core::mem::size_of::<u16>();
        specs[component.output_index] = Extended12PlaneSpec {
            len,
            stride,
            width: component.sample_width as usize,
        };
    }
    Ok(specs)
}

pub(super) fn progressive12_color_planes(
    plan: &PreparedProgressivePlan,
    initial_live_bytes: usize,
) -> Result<[Extended12Plane; 3], JpegError> {
    allocate_three_planes(
        progressive_plane_specs(plan)?,
        initial_live_bytes,
        plan.scratch_bytes,
    )
}

pub(super) fn progressive12_four_component_planes(
    plan: &PreparedProgressivePlan,
    initial_live_bytes: usize,
) -> Result<[Extended12Plane; 4], JpegError> {
    allocate_four_planes(
        progressive_plane_specs(plan)?,
        initial_live_bytes,
        plan.scratch_bytes,
    )
}

#[cfg(test)]
mod tests {
    use super::{preflight_plane_specs, Extended12PlaneSpec, JpegError};

    #[test]
    fn extended12_planes_share_an_exact_aggregate_cap_boundary() {
        let cap = j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;
        let elements = cap / (2 * core::mem::size_of::<u16>());
        let exact = [
            Extended12PlaneSpec {
                len: elements,
                ..Extended12PlaneSpec::default()
            },
            Extended12PlaneSpec {
                len: elements,
                ..Extended12PlaneSpec::default()
            },
        ];
        assert_eq!(
            preflight_plane_specs(&exact).expect("exact aggregate boundary"),
            cap
        );

        let over = [
            exact[0],
            Extended12PlaneSpec {
                len: elements + 1,
                ..Extended12PlaneSpec::default()
            },
        ];
        assert!(matches!(
            preflight_plane_specs(&over),
            Err(JpegError::MemoryCapExceeded { requested, cap: limit })
                if requested > limit && limit == cap
        ));
    }
}
