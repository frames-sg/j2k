// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible exact-i64 raw-pixel deinterleave ownership.

use alloc::vec::Vec;

use super::super::super::allocation::{
    checked_add_bytes, checked_element_bytes, host_allocation_failed,
};
use super::super::super::{
    raw_pixel_bytes_per_sample, read_le_sample_value, sign_extend_sample,
    NativeEncodePipelineError, NativeEncodePipelineResult, NativeEncodeSession,
};

pub(super) struct I64DeinterleaveRequest<'a, 'input> {
    pub(super) pixels: &'a [u8],
    pub(super) num_pixels: usize,
    pub(super) num_components: u16,
    pub(super) bit_depth: u8,
    pub(super) signed: bool,
    pub(super) retained_base_bytes: usize,
    pub(super) session: &'a NativeEncodeSession<'input>,
}

pub(super) fn try_deinterleave_to_i64(
    request: &I64DeinterleaveRequest<'_, '_>,
) -> NativeEncodePipelineResult<Vec<Vec<i64>>> {
    let component_count = usize::from(request.num_components);
    let bytes_per_sample = raw_pixel_bytes_per_sample(request.bit_depth)
        .map_err(NativeEncodePipelineError::invalid_input)?;
    let pixel_stride = component_count.checked_mul(bytes_per_sample).ok_or(
        crate::EncodeError::ArithmeticOverflow {
            what: "exact i64 pixel stride",
        },
    )?;
    let expected_len = request.num_pixels.checked_mul(pixel_stride).ok_or(
        crate::EncodeError::ArithmeticOverflow {
            what: "exact i64 image byte length",
        },
    )?;
    if request.pixels.len() < expected_len {
        return Err(NativeEncodePipelineError::invalid_input(
            "pixel data too short",
        ));
    }
    let outer_requested =
        checked_element_bytes::<Vec<i64>>(component_count, "exact i64 component plane owners")?;
    let one_plane_requested =
        checked_element_bytes::<i64>(request.num_pixels, "exact i64 component samples")?;
    let all_planes_requested = one_plane_requested.checked_mul(component_count).ok_or(
        crate::EncodeError::ArithmeticOverflow {
            what: "exact i64 component samples",
        },
    )?;
    request.session.checked_phase(
        checked_add_bytes(
            request.retained_base_bytes,
            checked_add_bytes(
                outer_requested,
                all_planes_requested,
                "exact i64 component planes",
            )?,
            "exact i64 component planes",
        )?,
        "exact i64 component planes",
    )?;
    let mut components = Vec::new();
    components
        .try_reserve_exact(component_count)
        .map_err(|_| host_allocation_failed("exact i64 component plane owners", outer_requested))?;
    check_component_planes(request.session, request.retained_base_bytes, &components)?;
    for _ in 0..component_count {
        let mut component = Vec::new();
        component
            .try_reserve_exact(request.num_pixels)
            .map_err(|_| {
                host_allocation_failed("exact i64 component samples", one_plane_requested)
            })?;
        component.resize(request.num_pixels, 0);
        components.push(component);
        check_component_planes(request.session, request.retained_base_bytes, &components)?;
    }
    fill_components(
        request.pixels,
        request.num_pixels,
        bytes_per_sample,
        request.bit_depth,
        request.signed,
        &mut components,
    )?;
    check_component_planes(request.session, request.retained_base_bytes, &components)?;
    Ok(components)
}

pub(super) fn component_planes_retained_bytes(
    components: &Vec<Vec<i64>>,
) -> NativeEncodePipelineResult<usize> {
    let mut bytes = checked_element_bytes::<Vec<i64>>(
        components.capacity(),
        "exact i64 component plane owners",
    )?;
    for component in components {
        bytes = checked_add_bytes(
            bytes,
            checked_element_bytes::<i64>(component.capacity(), "exact i64 component samples")?,
            "exact i64 component planes",
        )?;
    }
    Ok(bytes)
}

fn fill_components(
    pixels: &[u8],
    num_pixels: usize,
    bytes_per_sample: usize,
    bit_depth: u8,
    signed: bool,
    components: &mut [Vec<i64>],
) -> NativeEncodePipelineResult<()> {
    let pixel_stride = components.len().checked_mul(bytes_per_sample).ok_or(
        crate::EncodeError::InternalInvariant {
            what: "exact i64 fill pixel stride overflow",
        },
    )?;
    let unsigned_offset = if signed {
        0
    } else {
        1_i64 << (u32::from(bit_depth) - 1)
    };
    for (pixel_index, pixel) in pixels
        .chunks_exact(pixel_stride)
        .take(num_pixels)
        .enumerate()
    {
        for (component_index, component) in components.iter_mut().enumerate() {
            let offset = component_index.checked_mul(bytes_per_sample).ok_or(
                crate::EncodeError::InternalInvariant {
                    what: "exact i64 component sample offset overflow",
                },
            )?;
            let end = offset.checked_add(bytes_per_sample).ok_or(
                crate::EncodeError::InternalInvariant {
                    what: "exact i64 component sample extent overflow",
                },
            )?;
            let sample = pixel
                .get(offset..end)
                .ok_or(crate::EncodeError::InternalInvariant {
                    what: "exact i64 component sample range is out of bounds",
                })?;
            let raw = read_le_sample_value(sample, bit_depth);
            component[pixel_index] = if signed {
                sign_extend_sample(raw, bit_depth)
            } else {
                i64::try_from(raw).map_err(|_| crate::EncodeError::InternalInvariant {
                    what: "raw component sample exceeds i64",
                })? - unsigned_offset
            };
        }
    }
    Ok(())
}

fn check_component_planes(
    session: &NativeEncodeSession<'_>,
    retained_base_bytes: usize,
    components: &Vec<Vec<i64>>,
) -> NativeEncodePipelineResult<()> {
    session.checked_phase(
        checked_add_bytes(
            retained_base_bytes,
            component_planes_retained_bytes(components)?,
            "exact i64 component planes",
        )?,
        "exact i64 component planes",
    )?;
    Ok(())
}

#[cfg(test)]
#[path = "input/tests.rs"]
mod tests;
