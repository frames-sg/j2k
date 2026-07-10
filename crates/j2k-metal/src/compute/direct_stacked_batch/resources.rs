// SPDX-License-Identifier: MIT OR Apache-2.0

use std::mem::size_of;

use metal::Buffer;

use super::super::{
    BandRequiredRegion, DirectScratchBuffer, DirectStatusCheck, Error, J2kDirectBandId,
};

#[derive(Clone)]
pub(in super::super) struct DirectBandSlice {
    pub(in super::super) band_id: J2kDirectBandId,
    pub(in super::super) buffer: Buffer,
    pub(in super::super) offset_bytes: usize,
    pub(in super::super) window: BandRequiredRegion,
}

pub(super) struct StackedComponentResources {
    pub(super) band_sets: Vec<Vec<DirectBandSlice>>,
    pub(super) final_plane: Option<Buffer>,
}

pub(super) fn prepare_stacked_component_resources(count: usize) -> StackedComponentResources {
    StackedComponentResources {
        band_sets: vec![Vec::<DirectBandSlice>::new(); count],
        final_plane: None,
    }
}

pub(super) fn retain_metal_tier1_output(
    output: DirectScratchBuffer,
    buffers: Vec<Buffer>,
    status_check: DirectStatusCheck,
    retained_buffers: &mut Vec<Buffer>,
    status_checks: &mut Vec<DirectStatusCheck>,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Buffer {
    retained_buffers.extend(buffers);
    status_checks.push(status_check);
    let buffer = output.buffer.clone();
    scratch_buffers.push(output);
    buffer
}

pub(in super::super) fn lookup_direct_band_slice_entry(
    bands: &[DirectBandSlice],
    band_id: J2kDirectBandId,
    rect: j2k_native::J2kRect,
) -> Result<DirectBandSlice, Error> {
    bands
        .iter()
        .find(|existing| existing.band_id == band_id)
        .cloned()
        .ok_or_else(|| Error::MetalKernel {
            message: format!(
                "missing J2K MetalDirect device band {} for rect ({}, {}, {}, {})",
                band_id, rect.x0, rect.y0, rect.x1, rect.y1
            ),
        })
}

pub(in super::super) fn lookup_direct_band_slice(
    bands: &[DirectBandSlice],
    band_id: J2kDirectBandId,
    rect: j2k_native::J2kRect,
) -> Result<(Buffer, usize), Error> {
    let entry = lookup_direct_band_slice_entry(bands, band_id, rect)?;
    Ok((entry.buffer, entry.offset_bytes))
}

pub(in super::super) fn lookup_repeated_direct_band_layout_entry(
    band_sets: &[Vec<DirectBandSlice>],
    band_id: J2kDirectBandId,
    rect: j2k_native::J2kRect,
) -> Result<(DirectBandSlice, u32), Error> {
    let first_bands = band_sets.first().ok_or_else(|| Error::MetalKernel {
        message: "missing J2K MetalDirect repeated band set".to_string(),
    })?;
    let entry = lookup_direct_band_slice_entry(first_bands, band_id, rect)?;
    let stride_bytes = if let Some(second_bands) = band_sets.get(1) {
        let next = lookup_direct_band_slice_entry(second_bands, band_id, rect)?;
        next.offset_bytes
            .checked_sub(entry.offset_bytes)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K MetalDirect repeated band offsets are not monotonic".to_string(),
            })?
    } else {
        entry.window.width() as usize * entry.window.height() as usize * size_of::<f32>()
    };
    if stride_bytes % size_of::<f32>() != 0 {
        return Err(Error::MetalKernel {
            message: "J2K MetalDirect repeated band stride is not f32-aligned".to_string(),
        });
    }
    let stride_elements =
        u32::try_from(stride_bytes / size_of::<f32>()).map_err(|_| Error::MetalKernel {
            message: "J2K MetalDirect repeated band stride exceeds u32".to_string(),
        })?;
    Ok((entry, stride_elements))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_repeated_band_layout_preserves_validation_error() {
        let Err(error) = lookup_repeated_direct_band_layout_entry(
            &[],
            0,
            j2k_native::J2kRect {
                x0: 0,
                y0: 0,
                x1: 1,
                y1: 1,
            },
        ) else {
            panic!("empty repeated band layout must fail validation");
        };

        assert!(matches!(
            error,
            Error::MetalKernel { message }
                if message == "missing J2K MetalDirect repeated band set"
        ));
    }
}
