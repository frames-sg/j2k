// SPDX-License-Identifier: MIT OR Apache-2.0

//! Live-owner accounting and fallible allocation for native decode outputs.

use alloc::vec::Vec;

use crate::color::{
    Bitmap, ColorSpace, ComponentPlane, DecodedComponents, DecodedNativeComponents,
    NativeComponentPlane, RawBitmap,
};
use crate::error::Result;
use crate::image::allocation::DecodeOwnerBudget;
use crate::j2c::ComponentData;
use crate::try_reserve_decode_elements;

fn bit_capacity_bytes(capacity_bits: usize) -> usize {
    capacity_bits / 8 + usize::from(!capacity_bits.is_multiple_of(8))
}

#[derive(Default)]
pub(in crate::image) struct NativeOutputBudget {
    allocation: DecodeOwnerBudget,
}

impl NativeOutputBudget {
    pub(in crate::image) fn for_decoded_channels(
        retained_image_bytes: usize,
        components: &[ComponentData],
        component_owner_capacity: usize,
    ) -> Result<Self> {
        Ok(Self {
            allocation: DecodeOwnerBudget::for_components(
                retained_image_bytes,
                components,
                component_owner_capacity,
            )?,
        })
    }

    /// The allowance of a decode's returned native component planes. They
    /// are its output, like a caller's buffer, so the decode budget, which
    /// holds the working memory they are packed from, does not charge them;
    /// they are held to this allowance of their own before they are
    /// allocated, and each allocation to the per-allocation cap. The planes
    /// route and the rows route share it.
    pub(super) fn for_output() -> Self {
        Self::default()
    }

    pub(super) fn validate_component_pack(packed: &DecodedNativeComponents) -> Result<()> {
        let mut budget = Self::for_output();
        budget.include_elements::<NativeComponentPlane>(packed.planes.capacity())?;
        for plane in &packed.planes {
            budget.include_elements::<u8>(plane.data.capacity())?;
        }
        budget.include_color_space(&packed.color_space)
    }

    pub(in crate::image) fn validate_borrowed_pack(
        retained_image_bytes: usize,
        components: &[ComponentData],
        component_owner_capacity: usize,
        packed: &DecodedComponents<'_>,
    ) -> Result<usize> {
        let mut peak_budget =
            Self::for_decoded_channels(retained_image_bytes, components, component_owner_capacity)?;
        peak_budget.include_elements::<ComponentPlane<'_>>(packed.planes.capacity())?;
        peak_budget.include_color_space(&packed.color_space)?;

        // The source Image and its original color profile can be dropped after
        // this handoff. The decoded channel owners cannot: every returned plane
        // borrows its SIMD buffer from the decoder context. Record that exact
        // retained baseline alongside the output metadata's actual capacities.
        let mut live_budget = Self::for_decoded_channels(0, components, component_owner_capacity)?;
        live_budget.include_elements::<ComponentPlane<'_>>(packed.planes.capacity())?;
        live_budget.include_color_space(&packed.color_space)?;
        Ok(live_budget.allocation.bytes())
    }

    pub(in crate::image) fn validate_raw_pack(
        retained_image_bytes: usize,
        components: &[ComponentData],
        component_owner_capacity: usize,
        packed: &RawBitmap,
    ) -> Result<()> {
        let mut budget =
            Self::for_decoded_channels(retained_image_bytes, components, component_owner_capacity)?;
        budget.include_elements::<u8>(packed.data.capacity())?;
        budget.include_bit_capacity(packed.component_signed.capacity())
    }

    pub(in crate::image) fn validate_bitmap_pack(
        retained_image_bytes: usize,
        components: &[ComponentData],
        component_owner_capacity: usize,
        packed: &Bitmap,
    ) -> Result<()> {
        let mut budget =
            Self::for_decoded_channels(retained_image_bytes, components, component_owner_capacity)?;
        budget.include_elements::<u8>(packed.data.capacity())?;
        budget.include_color_space(&packed.color_space)
    }

    pub(in crate::image) fn include_elements<T>(&mut self, count: usize) -> Result<()> {
        self.allocation.include_elements::<T>(count)
    }

    pub(in crate::image) fn include_capacity_overage<T>(
        &mut self,
        planned_count: usize,
        actual_capacity: usize,
    ) -> Result<()> {
        self.allocation
            .include_capacity_overage::<T>(planned_count, actual_capacity)
    }

    pub(in crate::image) fn include_bit_capacity(&mut self, capacity_bits: usize) -> Result<()> {
        self.include_elements::<u8>(bit_capacity_bytes(capacity_bits))
    }

    pub(in crate::image) fn include_bit_capacity_overage(
        &mut self,
        planned_bits: usize,
        actual_capacity_bits: usize,
    ) -> Result<()> {
        let planned_bytes = bit_capacity_bytes(planned_bits);
        let actual_bytes = bit_capacity_bytes(actual_capacity_bits);
        if actual_bytes > planned_bytes {
            self.include_elements::<u8>(actual_bytes - planned_bytes)?;
        }
        Ok(())
    }

    #[cfg(test)]
    fn from_retained_image(retained_image_bytes: usize) -> Result<Self> {
        Ok(Self {
            allocation: DecodeOwnerBudget::from_retained_bytes(retained_image_bytes)?,
        })
    }

    pub(in crate::image) fn include_color_space_clone(
        &mut self,
        color_space: &ColorSpace,
    ) -> Result<()> {
        if let ColorSpace::Icc { profile, .. } = color_space {
            self.include_elements::<u8>(profile.len())?;
        }
        Ok(())
    }

    pub(in crate::image) fn include_color_space_clone_overage(
        &mut self,
        source: &ColorSpace,
        cloned: &ColorSpace,
    ) -> Result<()> {
        match (source, cloned) {
            (
                ColorSpace::Icc {
                    profile: source_profile,
                    ..
                },
                ColorSpace::Icc {
                    profile: cloned_profile,
                    ..
                },
            ) => {
                self.include_capacity_overage::<u8>(source_profile.len(), cloned_profile.capacity())
            }
            _ => Ok(()),
        }
    }

    fn include_color_space(&mut self, color_space: &ColorSpace) -> Result<()> {
        if let ColorSpace::Icc { profile, .. } = color_space {
            self.include_elements::<u8>(profile.capacity())?;
        }
        Ok(())
    }
}

pub(in crate::image) fn try_clone_color_space(color_space: &ColorSpace) -> Result<ColorSpace> {
    Ok(match color_space {
        ColorSpace::Gray => ColorSpace::Gray,
        ColorSpace::RGB => ColorSpace::RGB,
        ColorSpace::CMYK => ColorSpace::CMYK,
        ColorSpace::Unknown { num_channels } => ColorSpace::Unknown {
            num_channels: *num_channels,
        },
        ColorSpace::Icc {
            profile,
            num_channels,
        } => {
            let mut cloned_profile = Vec::new();
            try_reserve_decode_elements(&mut cloned_profile, profile.len())?;
            cloned_profile.extend_from_slice(profile);
            ColorSpace::Icc {
                profile: cloned_profile,
                num_channels: *num_channels,
            }
        }
    })
}

#[cfg(test)]
mod tests;
