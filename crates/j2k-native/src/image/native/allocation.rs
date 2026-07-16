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

    pub(super) fn for_raw_bitmap_with_decoded_channels(
        retained_image_bytes: usize,
        components: &[ComponentData],
        component_owner_capacity: usize,
        bitmap: &RawBitmap,
    ) -> Result<Self> {
        let mut budget =
            Self::for_decoded_channels(retained_image_bytes, components, component_owner_capacity)?;
        budget.include_raw_bitmap(bitmap)?;
        Ok(budget)
    }

    pub(super) fn for_native_components_with_decoded_channels(
        retained_image_bytes: usize,
        decoded_channels: &[ComponentData],
        component_owner_capacity: usize,
        components: &DecodedNativeComponents,
    ) -> Result<Self> {
        let mut budget = Self::for_decoded_channels(
            retained_image_bytes,
            decoded_channels,
            component_owner_capacity,
        )?;
        budget.include_native_components(components)?;
        Ok(budget)
    }

    pub(super) fn validate_component_pack(
        retained_image_bytes: usize,
        components: &[ComponentData],
        component_owner_capacity: usize,
        packed: &DecodedNativeComponents,
    ) -> Result<()> {
        let mut budget =
            Self::for_decoded_channels(retained_image_bytes, components, component_owner_capacity)?;
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

    pub(super) fn validate_raw_crop(
        retained_image_bytes: usize,
        components: &[ComponentData],
        component_owner_capacity: usize,
        full: &RawBitmap,
        cropped_capacity: usize,
    ) -> Result<()> {
        let mut budget = Self::for_raw_bitmap_with_decoded_channels(
            retained_image_bytes,
            components,
            component_owner_capacity,
            full,
        )?;
        budget.include_elements::<u8>(cropped_capacity)
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

    pub(super) fn validate_component_crop(
        retained_image_bytes: usize,
        decoded_channels: &[ComponentData],
        component_owner_capacity: usize,
        full: &DecodedNativeComponents,
        cropped_planes: &[NativeComponentPlane],
        cropped_plane_owner_capacity: usize,
    ) -> Result<()> {
        let mut budget = Self::for_native_components_with_decoded_channels(
            retained_image_bytes,
            decoded_channels,
            component_owner_capacity,
            full,
        )?;
        budget.include_elements::<NativeComponentPlane>(cropped_plane_owner_capacity)?;
        for plane in cropped_planes {
            budget.include_elements::<u8>(plane.data.capacity())?;
        }
        Ok(())
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

    fn include_raw_bitmap(&mut self, bitmap: &RawBitmap) -> Result<()> {
        self.include_elements::<u8>(bitmap.data.capacity())?;
        self.include_bit_capacity(bitmap.component_signed.capacity())
    }

    fn include_native_components(&mut self, components: &DecodedNativeComponents) -> Result<()> {
        self.include_elements::<NativeComponentPlane>(components.planes.capacity())?;
        for plane in &components.planes {
            self.include_elements::<u8>(plane.data.capacity())?;
        }
        self.include_color_space(&components.color_space)
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
