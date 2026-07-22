// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use crate::color::{
    native_component_plane_dimensions, DecodedNativeComponents, NativeComponentPlane, RawBitmap,
};
use crate::error::{DecodingError, Result, ValidationError};
use crate::j2c::{ComponentData, DecoderContext};
use crate::{
    checked_decode_byte_len2, checked_decode_byte_len3, native_bytes_per_sample,
    try_reserve_decode_elements,
};

use super::Image;

mod allocation;
pub(super) use allocation::{try_clone_color_space, NativeOutputBudget};

impl<'a> Image<'a> {
    pub(super) fn pack_native_component_planes(
        &self,
        components: &[ComponentData],
        component_owner_capacity: usize,
        dimensions: (u32, u32),
        retained_baseline_bytes: usize,
    ) -> Result<DecodedNativeComponents> {
        let mut budget = NativeOutputBudget::for_decoded_channels(
            retained_baseline_bytes,
            components,
            component_owner_capacity,
        )?;
        budget.include_elements::<NativeComponentPlane>(components.len())?;
        for (component_idx, component) in components.iter().enumerate() {
            let sampling = self.component_plane_sampling_at(component_idx);
            let bytes_per_sample = native_bytes_per_sample(component.bit_depth)?;
            let sample_count = component
                .integer_container
                .as_ref()
                .map_or(component.container.truncated().len(), Vec::len);
            let capacity = checked_decode_byte_len2(sample_count, bytes_per_sample)?;
            native_component_plane_dimensions(dimensions, sampling, sample_count)?;
            budget.include_elements::<u8>(capacity)?;
        }
        budget.include_color_space_clone(&self.color_space)?;

        let color_space = try_clone_color_space(&self.color_space)?;
        budget.include_color_space_clone_overage(&self.color_space, &color_space)?;
        let mut planes = Vec::new();
        try_reserve_decode_elements(&mut planes, components.len())?;
        budget.include_capacity_overage::<NativeComponentPlane>(
            components.len(),
            planes.capacity(),
        )?;
        for (component_idx, component) in components.iter().enumerate() {
            let sampling = self.component_plane_sampling_at(component_idx);
            let bytes_per_sample = native_bytes_per_sample(component.bit_depth)?;
            let sample_count = component
                .integer_container
                .as_ref()
                .map_or(component.container.truncated().len(), Vec::len);
            let plane_dimensions =
                native_component_plane_dimensions(dimensions, sampling, sample_count)?;
            let capacity = checked_decode_byte_len2(sample_count, bytes_per_sample)?;
            let mut data = Vec::new();
            try_reserve_decode_elements(&mut data, capacity)?;
            budget.include_capacity_overage::<u8>(capacity, data.capacity())?;
            for idx in 0..sample_count {
                Self::push_component_native_sample_bytes(
                    &mut data,
                    component,
                    idx,
                    component.bit_depth,
                );
            }
            if data.len() != capacity {
                return Err(DecodingError::CodeBlockDecodeFailure.into());
            }
            planes.push(NativeComponentPlane {
                data,
                dimensions: plane_dimensions,
                bit_depth: component.bit_depth,
                signed: component.signed,
                sampling,
                bytes_per_sample: u8::try_from(bytes_per_sample)
                    .map_err(|_| ValidationError::ImageTooLarge)?,
            });
        }

        let packed = DecodedNativeComponents {
            dimensions,
            color_space,
            has_alpha: self.has_alpha,
            planes,
        };
        NativeOutputBudget::validate_component_pack(
            retained_baseline_bytes,
            components,
            component_owner_capacity,
            &packed,
        )?;
        Ok(packed)
    }

    pub(super) fn requires_exact_integer_decode(&self) -> bool {
        for component in &self.header.component_infos {
            if component.requires_exact_integer_decode() {
                return true;
            }
        }
        false
    }

    pub(super) fn decode_native_region_via_full_decode(
        &self,
        roi: (u32, u32, u32, u32),
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<RawBitmap> {
        let full = self.decode_native_with_context(decoder_context)?;
        let (x, y, width, height) = roi;
        let bytes_per_pixel = usize::from(full.num_components)
            .checked_mul(usize::from(full.bytes_per_sample))
            .ok_or(ValidationError::ImageTooLarge)?;
        let row_bytes = (width as usize)
            .checked_mul(bytes_per_pixel)
            .ok_or(ValidationError::ImageTooLarge)?;
        let capacity = checked_decode_byte_len3(height as usize, width as usize, bytes_per_pixel)?;
        let retained_image_bytes = self.retained_metadata_bytes()?;
        let components = &decoder_context.tile_decode_context.channel_data;
        let component_owner_capacity = components.capacity();
        let mut budget = NativeOutputBudget::for_raw_bitmap_with_decoded_channels(
            retained_image_bytes,
            components,
            component_owner_capacity,
            &full,
        )?;
        budget.include_elements::<u8>(capacity)?;
        let mut data = Vec::new();
        try_reserve_decode_elements(&mut data, capacity)?;
        budget.include_capacity_overage::<u8>(capacity, data.capacity())?;
        let full_width = full.width as usize;
        let row_end = y
            .checked_add(height)
            .ok_or(ValidationError::ImageTooLarge)?;
        for row in y as usize..row_end as usize {
            let start = row
                .checked_mul(full_width)
                .and_then(|offset| offset.checked_add(x as usize))
                .and_then(|sample| sample.checked_mul(bytes_per_pixel))
                .ok_or(ValidationError::ImageTooLarge)?;
            data.extend_from_slice(&full.data[start..start + row_bytes]);
        }
        if data.len() != capacity {
            return Err(DecodingError::CodeBlockDecodeFailure.into());
        }
        NativeOutputBudget::validate_raw_crop(
            retained_image_bytes,
            components,
            component_owner_capacity,
            &full,
            data.capacity(),
        )?;

        Ok(RawBitmap {
            data,
            width,
            height,
            bit_depth: full.bit_depth,
            signed: full.signed,
            component_signed: full.component_signed,
            num_components: full.num_components,
            bytes_per_sample: full.bytes_per_sample,
        })
    }

    pub(super) fn decode_native_region_components_via_full_decode(
        &self,
        roi: (u32, u32, u32, u32),
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<DecodedNativeComponents> {
        let full = self.decode_native_components_with_context(decoder_context)?;
        let (_, _, width, height) = roi;
        let retained_image_bytes = self.retained_metadata_bytes()?;
        let decoded_channels = &decoder_context.tile_decode_context.channel_data;
        let component_owner_capacity = decoded_channels.capacity();
        let mut budget = NativeOutputBudget::for_native_components_with_decoded_channels(
            retained_image_bytes,
            decoded_channels,
            component_owner_capacity,
            &full,
        )?;
        budget.include_elements::<NativeComponentPlane>(full.planes.len())?;
        for plane in &full.planes {
            let crop = native_plane_crop(plane, full.dimensions, roi)?;
            budget.include_elements::<u8>(crop.capacity)?;
        }

        let mut planes = Vec::new();
        try_reserve_decode_elements(&mut planes, full.planes.len())?;
        budget.include_capacity_overage::<NativeComponentPlane>(
            full.planes.len(),
            planes.capacity(),
        )?;
        for plane in &full.planes {
            let crop = native_plane_crop(plane, full.dimensions, roi)?;
            let mut data = Vec::new();
            try_reserve_decode_elements(&mut data, crop.capacity)?;
            budget.include_capacity_overage::<u8>(crop.capacity, data.capacity())?;
            let full_width = plane.dimensions.0 as usize;
            let crop_row_end = crop
                .y
                .checked_add(crop.height)
                .ok_or(ValidationError::ImageTooLarge)?;
            for row in crop.y as usize..crop_row_end as usize {
                let start = row
                    .checked_mul(full_width)
                    .and_then(|offset| offset.checked_add(crop.x as usize))
                    .and_then(|sample| sample.checked_mul(usize::from(plane.bytes_per_sample)))
                    .ok_or(ValidationError::ImageTooLarge)?;
                let end = start
                    .checked_add(crop.row_bytes)
                    .ok_or(ValidationError::ImageTooLarge)?;
                let row = plane
                    .data
                    .get(start..end)
                    .ok_or(DecodingError::CodeBlockDecodeFailure)?;
                data.extend_from_slice(row);
            }
            if data.len() != crop.capacity {
                return Err(DecodingError::CodeBlockDecodeFailure.into());
            }
            planes.push(NativeComponentPlane {
                data,
                dimensions: (crop.width, crop.height),
                bit_depth: plane.bit_depth,
                signed: plane.signed,
                sampling: plane.sampling,
                bytes_per_sample: plane.bytes_per_sample,
            });
        }

        NativeOutputBudget::validate_component_crop(
            retained_image_bytes,
            decoded_channels,
            component_owner_capacity,
            &full,
            &planes,
            planes.capacity(),
        )?;

        Ok(DecodedNativeComponents {
            dimensions: (width, height),
            color_space: full.color_space,
            has_alpha: full.has_alpha,
            planes,
        })
    }

    pub(super) fn push_component_native_sample_bytes(
        out: &mut Vec<u8>,
        component: &ComponentData,
        index: usize,
        bit_depth: u8,
    ) {
        if let Some(samples) = component.integer_container.as_ref() {
            Self::push_native_i64_sample_bytes(out, samples[index], bit_depth, component.signed);
        } else {
            Self::push_native_sample_bytes(
                out,
                component.container.truncated()[index],
                bit_depth,
                component.signed,
            );
        }
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "samples are clamped to the declared signed or unsigned component range before packing"
    )]
    fn push_native_i64_sample_bytes(out: &mut Vec<u8>, sample: i64, bit_depth: u8, signed: bool) {
        if signed {
            let magnitude_bits = u32::from(bit_depth.saturating_sub(1));
            let min = -(1_i64 << magnitude_bits);
            let max = (1_i64 << magnitude_bits) - 1;
            let clamped = sample.clamp(min, max);
            if bit_depth <= 8 {
                out.push((clamped as i8) as u8);
            } else if bit_depth <= 16 {
                out.extend_from_slice(&(clamped as i16).to_le_bytes());
            } else {
                let bytes = clamped.to_le_bytes();
                let byte_count = native_bytes_per_sample(bit_depth).unwrap_or(8);
                out.extend_from_slice(&bytes[..byte_count]);
            }
        } else {
            let max = (1u64 << u32::from(bit_depth)) - 1;
            let clamped = if sample <= 0 {
                0
            } else {
                (sample as u64).min(max)
            };
            if bit_depth <= 8 {
                out.push(clamped as u8);
            } else if bit_depth <= 16 {
                out.extend_from_slice(&(clamped as u16).to_le_bytes());
            } else {
                let bytes = clamped.to_le_bytes();
                let byte_count = native_bytes_per_sample(bit_depth).unwrap_or(8);
                out.extend_from_slice(&bytes[..byte_count]);
            }
        }
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "rounded samples are range-checked before stable native-width packing"
    )]
    pub(crate) fn push_native_sample_bytes(
        out: &mut Vec<u8>,
        sample: f32,
        bit_depth: u8,
        signed: bool,
    ) {
        let sample = f64::from(sample);
        if signed {
            let magnitude_bits = u32::from(bit_depth.saturating_sub(1));
            let min = -(1_i64 << magnitude_bits);
            let max = (1_i64 << magnitude_bits) - 1;
            let clamped = if sample.is_nan() {
                0
            } else if sample <= min as f64 {
                min
            } else if sample >= max as f64 {
                max
            } else if sample >= 0.0 {
                (sample + 0.5) as i64
            } else {
                (sample - 0.5) as i64
            };
            if bit_depth <= 8 {
                out.push((clamped as i8) as u8);
            } else if bit_depth <= 16 {
                out.extend_from_slice(&(clamped as i16).to_le_bytes());
            } else {
                let bytes = clamped.to_le_bytes();
                let byte_count = native_bytes_per_sample(bit_depth).unwrap_or(8);
                out.extend_from_slice(&bytes[..byte_count]);
            }
        } else {
            let max = (1u64 << u32::from(bit_depth)) - 1;
            let clamped = if sample.is_nan() || sample <= 0.0 {
                0
            } else if sample >= max as f64 {
                max
            } else {
                (sample + 0.5) as u64
            };
            if bit_depth <= 8 {
                out.push(clamped as u8);
            } else if bit_depth <= 16 {
                out.extend_from_slice(&(clamped as u16).to_le_bytes());
            } else {
                let bytes = clamped.to_le_bytes();
                let byte_count = native_bytes_per_sample(bit_depth).unwrap_or(8);
                out.extend_from_slice(&bytes[..byte_count]);
            }
        }
    }
}

#[derive(Clone, Copy)]
struct NativePlaneCrop {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    row_bytes: usize,
    capacity: usize,
}

fn native_plane_crop(
    plane: &NativeComponentPlane,
    full_dimensions: (u32, u32),
    roi: (u32, u32, u32, u32),
) -> Result<NativePlaneCrop> {
    let (x, y, width, height) = roi;
    let (crop_x, crop_y, crop_width, crop_height) = if plane.dimensions == full_dimensions {
        (x, y, width, height)
    } else {
        let x1 = x.checked_add(width).ok_or(ValidationError::ImageTooLarge)?;
        let y1 = y
            .checked_add(height)
            .ok_or(ValidationError::ImageTooLarge)?;
        let (x_rsiz, y_rsiz) = plane.sampling;
        if x_rsiz == 0 || y_rsiz == 0 {
            return Err(DecodingError::CodeBlockDecodeFailure.into());
        }
        let crop_x = x / u32::from(x_rsiz);
        let crop_y = y / u32::from(y_rsiz);
        let crop_end_x = x1.div_ceil(u32::from(x_rsiz)).min(plane.dimensions.0);
        let crop_end_y = y1.div_ceil(u32::from(y_rsiz)).min(plane.dimensions.1);
        (
            crop_x,
            crop_y,
            crop_end_x.saturating_sub(crop_x),
            crop_end_y.saturating_sub(crop_y),
        )
    };
    let bytes_per_sample = usize::from(plane.bytes_per_sample);
    let row_bytes = (crop_width as usize)
        .checked_mul(bytes_per_sample)
        .ok_or(ValidationError::ImageTooLarge)?;
    let capacity =
        checked_decode_byte_len3(crop_height as usize, crop_width as usize, bytes_per_sample)?;
    Ok(NativePlaneCrop {
        x: crop_x,
        y: crop_y,
        width: crop_width,
        height: crop_height,
        row_bytes,
        capacity,
    })
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::native_plane_crop;
    use crate::error::{DecodeError, DecodingError};
    use crate::NativeComponentPlane;

    fn plane(dimensions: (u32, u32), sampling: (u8, u8)) -> NativeComponentPlane {
        NativeComponentPlane {
            data: Vec::new(),
            dimensions,
            bit_depth: 16,
            signed: false,
            sampling,
            bytes_per_sample: 2,
        }
    }

    #[test]
    fn subsampled_native_crop_rounds_outward_to_cover_the_requested_region() {
        let crop = native_plane_crop(&plane((4, 4), (2, 2)), (8, 8), (1, 1, 4, 4))
            .expect("valid subsampled crop");

        assert_eq!((crop.x, crop.y), (0, 0));
        assert_eq!((crop.width, crop.height), (3, 3));
        assert_eq!(crop.row_bytes, 6);
        assert_eq!(crop.capacity, 18);
    }

    #[test]
    fn subsampled_native_crop_rejects_zero_sampling_without_panicking() {
        assert!(matches!(
            native_plane_crop(&plane((4, 4), (0, 2)), (8, 8), (0, 0, 4, 4)),
            Err(DecodeError::Decoding(DecodingError::CodeBlockDecodeFailure))
        ));
    }
}
