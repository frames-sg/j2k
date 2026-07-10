// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    checked_decode_byte_len2, checked_decode_byte_len3, math, native_bytes_per_sample,
    native_component_plane_dimensions, ComponentData, DecodedNativeComponents, DecoderContext,
    Image, NativeComponentPlane, RawBitmap, Result, ValidationError, Vec,
};

impl<'a> Image<'a> {
    pub(super) fn pack_native_component_planes(
        &self,
        components: &[ComponentData],
        dimensions: (u32, u32),
    ) -> Result<DecodedNativeComponents> {
        let sampling = self.component_plane_sampling(components.len());
        let mut planes = Vec::with_capacity(components.len());
        for (component, sampling) in components.iter().zip(sampling) {
            let bytes_per_sample = native_bytes_per_sample(component.bit_depth)?;
            let sample_count = component
                .integer_container
                .as_ref()
                .map_or(component.container.truncated().len(), Vec::len);
            let plane_dimensions =
                native_component_plane_dimensions(dimensions, sampling, sample_count)?;
            let capacity = checked_decode_byte_len2(sample_count, bytes_per_sample)?;
            let mut data = Vec::with_capacity(capacity);
            for idx in 0..sample_count {
                Self::push_component_native_sample_bytes(
                    &mut data,
                    component,
                    idx,
                    component.bit_depth,
                );
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

        Ok(DecodedNativeComponents {
            dimensions,
            color_space: self.color_space.clone(),
            has_alpha: self.has_alpha,
            planes,
        })
    }

    pub(super) fn requires_exact_integer_decode(&self) -> bool {
        self.header
            .component_infos
            .iter()
            .any(|component| component.requires_exact_integer_decode())
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
        let mut data = Vec::with_capacity(capacity);
        let full_width = full.width as usize;
        for row in y as usize..(y + height) as usize {
            let start = row
                .checked_mul(full_width)
                .and_then(|offset| offset.checked_add(x as usize))
                .and_then(|sample| sample.checked_mul(bytes_per_pixel))
                .ok_or(ValidationError::ImageTooLarge)?;
            data.extend_from_slice(&full.data[start..start + row_bytes]);
        }

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
        let (x, y, width, height) = roi;
        let mut planes = Vec::with_capacity(full.planes.len());
        for plane in &full.planes {
            let bytes_per_sample = usize::from(plane.bytes_per_sample);
            let (crop_x, crop_y, crop_width, crop_height) = if plane.dimensions == full.dimensions {
                (x, y, width, height)
            } else {
                let x1 = x.checked_add(width).ok_or(ValidationError::ImageTooLarge)?;
                let y1 = y
                    .checked_add(height)
                    .ok_or(ValidationError::ImageTooLarge)?;
                let (x_rsiz, y_rsiz) = plane.sampling;
                let crop_x = x / u32::from(x_rsiz);
                let crop_y = y / u32::from(y_rsiz);
                let crop_x1 = x1.div_ceil(u32::from(x_rsiz)).min(plane.dimensions.0);
                let crop_y1 = y1.div_ceil(u32::from(y_rsiz)).min(plane.dimensions.1);
                (
                    crop_x,
                    crop_y,
                    crop_x1.saturating_sub(crop_x),
                    crop_y1.saturating_sub(crop_y),
                )
            };
            let row_bytes = (crop_width as usize)
                .checked_mul(bytes_per_sample)
                .ok_or(ValidationError::ImageTooLarge)?;
            let capacity = checked_decode_byte_len3(
                crop_height as usize,
                crop_width as usize,
                bytes_per_sample,
            )?;
            let mut data = Vec::with_capacity(capacity);
            let full_width = plane.dimensions.0 as usize;
            for row in crop_y as usize..(crop_y + crop_height) as usize {
                let start = row
                    .checked_mul(full_width)
                    .and_then(|offset| offset.checked_add(crop_x as usize))
                    .and_then(|sample| sample.checked_mul(bytes_per_sample))
                    .ok_or(ValidationError::ImageTooLarge)?;
                data.extend_from_slice(&plane.data[start..start + row_bytes]);
            }
            planes.push(NativeComponentPlane {
                data,
                dimensions: (crop_width, crop_height),
                bit_depth: plane.bit_depth,
                signed: plane.signed,
                sampling: plane.sampling,
                bytes_per_sample: plane.bytes_per_sample,
            });
        }

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

    pub(crate) fn push_native_sample_bytes(
        out: &mut Vec<u8>,
        sample: f32,
        bit_depth: u8,
        signed: bool,
    ) {
        let rounded = math::round_f32(sample);
        if signed {
            let magnitude_bits = u32::from(bit_depth.saturating_sub(1));
            let min = -(1_i64 << magnitude_bits);
            let max = (1_i64 << magnitude_bits) - 1;
            let rounded = f64::from(rounded);
            let clamped = if rounded.is_nan() {
                0
            } else if rounded <= min as f64 {
                min
            } else if rounded >= max as f64 {
                max
            } else {
                rounded as i64
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
            let rounded = f64::from(rounded);
            let clamped = if rounded.is_nan() || rounded <= 0.0 {
                0
            } else if rounded >= max as f64 {
                max
            } else {
                rounded as u64
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
