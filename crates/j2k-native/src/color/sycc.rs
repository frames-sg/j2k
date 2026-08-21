// SPDX-License-Identifier: MIT OR Apache-2.0

//! JPEG 2000 sYCC-to-RGB conversion.

use crate::error::bail;
use crate::j2c::ComponentData;
use crate::math::{f32x8, Simd, SIMD_WIDTH};
use crate::{ColorError, Result};

#[expect(
    clippy::cast_precision_loss,
    reason = "JPEG 2000 sYCC conversion intentionally uses f32 SIMD arithmetic"
)]
#[inline]
pub(crate) fn sycc_to_rgb<S: Simd>(
    simd: S,
    components: &mut [ComponentData],
    bit_depth: u8,
) -> Result<()> {
    let offset = (1_u64 << (u32::from(bit_depth) - 1)) as f32;
    let max_value = ((1_u64 << u32::from(bit_depth)) - 1) as f32;
    let (head, _) = components
        .split_at_mut_checked(3)
        .ok_or(ColorError::SyccConversionFailed)?;
    let [luma, blue_chroma, red_chroma] = head else {
        bail!(ColorError::SyccConversionFailed);
    };

    let offset_v = f32x8::splat(simd, offset);
    let max_v = f32x8::splat(simd, max_value);
    let zero_v = f32x8::splat(simd, 0.0);
    let red_chroma_to_red = f32x8::splat(simd, 1.402);
    let blue_chroma_to_green = f32x8::splat(simd, -0.344_136);
    let red_chroma_to_green = f32x8::splat(simd, -0.714_136);
    let blue_chroma_to_blue = f32x8::splat(simd, 1.772);

    for ((luma_chunk, blue_chroma_chunk), red_chroma_chunk) in luma
        .container
        .chunks_exact_mut(SIMD_WIDTH)
        .zip(blue_chroma.container.chunks_exact_mut(SIMD_WIDTH))
        .zip(red_chroma.container.chunks_exact_mut(SIMD_WIDTH))
    {
        let luma_values = f32x8::from_slice(simd, luma_chunk);
        let blue_chroma_values = f32x8::from_slice(simd, blue_chroma_chunk) - offset_v;
        let red_chroma_values = f32x8::from_slice(simd, red_chroma_chunk) - offset_v;
        let red = red_chroma_values.mul_add(red_chroma_to_red, luma_values);
        let green = red_chroma_values.mul_add(
            red_chroma_to_green,
            blue_chroma_values.mul_add(blue_chroma_to_green, luma_values),
        );
        let blue = blue_chroma_values.mul_add(blue_chroma_to_blue, luma_values);
        red.min(max_v).max(zero_v).store(luma_chunk);
        green.min(max_v).max(zero_v).store(blue_chroma_chunk);
        blue.min(max_v).max(zero_v).store(red_chroma_chunk);
    }

    luma.integer_container = None;
    blue_chroma.integer_container = None;
    red_chroma.integer_container = None;
    Ok(())
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::math::{dispatch, Level, SimdBuffer};

    #[test]
    fn sycc_conversion_discards_pretransform_integer_shadows() {
        let component = |value: u8| ComponentData {
            container: SimdBuffer::<SIMD_WIDTH>::new(vec![f32::from(value); SIMD_WIDTH]),
            integer_container: Some(vec![i64::from(value); SIMD_WIDTH]),
            bit_depth: 8,
            signed: false,
        };
        let mut components = vec![component(128), component(128), component(128)];
        dispatch!(Level::new(), simd => sycc_to_rgb(simd, &mut components, 8))
            .expect("sYCC conversion");
        assert!(
            components
                .iter()
                .all(|component| component.integer_container.is_none()),
            "native packing must not reuse pre-transform exact samples"
        );
    }
}
