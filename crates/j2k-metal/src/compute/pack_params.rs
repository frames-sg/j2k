// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::Error;

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(super) struct J2kScalarPackParams {
    pub(super) max_value: f32,
    pub(super) u8_scale: f32,
    pub(super) u16_scale: f32,
}

#[cfg(target_os = "macos")]
pub(super) fn j2k_scalar_pack_params(bit_depth: u32) -> J2kScalarPackParams {
    let clamped = bit_depth.min(16);
    let max_value_u16 = ((1u32 << clamped) - 1).max(1) as u16;
    let max_value = f32::from(max_value_u16);
    let u8_scale = 255.0 / max_value;
    let u16_scale = if bit_depth <= 8 {
        65_535.0 / max_value
    } else {
        1.0
    };
    J2kScalarPackParams {
        max_value,
        u8_scale,
        u16_scale,
    }
}

#[cfg(target_os = "macos")]
pub(super) fn j2k_u32_param(value: usize, message: &'static str) -> Result<u32, Error> {
    u32::try_from(value).map_err(|_| Error::MetalKernel {
        message: message.to_string(),
    })
}

#[cfg(target_os = "macos")]
pub(super) fn j2k_pack_scale_arrays(bit_depths: [u32; 4]) -> ([f32; 4], [f32; 4], [f32; 4]) {
    let mut max_values = [1.0f32; 4];
    let mut u8_scales = [255.0f32; 4];
    let mut u16_scales = [65_535.0f32; 4];
    for (index, bit_depth) in bit_depths.into_iter().enumerate() {
        let params = j2k_scalar_pack_params(bit_depth);
        max_values[index] = params.max_value;
        u8_scales[index] = params.u8_scale;
        u16_scales[index] = params.u16_scale;
    }
    (max_values, u8_scales, u16_scales)
}
