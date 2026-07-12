// SPDX-License-Identifier: MIT OR Apache-2.0

//! Component sample validation and conversion into the transform float domain.

use super::super::{
    allocation::{checked_add_bytes, checked_element_bytes, host_allocation_failed},
    raw_pixel_bytes_per_sample, read_le_sample_value, sign_extend_sample,
    EncodeComponentSampleInfo, NativeEncodePipelineResult, NativeEncodeSession, Vec,
    MAX_RAW_PIXEL_ENCODE_BIT_DEPTH,
};

pub(in crate::j2c::encode) fn validate_deinterleaved_components(
    components: Vec<Vec<f32>>,
    num_components: u16,
    num_pixels: usize,
) -> Result<Vec<Vec<f32>>, &'static str> {
    if components.len() != usize::from(num_components) {
        return Err("accelerated deinterleave component count mismatch");
    }
    if components
        .iter()
        .any(|component| component.len() != num_pixels)
    {
        return Err("accelerated deinterleave component length mismatch");
    }
    Ok(components)
}

#[expect(
    clippy::cast_precision_loss,
    reason = "the codec float domain intentionally receives bounded integer samples or metadata at this rounding boundary"
)]
pub(in crate::j2c::encode) fn try_component_plane_to_f32_for_session(
    data: &[u8],
    width: u32,
    height: u32,
    bit_depth: u8,
    signed: bool,
    session: &NativeEncodeSession<'_>,
    retained_base_bytes: usize,
) -> NativeEncodePipelineResult<Vec<f32>> {
    let sample_count = usize::try_from(width)
        .map_err(|_| crate::EncodeError::ArithmeticOverflow {
            what: "typed component width",
        })?
        .checked_mul(usize::try_from(height).map_err(|_| {
            crate::EncodeError::ArithmeticOverflow {
                what: "typed component height",
            }
        })?)
        .ok_or(crate::EncodeError::ArithmeticOverflow {
            what: "typed component sample count",
        })?;
    let bytes_per_sample = raw_pixel_bytes_per_sample(bit_depth)
        .map_err(|what| crate::EncodeError::InvalidInput { what })?;
    let expected_len = sample_count.checked_mul(bytes_per_sample).ok_or(
        crate::EncodeError::ArithmeticOverflow {
            what: "typed component byte length",
        },
    )?;
    if data.len() != expected_len {
        return Err(crate::EncodeError::InvalidInput {
            what: "component plane data length mismatch",
        }
        .into());
    }

    let requested_bytes =
        checked_element_bytes::<f32>(sample_count, "typed component floating-point samples")?;
    session.checked_phase(
        checked_add_bytes(
            retained_base_bytes,
            requested_bytes,
            "typed component sample conversion",
        )?,
        "typed component sample conversion",
    )?;
    let mut samples = Vec::new();
    samples.try_reserve_exact(sample_count).map_err(|_| {
        host_allocation_failed("typed component floating-point samples", requested_bytes)
    })?;
    let actual_bytes =
        checked_element_bytes::<f32>(samples.capacity(), "typed component floating-point samples")?;
    session.checked_phase(
        checked_add_bytes(
            retained_base_bytes,
            actual_bytes,
            "typed component sample conversion",
        )?,
        "typed component sample conversion",
    )?;

    let unsigned_offset = if signed {
        0.0
    } else {
        (1_u64 << (u32::from(bit_depth) - 1)) as f32
    };
    for sample in data.chunks_exact(bytes_per_sample) {
        let raw = read_le_sample_value(sample, bit_depth);
        samples.push(if signed {
            sign_extend_sample(raw, bit_depth) as f32
        } else {
            raw as f32 - unsigned_offset
        });
    }
    Ok(samples)
}

pub(in crate::j2c::encode) fn validate_component_sample_info(
    component_sample_info: &[EncodeComponentSampleInfo],
    num_components: usize,
) -> Result<(), &'static str> {
    if component_sample_info.is_empty() {
        return Ok(());
    }
    if component_sample_info.len() != num_components {
        return Err("component sample metadata count does not match component count");
    }
    if component_sample_info
        .iter()
        .any(|info| info.bit_depth == 0 || info.bit_depth > MAX_RAW_PIXEL_ENCODE_BIT_DEPTH)
    {
        return Err("unsupported bit depth");
    }
    Ok(())
}
