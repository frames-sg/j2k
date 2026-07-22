use super::{
    bail, execute_component_plan, floor_f32, prepare_direct_scratch, round_f32, DecodingError,
    DirectComponentPlane, J2kDirectColorPlan, J2kDirectCpuScratch, J2kRect, J2kWaveletTransform,
    Result,
};

/// Execute a adapter direct RGB plan on the CPU and write an RGB8 output region.
///
/// # Errors
///
/// Returns an error for invalid plan geometry, output bounds, or decode-stage failure.
pub fn execute_direct_color_plan_rgb8_into(
    plan: &J2kDirectColorPlan,
    output_region: J2kRect,
    scratch: &mut J2kDirectCpuScratch,
    out: &mut [u8],
    stride: usize,
) -> Result<()> {
    execute_direct_color_plan_u8_into(
        plan,
        output_region,
        scratch,
        out,
        stride,
        DirectColorU8Output::Rgb8,
    )
}

/// Execute a adapter direct RGB plan on the CPU and write an RGBA8 output region.
///
/// # Errors
///
/// Returns an error for invalid plan geometry, output bounds, or decode-stage failure.
pub fn execute_direct_color_plan_rgba8_into(
    plan: &J2kDirectColorPlan,
    output_region: J2kRect,
    scratch: &mut J2kDirectCpuScratch,
    out: &mut [u8],
    stride: usize,
) -> Result<()> {
    execute_direct_color_plan_u8_into(
        plan,
        output_region,
        scratch,
        out,
        stride,
        DirectColorU8Output::Rgba8,
    )
}

#[derive(Clone, Copy)]
enum DirectColorU8Output {
    Rgb8,
    Rgba8,
}

impl DirectColorU8Output {
    const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Rgb8 => 3,
            Self::Rgba8 => 4,
        }
    }
}

fn execute_direct_color_plan_u8_into(
    plan: &J2kDirectColorPlan,
    output_region: J2kRect,
    scratch: &mut J2kDirectCpuScratch,
    out: &mut [u8],
    stride: usize,
    output: DirectColorU8Output,
) -> Result<()> {
    if plan.component_plans.len() != 3 {
        bail!(DecodingError::UnsupportedFeature(
            "direct CPU color plan requires three components"
        ));
    }
    validate_output_region(plan, output_region, out.len(), stride, output)?;

    let workspace_budget = prepare_direct_scratch(plan, scratch)?;
    for (component_index, component_plan) in plan.component_plans.iter().enumerate() {
        let band_scratch = &mut scratch.component_band_sets[component_index];
        let plane = &mut scratch.component_planes[component_index];
        execute_component_plan(component_plan, band_scratch, plane, workspace_budget)?;
    }

    let [plane0, plane1, plane2, ..] = scratch.component_planes.as_mut_slice() else {
        bail!(DecodingError::CodeBlockDecodeFailure);
    };
    if plan.mct {
        apply_inverse_mct(
            plan.transform,
            plan.bit_depths,
            false,
            plane0,
            plane1,
            plane2,
        )?;
    }
    write_rgb8_region(
        [plane0, plane1, plane2],
        plan.bit_depths,
        output_region,
        out,
        stride,
        output,
    )
}

fn apply_inverse_mct(
    transform: J2kWaveletTransform,
    bit_depths: [u8; 3],
    signed: bool,
    plane0: &mut DirectComponentPlane,
    plane1: &mut DirectComponentPlane,
    plane2: &mut DirectComponentPlane,
) -> Result<()> {
    let region = J2kRect {
        x0: 0,
        y0: 0,
        x1: plane0.width,
        y1: plane0.height,
    };
    apply_inverse_mct_region(
        transform, bit_depths, signed, region, plane0, plane1, plane2,
    )
}

pub(super) fn apply_inverse_mct_region(
    transform: J2kWaveletTransform,
    bit_depths: [u8; 3],
    signed: bool,
    region: J2kRect,
    plane0: &mut DirectComponentPlane,
    plane1: &mut DirectComponentPlane,
    plane2: &mut DirectComponentPlane,
) -> Result<()> {
    if plane0.width != plane1.width
        || plane1.width != plane2.width
        || plane0.height != plane1.height
        || plane1.height != plane2.height
        || plane0.samples.len() != plane1.samples.len()
        || plane1.samples.len() != plane2.samples.len()
        || region.x0 > region.x1
        || region.y0 > region.y1
        || region.x1 > plane0.width
        || region.y1 > plane0.height
    {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }

    let addend0 = if signed {
        0.0
    } else {
        sign_addend(bit_depths[0])
    };
    let addend1 = if signed {
        0.0
    } else {
        sign_addend(bit_depths[1])
    };
    let addend2 = if signed {
        0.0
    } else {
        sign_addend(bit_depths[2])
    };
    let plane_width = plane0.width as usize;
    for y in region.y0 as usize..region.y1 as usize {
        let row_start = y
            .checked_mul(plane_width)
            .and_then(|start| start.checked_add(region.x0 as usize))
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        let row_end = row_start
            .checked_add(region.width() as usize)
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        for index in row_start..row_end {
            let src0 = plane0.samples[index];
            let src1 = plane1.samples[index];
            let src2 = plane2.samples[index];
            let (out0, out1, out2) = match transform {
                J2kWaveletTransform::Irreversible97 => (
                    src0 + 1.402 * src2,
                    src0 - 0.34413 * src1 - 0.71414 * src2,
                    src0 + 1.772 * src1,
                ),
                J2kWaveletTransform::Reversible53 => {
                    let i1 = src0 - floor_f32((src2 + src1) * 0.25);
                    (src2 + i1, i1, src1 + i1)
                }
            };
            plane0.samples[index] = out0 + addend0;
            plane1.samples[index] = out1 + addend1;
            plane2.samples[index] = out2 + addend2;
        }
    }
    Ok(())
}

fn write_rgb8_region(
    planes: [&DirectComponentPlane; 3],
    bit_depths: [u8; 3],
    output_region: J2kRect,
    out: &mut [u8],
    stride: usize,
    output: DirectColorU8Output,
) -> Result<()> {
    let width = output_region.width() as usize;
    let height = output_region.height() as usize;
    let bytes_per_pixel = output.bytes_per_pixel();
    let row_bytes = width
        .checked_mul(bytes_per_pixel)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    for plane in planes {
        if output_region.x1 > plane.width || output_region.y1 > plane.height {
            bail!(DecodingError::CodeBlockDecodeFailure);
        }
    }

    for y in 0..height {
        let src_y = output_region.y0 as usize + y;
        let dst = &mut out[y * stride..y * stride + row_bytes];
        for x in 0..width {
            let src_x = output_region.x0 as usize + x;
            let dst = &mut dst[x * bytes_per_pixel..x * bytes_per_pixel + bytes_per_pixel];
            for channel in 0..3 {
                let plane = planes[channel];
                let sample = plane.samples[src_y * plane.width as usize + src_x];
                dst[channel] = sample_as_u8(sample, bit_depths[channel]);
            }
            if matches!(output, DirectColorU8Output::Rgba8) {
                dst[3] = u8::MAX;
            }
        }
    }
    Ok(())
}

fn validate_output_region(
    plan: &J2kDirectColorPlan,
    output_region: J2kRect,
    out_len: usize,
    stride: usize,
    output: DirectColorU8Output,
) -> Result<()> {
    if output_region.x1 > plan.dimensions.0
        || output_region.y1 > plan.dimensions.1
        || output_region.x0 > output_region.x1
        || output_region.y0 > output_region.y1
    {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    let bytes_per_pixel = u32::try_from(output.bytes_per_pixel())
        .map_err(|_| DecodingError::CodeBlockDecodeFailure)?;
    let row_bytes = output_region
        .width()
        .checked_mul(bytes_per_pixel)
        .and_then(|len| usize::try_from(len).ok())
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    if stride < row_bytes {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    let height = usize::try_from(output_region.height())
        .map_err(|_| DecodingError::CodeBlockDecodeFailure)?;
    let required = if height == 0 {
        0
    } else {
        stride
            .checked_mul(height - 1)
            .and_then(|prefix| prefix.checked_add(row_bytes))
            .ok_or(DecodingError::CodeBlockDecodeFailure)?
    };
    if out_len < required {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    Ok(())
}

#[expect(
    clippy::cast_precision_loss,
    reason = "direct CPU color math intentionally uses the decoder's f32 sample representation"
)]
fn sign_addend(bit_depth: u8) -> f32 {
    (1_u32 << (bit_depth - 1)) as f32
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "samples are rounded and clamped before stable 8-bit output quantization"
)]
fn sample_as_u8(sample: f32, bit_depth: u8) -> u8 {
    let rounded = round_f32(sample);
    if bit_depth == 8 {
        return rounded.clamp(0.0, f32::from(u8::MAX)) as u8;
    }
    let max_value = if bit_depth >= 16 {
        f32::from(u16::MAX)
    } else {
        f32::from(((1_u16 << bit_depth) - 1).max(1))
    };
    round_f32((rounded.clamp(0.0, max_value) / max_value) * f32::from(u8::MAX)) as u8
}
