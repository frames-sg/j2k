// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU output allocation, exact sample conversion, and fallback materialization.

use super::{
    backend, decode_prepared_plan_samples, size_of, BatchDecodeOptions, BatchGroupInfo,
    BatchInfrastructureError, BatchLayout, BatchWorker, CpuDecodeParallelism, Downscale, J2kError,
    NativeSampleType, PreparedImage, Rect, Vec,
};

pub(super) fn ensure_batch_output_within_cap(
    samples: usize,
    sample_type: NativeSampleType,
) -> Result<(), BatchInfrastructureError> {
    let width = match sample_type {
        NativeSampleType::U8 => size_of::<u8>(),
        NativeSampleType::U16 => size_of::<u16>(),
        NativeSampleType::I16 => size_of::<i16>(),
        _ => {
            return Err(BatchInfrastructureError::UnsupportedContract {
                what: "owned CPU batch sample width",
            });
        }
    };
    let bytes = samples
        .checked_mul(width)
        .ok_or(BatchInfrastructureError::AllocationTooLarge {
            what: "J2K owned batch output",
            requested: usize::MAX,
            cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        })?;
    if bytes > j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES {
        return Err(BatchInfrastructureError::AllocationTooLarge {
            what: "J2K owned batch output",
            requested: bytes,
            cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        });
    }
    Ok(())
}

pub(super) fn try_zeroed_vec<T: Clone>(
    len: usize,
    zero: T,
) -> Result<Vec<T>, BatchInfrastructureError> {
    let bytes = len.saturating_mul(size_of::<T>());
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| BatchInfrastructureError::HostAllocationFailed {
            what: "J2K owned batch output",
            bytes,
        })?;
    values.resize(len, zero);
    Ok(values)
}

fn image_components<'ctx, 'input>(
    image: &'input PreparedImage,
    options: BatchDecodeOptions,
    parallelism: CpuDecodeParallelism,
    context: &'ctx mut j2k_native::DecoderContext<'input>,
) -> Result<j2k_native::DecodedComponents<'ctx>, J2kError> {
    context.set_cpu_decode_parallelism(parallelism.to_native());
    let plan = image.plan();
    let target_resolution = (plan.scale() != Downscale::None).then_some((
        plan.source_dims().0.div_ceil(plan.scale().denominator()),
        plan.source_dims().1.div_ceil(plan.scale().denominator()),
    ));
    let decoded = backend::image(image.bytes(), options.settings, target_resolution)?;
    let output_rect = plan.output_rect();
    let decoded_dims = (decoded.width(), decoded.height());
    if output_rect == Rect::full(decoded_dims) {
        decoded
            .decode_components_with_context(context)
            .map_err(J2kError::from_native_decode_error)
    } else {
        decoded
            .decode_region_components_with_context(
                (output_rect.x, output_rect.y, output_rect.w, output_rect.h),
                context,
            )
            .map_err(J2kError::from_native_decode_error)
    }
}

pub(super) fn decode_image_u8<'image>(
    image: &'image PreparedImage,
    options: BatchDecodeOptions,
    info: &BatchGroupInfo,
    parallelism: CpuDecodeParallelism,
    context: &mut j2k_native::DecoderContext<'image>,
    worker: &mut BatchWorker,
    out: &mut [u8],
) -> Result<(), J2kError> {
    if decode_prepared_plan_samples(image, info, worker, out, |sample, precision| {
        u8::try_from(round_unsigned(sample, precision)).unwrap_or(u8::MAX)
    })? {
        return Ok(());
    }
    let components = image_components(image, options, parallelism, context)?;
    validate_decoded_components(&components, info)?;
    write_samples(&components, info, out, |sample, precision| {
        u8::try_from(round_unsigned(sample, precision)).unwrap_or(u8::MAX)
    });
    Ok(())
}

pub(super) fn decode_image_u16<'image>(
    image: &'image PreparedImage,
    options: BatchDecodeOptions,
    info: &BatchGroupInfo,
    parallelism: CpuDecodeParallelism,
    context: &mut j2k_native::DecoderContext<'image>,
    worker: &mut BatchWorker,
    out: &mut [u16],
) -> Result<(), J2kError> {
    if decode_prepared_plan_samples(image, info, worker, out, |sample, precision| {
        u16::try_from(round_unsigned(sample, precision)).unwrap_or(u16::MAX)
    })? {
        return Ok(());
    }
    let components = image_components(image, options, parallelism, context)?;
    validate_decoded_components(&components, info)?;
    write_samples(&components, info, out, |sample, precision| {
        u16::try_from(round_unsigned(sample, precision)).unwrap_or(u16::MAX)
    });
    Ok(())
}

pub(super) fn decode_image_i16<'image>(
    image: &'image PreparedImage,
    options: BatchDecodeOptions,
    info: &BatchGroupInfo,
    parallelism: CpuDecodeParallelism,
    context: &mut j2k_native::DecoderContext<'image>,
    worker: &mut BatchWorker,
    out: &mut [i16],
) -> Result<(), J2kError> {
    if decode_prepared_plan_samples(image, info, worker, out, round_signed)? {
        return Ok(());
    }
    let components = image_components(image, options, parallelism, context)?;
    validate_decoded_components(&components, info)?;
    write_samples(&components, info, out, round_signed);
    Ok(())
}

fn validate_decoded_components(
    components: &j2k_native::DecodedComponents<'_>,
    info: &BatchGroupInfo,
) -> Result<(), J2kError> {
    if components.dimensions() != info.dimensions
        || components.planes().len() != info.color.channels()
    {
        return Err(J2kError::internal_backend(
            "prepared batch output metadata changed during decode",
        ));
    }
    let expected = (info.dimensions.0 as usize)
        .checked_mul(info.dimensions.1 as usize)
        .ok_or_else(|| J2kError::internal_backend("prepared batch sample count overflow"))?;
    for (component, plane) in components.planes().iter().enumerate() {
        if plane.dimensions() != info.dimensions || plane.samples().len() < expected {
            return Err(J2kError::BackendComponentPlaneTooShort {
                component,
                samples: plane.samples().len(),
                expected,
            });
        }
    }
    Ok(())
}

fn write_samples<T: Copy>(
    components: &j2k_native::DecodedComponents<'_>,
    info: &BatchGroupInfo,
    out: &mut [T],
    convert: impl Fn(f32, u8) -> T,
) {
    let pixels = info.dimensions.0 as usize * info.dimensions.1 as usize;
    let channels = info.color.channels();
    for pixel in 0..pixels {
        for (channel, plane) in components.planes().iter().enumerate() {
            let destination = match info.layout {
                BatchLayout::Nchw => channel * pixels + pixel,
                BatchLayout::Nhwc => pixel * channels + channel,
            };
            out[destination] = convert(plane.samples()[pixel], info.precision);
        }
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "sample is rounded only after clamping to the declared unsigned precision"
)]
fn round_unsigned(sample: f32, precision: u8) -> u32 {
    let max = (1_u32 << u32::from(precision)) - 1;
    if sample.is_nan() || sample <= 0.0 {
        0
    } else if f64::from(sample) >= f64::from(max) {
        max
    } else {
        (f64::from(sample) + 0.5) as u32
    }
}

pub(super) fn convert_u8(sample: f32, precision: u8) -> u8 {
    u8::try_from(round_unsigned(sample, precision)).unwrap_or(u8::MAX)
}

pub(super) fn convert_u16(sample: f32, precision: u8) -> u16 {
    u16::try_from(round_unsigned(sample, precision)).unwrap_or(u16::MAX)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "sample is rounded only after clamping to the declared signed precision"
)]
pub(super) fn round_signed(sample: f32, precision: u8) -> i16 {
    let magnitude_bits = u32::from(precision.saturating_sub(1));
    let min = -(1_i32 << magnitude_bits);
    let max = (1_i32 << magnitude_bits) - 1;
    let sample = f64::from(sample);
    let rounded = if sample.is_nan() {
        0
    } else if sample <= f64::from(min) {
        min
    } else if sample >= f64::from(max) {
        max
    } else if sample >= 0.0 {
        (sample + 0.5) as i32
    } else {
        (sample - 0.5) as i32
    };
    i16::try_from(rounded).unwrap_or(if rounded.is_negative() {
        i16::MIN
    } else {
        i16::MAX
    })
}
