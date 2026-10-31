// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU output allocation, exact sample conversion, and fallback materialization.

use super::{
    backend, decode_prepared_plan_samples, size_of, BatchDecodeOptions, BatchGroupInfo,
    BatchInfrastructureError, BatchLayout, BatchWorker, CpuDecodeParallelism, CpuExecution,
    J2kError, NativeSampleType, PreparedImage, Rect, Vec,
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
    let decoded = backend::image_at_scale(image.bytes(), options.settings, plan.scale())?;
    let output_rect = plan.output_rect();
    let decoded_dims = (decoded.width(), decoded.height());
    // Every batch sample type is an integer output, so irreversible samples
    // are rounded before the level shift exactly as the single-image decode.
    if output_rect == Rect::full(decoded_dims) {
        decoded
            .decode_components_for_integer_output_with_context(context)
            .map_err(J2kError::from_native_decode_error)
    } else {
        decoded
            .decode_region_components_for_integer_output_with_context(
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
    match image.cpu_execution() {
        CpuExecution::ExceedsDecodeBound => return Err(CpuExecution::decode_bound_error()),
        CpuExecution::StreamedRows => {
            return decode_streamed_samples(
                image,
                options,
                info,
                parallelism,
                context,
                out,
                convert_u8,
            );
        }
        CpuExecution::OffsetPlan { .. } | CpuExecution::ComponentPlanes => {}
    }
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
    if image.cpu_execution() == CpuExecution::ExceedsDecodeBound {
        return Err(CpuExecution::decode_bound_error());
    }
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
    if image.cpu_execution() == CpuExecution::ExceedsDecodeBound {
        return Err(CpuExecution::decode_bound_error());
    }
    if decode_prepared_plan_samples(image, info, worker, out, round_signed)? {
        return Ok(());
    }
    let components = image_components(image, options, parallelism, context)?;
    validate_decoded_components(&components, info)?;
    write_samples(&components, info, out, round_signed);
    Ok(())
}

/// Decode `image` through the bounded region decode, which hands over its
/// rows as they are finished, writing each sample exactly as
/// [`write_samples`] writes whole planes. Working memory stays bounded by the
/// streamed decode instead of growing with the image.
fn decode_streamed_samples<'image, T: Copy>(
    image: &'image PreparedImage,
    options: BatchDecodeOptions,
    info: &BatchGroupInfo,
    parallelism: CpuDecodeParallelism,
    context: &mut j2k_native::DecoderContext<'image>,
    out: &mut [T],
    convert: fn(f32, u8) -> T,
) -> Result<(), J2kError> {
    context.set_cpu_decode_parallelism(parallelism.to_native());
    let plan = image.plan();
    let decoded = backend::image_at_scale(image.bytes(), options.settings, plan.scale())?;
    let output_rect = plan.output_rect();
    if (output_rect.w, output_rect.h) != info.dimensions {
        return Err(J2kError::internal_backend(
            "prepared batch output metadata changed during decode",
        ));
    }
    let mut sink = SampleRowSink::new(info, out, convert)?;
    decoded
        .decode_region_rows_for_integer_output_with_context(
            (output_rect.x, output_rect.y, output_rect.w, output_rect.h),
            context,
            &mut sink,
        )
        .map_err(J2kError::from_native_decode_error)?;
    sink.finish()
}

/// Writes streamed region rectangles into one image's dense batch samples.
struct SampleRowSink<'out, 'info, T> {
    info: &'info BatchGroupInfo,
    out: &'out mut [T],
    convert: fn(f32, u8) -> T,
    pixels: usize,
    written_pixels: usize,
    malformed: bool,
}

impl<'out, 'info, T: Copy> SampleRowSink<'out, 'info, T> {
    fn new(
        info: &'info BatchGroupInfo,
        out: &'out mut [T],
        convert: fn(f32, u8) -> T,
    ) -> Result<Self, J2kError> {
        let pixels = (info.dimensions.0 as usize)
            .checked_mul(info.dimensions.1 as usize)
            .ok_or_else(|| J2kError::internal_backend("prepared batch sample count overflow"))?;
        if pixels.checked_mul(info.color.channels()) != Some(out.len()) {
            return Err(J2kError::internal_backend(
                "prepared batch output slot does not match its image",
            ));
        }
        Ok(Self {
            info,
            out,
            convert,
            pixels,
            written_pixels: 0,
            malformed: false,
        })
    }

    /// Every pixel arrived exactly once (rectangles never overlap).
    fn finish(self) -> Result<(), J2kError> {
        if self.malformed || self.written_pixels != self.pixels {
            return Err(J2kError::internal_backend(
                "streamed batch decode did not cover its output exactly",
            ));
        }
        Ok(())
    }

    fn write_rectangle(&mut self, x: u32, y: u32, planes: &[j2k_native::ComponentPlane<'_>]) {
        let channels = self.info.color.channels();
        let width = self.info.dimensions.0 as usize;
        let Some((rect_width, rect_height)) = planes.first().map(|plane| plane.dimensions()) else {
            self.malformed = true;
            return;
        };
        let (x, y, rect_width, rect_height) = (
            x as usize,
            y as usize,
            rect_width as usize,
            rect_height as usize,
        );
        let rect_pixels = rect_width * rect_height;
        if planes.len() != channels
            || x + rect_width > width
            || y + rect_height > self.info.dimensions.1 as usize
            || planes.iter().any(|plane| {
                plane.dimensions() != (rect_width as u32, rect_height as u32)
                    || plane.samples().len() < rect_pixels
            })
        {
            self.malformed = true;
            return;
        }
        let precision = self.info.precision;
        let planar = self.info.layout == BatchLayout::Nchw || channels == 1;
        for (channel, plane) in planes.iter().enumerate() {
            for (row, samples) in plane.samples()[..rect_pixels]
                .chunks_exact(rect_width)
                .enumerate()
            {
                let pixel = (y + row) * width + x;
                if planar {
                    let destination = &mut self.out[channel * self.pixels + pixel..][..rect_width];
                    for (destination, &sample) in destination.iter_mut().zip(samples) {
                        *destination = (self.convert)(sample, precision);
                    }
                } else {
                    let destination = &mut self.out[pixel * channels..][..rect_width * channels];
                    for (pixel, &sample) in destination.chunks_exact_mut(channels).zip(samples) {
                        pixel[channel] = (self.convert)(sample, precision);
                    }
                }
            }
        }
        self.written_pixels += rect_pixels;
    }
}

impl<T: Copy> j2k_native::ComponentRowSink for SampleRowSink<'_, '_, T> {
    fn write_rows(&mut self, x: u32, y: u32, planes: &[j2k_native::ComponentPlane<'_>]) {
        if !self.malformed {
            self.write_rectangle(x, y, planes);
        }
    }
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
    let precision = info.precision;
    for (channel, plane) in components.planes().iter().enumerate() {
        let samples = &plane.samples()[..pixels];
        if info.layout == BatchLayout::Nchw || channels == 1 {
            // One contiguous plane per channel.
            let destination = &mut out[channel * pixels..][..pixels];
            for (destination, &sample) in destination.iter_mut().zip(samples) {
                *destination = convert(sample, precision);
            }
        } else {
            for (pixel, &sample) in out.chunks_exact_mut(channels).zip(samples) {
                pixel[channel] = convert(sample, precision);
            }
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
