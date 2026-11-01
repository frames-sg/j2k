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
        CpuExecution::ExceedsDecodeBound => Err(CpuExecution::decode_bound_error()),
        CpuExecution::StreamedRows | CpuExecution::StreamedRowsOrPlan { .. } => {
            decode_streamed_samples(
                image,
                options,
                info,
                parallelism,
                context,
                out,
                convert_u8_row,
            )
        }
        CpuExecution::OffsetPlan { .. } | CpuExecution::ComponentPlanes => {
            decode_planned_u8(image, options, info, parallelism, context, worker, out)
        }
    }
}

/// [`decode_image_u8`] for the only image of a Gray8 group: an image that may
/// stream or decode its plan decodes its plan, which spreads one component's
/// code-block rows over every core while nothing else is in flight.
pub(super) fn decode_lone_gray_u8<'image>(
    image: &'image PreparedImage,
    options: BatchDecodeOptions,
    info: &BatchGroupInfo,
    parallelism: CpuDecodeParallelism,
    context: &mut j2k_native::DecoderContext<'image>,
    worker: &mut BatchWorker,
    out: &mut [u8],
) -> Result<(), J2kError> {
    if matches!(
        image.cpu_execution(),
        CpuExecution::StreamedRowsOrPlan { .. }
    ) {
        decode_planned_u8(image, options, info, parallelism, context, worker, out)
    } else {
        decode_image_u8(image, options, info, parallelism, context, worker, out)
    }
}

fn decode_planned_u8<'image>(
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
/// [`write_samples`] writes whole planes, a row at a time through
/// `convert_row` (see [`convert_u8_row`]). Working memory stays bounded by
/// the streamed decode instead of growing with the image.
fn decode_streamed_samples<'image, T: Copy>(
    image: &'image PreparedImage,
    options: BatchDecodeOptions,
    info: &BatchGroupInfo,
    parallelism: CpuDecodeParallelism,
    context: &mut j2k_native::DecoderContext<'image>,
    out: &mut [T],
    convert_row: ConvertRow<T>,
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
    let mut sink = SampleRowSink::new(info, out, convert_row)?;
    decoded
        .decode_region_rows_for_integer_output_with_context(
            (output_rect.x, output_rect.y, output_rect.w, output_rect.h),
            context,
            &mut sink,
        )
        .map_err(J2kError::from_native_decode_error)?;
    sink.finish()
}

/// Converts a row of decoded samples of the given precision into
/// `destination`, one sample every `stride` elements from its start.
type ConvertRow<T> = fn(&mut [T], usize, &[f32], u8);

/// Writes streamed region rectangles into one image's dense batch samples.
/// The decode hands each batch of rows over on its reading thread, between
/// the batches it reconstructs, while the pool's band workers decode only as
/// far ahead as their rings hold, so the rows convert a row at a time.
struct SampleRowSink<'out, 'info, T> {
    info: &'info BatchGroupInfo,
    out: &'out mut [T],
    convert_row: ConvertRow<T>,
    pixels: usize,
    written_pixels: usize,
    malformed: bool,
}

impl<'out, 'info, T: Copy> SampleRowSink<'out, 'info, T> {
    fn new(
        info: &'info BatchGroupInfo,
        out: &'out mut [T],
        convert_row: ConvertRow<T>,
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
            convert_row,
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
        let Some(rect_dimensions) = planes.first().map(j2k_native::ComponentPlane::dimensions)
        else {
            self.malformed = true;
            return;
        };
        let (x, y, rect_width, rect_height) = (
            x as usize,
            y as usize,
            rect_dimensions.0 as usize,
            rect_dimensions.1 as usize,
        );
        let rect_pixels = rect_width * rect_height;
        if planes.len() != channels
            || x + rect_width > width
            || y + rect_height > self.info.dimensions.1 as usize
            || planes.iter().any(|plane| {
                plane.dimensions() != rect_dimensions || plane.samples().len() < rect_pixels
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
                    (self.convert_row)(destination, 1, samples, precision);
                } else {
                    let destination = &mut self.out[pixel * channels..][..rect_width * channels];
                    (self.convert_row)(&mut destination[channel..], channels, samples, precision);
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

/// [`convert_u8`] for a row of `samples`, written `stride` apart from the
/// start of `destination`. It rounds and saturates in `f32`, which gives the
/// same bytes: `f32::round` is exact and, above zero, equals the `f64`
/// floor of the sample plus one half, and any value past the precision's
/// maximum saturates. Per sample, `convert_u8` costs a call and `f64` math;
/// over a row, the compiler can vectorize the contiguous rows of planar
/// output, and interleaved rows convert in a tight loop.
fn convert_u8_row(destination: &mut [u8], stride: usize, samples: &[f32], precision: u8) {
    #[expect(
        clippy::cast_precision_loss,
        reason = "a u8 output's maximum is far below f32's exact integer range"
    )]
    let ceiling = ((1_u32 << u32::from(precision)) - 1) as f32;
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the sample is positive and rounded below the ceiling before the cast, which saturates"
    )]
    let round = |sample: f32| -> u8 {
        let rounded = if sample > 0.0 {
            sample.min(ceiling).round() as u32
        } else {
            0
        };
        u8::try_from(rounded).unwrap_or(u8::MAX)
    };
    if stride == 1 {
        for (destination, &sample) in destination.iter_mut().zip(samples) {
            *destination = round(sample);
        }
    } else {
        for (destination, &sample) in destination.iter_mut().step_by(stride).zip(samples) {
            *destination = round(sample);
        }
    }
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

#[cfg(test)]
mod tests {
    use super::{convert_u8, convert_u8_row};
    use alloc::vec::Vec;

    /// Samples around every rounding and saturation boundary of 8-bit and
    /// narrower outputs, and the values no decode should produce.
    fn edge_samples() -> Vec<f32> {
        let mut samples = vec![
            0.0,
            -0.0,
            f32::from_bits(1),
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            -1.0e-30,
            f32::NAN,
            -f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MAX,
            -f32::MAX,
            1.0e9,
            65_535.5,
            16_777_216.0,
            16_777_218.0,
        ];
        for whole in -2_i16..=300 {
            let whole = f32::from(whole);
            for offset in [-0.5_f32, -0.25, -1.0e-3, 0.0, 1.0e-3, 0.25, 0.5] {
                let sample = whole + offset;
                samples.push(sample);
                for ulps in 1..=2 {
                    samples.push(f32::from_bits(sample.to_bits().wrapping_add(ulps)));
                    samples.push(f32::from_bits(sample.to_bits().wrapping_sub(ulps)));
                }
            }
        }
        samples
    }

    #[test]
    fn rows_convert_as_each_sample_does() {
        let samples = edge_samples();
        for precision in 1..=16 {
            let expected: Vec<u8> = samples
                .iter()
                .map(|&sample| convert_u8(sample, precision))
                .collect();
            let mut planar = vec![0x5a; samples.len()];
            convert_u8_row(&mut planar, 1, &samples, precision);
            assert_eq!(planar, expected, "planar, precision {precision}");
            // Interleaved: every third byte, from the second, and none other.
            let mut interleaved = vec![0x5a; samples.len() * 3 - 1];
            convert_u8_row(&mut interleaved[1..], 3, &samples, precision);
            for (index, &byte) in interleaved.iter().enumerate() {
                if index % 3 == 1 {
                    assert_eq!(
                        byte,
                        expected[index / 3],
                        "sample {index}, precision {precision}"
                    );
                } else {
                    assert_eq!(
                        byte, 0x5a,
                        "byte {index} between samples, precision {precision}"
                    );
                }
            }
        }
    }
}
