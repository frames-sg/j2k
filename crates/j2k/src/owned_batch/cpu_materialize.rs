// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU output allocation, exact sample conversion, and fallback materialization.

use super::{
    backend, decode_prepared_plan_samples, size_of, BatchDecodeOptions, BatchGroupInfo,
    BatchInfrastructureError, BatchLayout, BatchWorker, CpuDecodeParallelism, CpuExecution,
    CpuPreparedImage, J2kError, Rect, Vec,
};

pub(super) fn ensure_batch_output_within_cap(
    info: &BatchGroupInfo,
    image_count: usize,
) -> Result<(), BatchInfrastructureError> {
    if info.sample_bytes().is_none() {
        return Err(BatchInfrastructureError::UnsupportedContract {
            what: "owned CPU batch sample width",
        });
    }
    let bytes =
        info.output_bytes(image_count)
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
    image: &'input CpuPreparedImage,
    _options: BatchDecodeOptions,
    parallelism: CpuDecodeParallelism,
    context: &'ctx mut j2k_native::DecoderContext<'input>,
    output_bytes: usize,
) -> Result<j2k_native::DecodedComponents<'ctx>, J2kError> {
    context.set_cpu_decode_parallelism(parallelism.to_native());
    let decoded = backend::image_for_operation(
        image.bytes(),
        image.settings(),
        image.reduction_levels(),
        image.retained_bytes().saturating_add(output_bytes),
        image.control(),
    )?;
    let output_rect = image.output_rect();
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
    image: &'image CpuPreparedImage,
    options: BatchDecodeOptions,
    info: &BatchGroupInfo,
    parallelism: CpuDecodeParallelism,
    context: &mut j2k_native::DecoderContext<'image>,
    worker: &mut BatchWorker,
    (out, output_bytes): (&mut [u8], usize),
) -> Result<(), J2kError> {
    match image.cpu_execution() {
        CpuExecution::ExceedsDecodeBound => Err(CpuExecution::decode_bound_error()),
        CpuExecution::StreamedRows | CpuExecution::StreamedRowsOrPlan { .. } => {
            let streamed = decode_streamed_samples(
                image,
                options,
                info,
                parallelism,
                context,
                (out, output_bytes),
                convert_u8_row,
            );
            if matches!(streamed, Err(J2kError::ResourceLimit { .. }))
                && matches!(
                    image.cpu_execution(),
                    CpuExecution::StreamedRowsOrPlan { .. }
                )
            {
                // The receiving session admitted this plan. Release streamed
                // scratch before retrying; the plan rewrites the entire owned
                // output, including any rows already produced by the stream.
                *context = context.fresh_for_operation();
                decode_planned_u8(
                    image,
                    options,
                    info,
                    parallelism,
                    context,
                    worker,
                    (out, output_bytes),
                )
            } else {
                streamed
            }
        }
        CpuExecution::OffsetPlan { .. } | CpuExecution::ComponentPlanes => decode_planned_u8(
            image,
            options,
            info,
            parallelism,
            context,
            worker,
            (out, output_bytes),
        ),
    }
}

/// [`decode_image_u8`] for the only image of a Gray8 group: an image that may
/// stream or decode its plan decodes its plan, which spreads one component's
/// code-block rows over every core while nothing else is in flight.
pub(super) fn decode_lone_gray_u8<'image>(
    image: &'image CpuPreparedImage,
    options: BatchDecodeOptions,
    info: &BatchGroupInfo,
    parallelism: CpuDecodeParallelism,
    context: &mut j2k_native::DecoderContext<'image>,
    worker: &mut BatchWorker,
    (out, output_bytes): (&mut [u8], usize),
) -> Result<(), J2kError> {
    if matches!(
        image.cpu_execution(),
        CpuExecution::StreamedRowsOrPlan { .. }
    ) {
        decode_planned_u8(
            image,
            options,
            info,
            parallelism,
            context,
            worker,
            (out, output_bytes),
        )
    } else {
        decode_image_u8(
            image,
            options,
            info,
            parallelism,
            context,
            worker,
            (out, output_bytes),
        )
    }
}

fn decode_planned_u8<'image>(
    image: &'image CpuPreparedImage,
    options: BatchDecodeOptions,
    info: &BatchGroupInfo,
    parallelism: CpuDecodeParallelism,
    context: &mut j2k_native::DecoderContext<'image>,
    worker: &mut BatchWorker,
    (out, output_bytes): (&mut [u8], usize),
) -> Result<(), J2kError> {
    if decode_prepared_plan_samples(image, info, worker, out, |sample, precision| {
        u8::try_from(round_unsigned(sample, precision)).unwrap_or(u8::MAX)
    })? {
        return Ok(());
    }
    let components = image_components(image, options, parallelism, context, output_bytes)?;
    write_decoded_samples(&components, info, out, |sample, precision| {
        u8::try_from(round_unsigned(sample, precision)).unwrap_or(u8::MAX)
    })
}

pub(super) fn decode_image_u16<'image>(
    image: &'image CpuPreparedImage,
    options: BatchDecodeOptions,
    info: &BatchGroupInfo,
    parallelism: CpuDecodeParallelism,
    context: &mut j2k_native::DecoderContext<'image>,
    worker: &mut BatchWorker,
    (out, output_bytes): (&mut [u16], usize),
) -> Result<(), J2kError> {
    match image.cpu_execution() {
        CpuExecution::ExceedsDecodeBound => return Err(CpuExecution::decode_bound_error()),
        CpuExecution::StreamedRows | CpuExecution::StreamedRowsOrPlan { .. } => {
            return decode_streamed_samples(
                image,
                options,
                info,
                parallelism,
                context,
                (out, output_bytes),
                convert_u16_row,
            );
        }
        CpuExecution::OffsetPlan { .. } | CpuExecution::ComponentPlanes => {}
    }
    if decode_prepared_plan_samples(image, info, worker, out, |sample, precision| {
        u16::try_from(round_unsigned(sample, precision)).unwrap_or(u16::MAX)
    })? {
        return Ok(());
    }
    let components = image_components(image, options, parallelism, context, output_bytes)?;
    write_decoded_samples(&components, info, out, |sample, precision| {
        u16::try_from(round_unsigned(sample, precision)).unwrap_or(u16::MAX)
    })
}

pub(super) fn decode_image_i16<'image>(
    image: &'image CpuPreparedImage,
    options: BatchDecodeOptions,
    info: &BatchGroupInfo,
    parallelism: CpuDecodeParallelism,
    context: &mut j2k_native::DecoderContext<'image>,
    worker: &mut BatchWorker,
    (out, output_bytes): (&mut [i16], usize),
) -> Result<(), J2kError> {
    match image.cpu_execution() {
        CpuExecution::ExceedsDecodeBound => return Err(CpuExecution::decode_bound_error()),
        CpuExecution::StreamedRows | CpuExecution::StreamedRowsOrPlan { .. } => {
            return decode_streamed_samples(
                image,
                options,
                info,
                parallelism,
                context,
                (out, output_bytes),
                convert_i16_row,
            );
        }
        CpuExecution::OffsetPlan { .. } | CpuExecution::ComponentPlanes => {}
    }
    if decode_prepared_plan_samples(image, info, worker, out, round_signed)? {
        return Ok(());
    }
    let components = image_components(image, options, parallelism, context, output_bytes)?;
    write_decoded_samples(&components, info, out, round_signed)
}

/// Decode `image` through the bounded region decode, which hands over its
/// rows as they are finished, writing each sample exactly as
/// [`write_batch_samples`] writes whole planes, a row at a time through
/// `convert_row` (see [`convert_u8_row`]). Working memory stays bounded by
/// the streamed decode instead of growing with the image.
fn decode_streamed_samples<'image, T: Copy>(
    image: &'image CpuPreparedImage,
    _options: BatchDecodeOptions,
    info: &BatchGroupInfo,
    parallelism: CpuDecodeParallelism,
    context: &mut j2k_native::DecoderContext<'image>,
    (out, output_bytes): (&mut [T], usize),
    convert_row: ConvertRow<T>,
) -> Result<(), J2kError> {
    context.set_cpu_decode_parallelism(parallelism.to_native());
    let decoded = backend::image_for_operation(
        image.bytes(),
        image.settings(),
        image.reduction_levels(),
        image.retained_bytes().saturating_add(output_bytes),
        image.control(),
    )?;
    let output_rect = image.output_rect();
    if (output_rect.w, output_rect.h) != info.dimensions {
        return Err(J2kError::internal_backend(
            "prepared batch output metadata changed during decode",
        ));
    }
    let mut sink = SampleRowSink::new(info, out, convert_row)?;
    #[cfg(test)]
    let baseline = STREAMED_RETAINED_BASELINE.with(core::cell::Cell::get);
    #[cfg(not(test))]
    let baseline = 0;
    let result = decoded.decode_region_rows_for_integer_output_with_retained_baseline(
        (output_rect.x, output_rect.y, output_rect.w, output_rect.h),
        context,
        &mut sink,
        baseline,
    );
    // A malformed hand-over stops the decode; report it rather than the stop.
    if sink.malformed {
        return sink.finish();
    }
    result.map_err(J2kError::from_native_decode_error)?;
    sink.finish()
}

#[cfg(test)]
std::thread_local! {
    /// Lower the streamed decode's remaining budget on small test images.
    pub(super) static STREAMED_RETAINED_BASELINE: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
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
        {
            self.malformed = true;
            return;
        }
        if planes.iter().enumerate().any(|(component, plane)| {
            validate_sample_plane(
                component,
                plane.dimensions(),
                plane.bit_depth(),
                plane.samples().len(),
                rect_dimensions,
                self.info.precision,
                rect_pixels,
            )
            .is_err()
        }) {
            self.malformed = true;
            return;
        }
        for (channel, plane) in planes.iter().enumerate() {
            for (row, samples) in plane.samples()[..rect_pixels]
                .chunks_exact(rect_width)
                .enumerate()
            {
                let pixel = (y + row) * width + x;
                let Some((destination, stride)) = channel_destination(
                    self.out,
                    self.info.layout,
                    self.pixels,
                    channels,
                    channel,
                    pixel,
                    rect_width,
                ) else {
                    self.malformed = true;
                    return;
                };
                (self.convert_row)(destination, stride, samples, self.info.precision);
            }
        }
        self.written_pixels += rect_pixels;
    }
}

impl<T: Copy> j2k_native::ComponentRowSink for SampleRowSink<'_, '_, T> {
    fn stopped(&self) -> bool {
        self.malformed
    }

    fn write_rows(&mut self, x: u32, y: u32, planes: &[j2k_native::ComponentPlane<'_>]) {
        if !self.malformed {
            self.write_rectangle(x, y, planes);
        }
    }
}

fn write_decoded_samples<T: Copy>(
    components: &j2k_native::DecodedComponents<'_>,
    info: &BatchGroupInfo,
    out: &mut [T],
    convert: impl Fn(f32, u8) -> T,
) -> Result<(), J2kError> {
    write_batch_samples(
        components.dimensions(),
        components.planes().len(),
        |component| {
            components
                .planes()
                .get(component)
                .map(|plane| (plane.dimensions(), plane.bit_depth(), plane.samples()))
        },
        info,
        out,
        convert,
    )
}

/// Validate and write every decoded plane through the batch's native layout.
pub(super) fn write_batch_samples<'a, T: Copy>(
    dimensions: (u32, u32),
    component_count: usize,
    mut plane: impl FnMut(usize) -> Option<((u32, u32), u8, &'a [f32])>,
    info: &BatchGroupInfo,
    out: &mut [T],
    convert: impl Fn(f32, u8) -> T,
) -> Result<(), J2kError> {
    if dimensions != info.dimensions
        || component_count != info.color.channels()
        || info.samples_per_image() != Some(out.len())
    {
        return Err(J2kError::internal_backend(
            "prepared batch output metadata changed during decode",
        ));
    }
    let pixels = (info.dimensions.0 as usize)
        .checked_mul(info.dimensions.1 as usize)
        .ok_or_else(|| J2kError::internal_backend("prepared batch sample count overflow"))?;
    let channels = info.color.channels();
    for channel in 0..channels {
        let (plane_dimensions, bit_depth, samples) = plane(channel).ok_or_else(|| {
            J2kError::internal_backend("prepared codec component plane is missing")
        })?;
        validate_sample_plane(
            channel,
            plane_dimensions,
            bit_depth,
            samples.len(),
            info.dimensions,
            info.precision,
            pixels,
        )?;
        let (destination, stride) =
            channel_destination(out, info.layout, pixels, channels, channel, 0, pixels)
                .ok_or_else(|| {
                    J2kError::internal_backend("prepared batch output slot is too short")
                })?;
        if stride == 1 {
            for (destination, &sample) in destination.iter_mut().zip(&samples[..pixels]) {
                *destination = convert(sample, bit_depth);
            }
        } else {
            for (destination, &sample) in destination
                .iter_mut()
                .step_by(stride)
                .zip(&samples[..pixels])
            {
                *destination = convert(sample, bit_depth);
            }
        }
    }
    Ok(())
}

/// Validate one decoded sample plane against its prepared batch contract.
pub(super) fn validate_sample_plane(
    component: usize,
    dimensions: (u32, u32),
    bit_depth: u8,
    samples: usize,
    expected_dimensions: (u32, u32),
    expected_bit_depth: u8,
    expected_samples: usize,
) -> Result<(), J2kError> {
    if dimensions != expected_dimensions
        || bit_depth != expected_bit_depth
        || samples < expected_samples
    {
        return Err(J2kError::BackendComponentPlaneTooShort {
            component,
            samples,
            expected: expected_samples,
        });
    }
    Ok(())
}

/// The destination span and sample stride for one channel of consecutive pixels.
pub(super) fn channel_destination<T>(
    out: &mut [T],
    layout: BatchLayout,
    pixels: usize,
    channels: usize,
    channel: usize,
    first_pixel: usize,
    count: usize,
) -> Option<(&mut [T], usize)> {
    let end_pixel = first_pixel.checked_add(count)?;
    if channel >= channels || end_pixel > pixels {
        return None;
    }
    let planar = layout == BatchLayout::Nchw || channels == 1;
    let stride = if planar { 1 } else { channels };
    let start = if planar {
        channel.checked_mul(pixels)?.checked_add(first_pixel)?
    } else {
        first_pixel.checked_mul(channels)?.checked_add(channel)?
    };
    let len = if count == 0 {
        0
    } else {
        (count - 1).checked_mul(stride)?.checked_add(1)?
    };
    Some((out.get_mut(start..start.checked_add(len)?)?, stride))
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
/// start of `destination`. It rounds and saturates in `f32` as native
/// packing does (see [`j2k_native::round_clamped_native`]), clamped to the
/// precision's maximum and a byte, which gives the same bytes: `f32::round`
/// is exact and, above zero, equals the `f64` floor of the sample plus one
/// half, and any value past the precision's maximum saturates. Per sample,
/// `convert_u8` costs a call and `f64` math; over a row, the compiler can
/// vectorize the contiguous rows of planar output, and interleaved rows
/// convert in a tight loop, both at the CPU's SIMD level (see
/// [`j2k_native::at_simd_level`]).
#[expect(
    clippy::cast_possible_truncation,
    reason = "the sample is rounded and clamped to a byte before the cast"
)]
fn convert_u8_row(destination: &mut [u8], stride: usize, samples: &[f32], precision: u8) {
    #[expect(
        clippy::cast_precision_loss,
        reason = "a 16-bit precision's maximum is within f32's exact integer range"
    )]
    let ceiling = (((1_u32 << u32::from(precision)) - 1) as f32).min(f32::from(u8::MAX));
    j2k_native::at_simd_level(
        #[inline(always)]
        || {
            if stride == 1 {
                for (destination, &sample) in destination.iter_mut().zip(samples) {
                    *destination = j2k_native::round_clamped_native(sample, 0.0, ceiling) as u8;
                }
            } else {
                for (destination, &sample) in destination.iter_mut().step_by(stride).zip(samples) {
                    *destination = j2k_native::round_clamped_native(sample, 0.0, ceiling) as u8;
                }
            }
        },
    );
}

pub(super) fn convert_u16(sample: f32, precision: u8) -> u16 {
    u16::try_from(round_unsigned(sample, precision)).unwrap_or(u16::MAX)
}

/// [`convert_u16`] for a row of `samples`, written `stride` apart from the
/// start of `destination`: the planned route's conversion, sample by sample.
fn convert_u16_row(destination: &mut [u16], stride: usize, samples: &[f32], precision: u8) {
    for (destination, &sample) in destination.iter_mut().step_by(stride).zip(samples) {
        *destination = convert_u16(sample, precision);
    }
}

/// [`round_signed`] for a row of `samples`, written `stride` apart from the
/// start of `destination`: the planned route's conversion, sample by sample.
fn convert_i16_row(destination: &mut [i16], stride: usize, samples: &[f32], precision: u8) {
    for (destination, &sample) in destination.iter_mut().step_by(stride).zip(samples) {
        *destination = round_signed(sample, precision);
    }
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
    use super::{convert_u8, convert_u8_row, validate_sample_plane};
    use crate::J2kError;
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

    #[test]
    fn sample_plane_validation_rejects_precision_drift() {
        let error = validate_sample_plane(1, (8, 8), 10, 64, (8, 8), 12, 64)
            .expect_err("prepared precision must match decoded plane");
        assert!(matches!(
            error,
            J2kError::BackendComponentPlaneTooShort {
                component: 1,
                samples: 64,
                expected: 64,
            }
        ));
    }
}
