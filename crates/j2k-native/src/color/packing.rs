// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared 8-bit sample conversion and interleaved packing policy.

use crate::error::{bail, ColorError};
use crate::j2c::mct::DeferredMct;
use crate::j2c::ComponentData;
use crate::jp2::DecodedImage;
use crate::math;
use crate::{DecodingError, Result, ValidationError};

pub(crate) fn validate_interleaved_output_buffer(
    image: &DecodedImage<'_, '_>,
    buf: &[u8],
) -> Result<()> {
    let required_len = interleaved_output_len(image)?;
    if buf.len() < required_len {
        bail!(DecodingError::OutputBufferTooSmall);
    }
    Ok(())
}

fn interleaved_output_len(image: &DecodedImage<'_, '_>) -> Result<usize> {
    let Some(first) = image.decoded_components.first() else {
        bail!(DecodingError::CodeBlockDecodeFailure);
    };
    first
        .container
        .truncated()
        .len()
        .checked_mul(image.decoded_components.len())
        .ok_or(ValidationError::ImageTooLarge.into())
}

#[derive(Clone, Copy)]
struct SampleConversionPolicy {
    uniform_bit_depth: Option<u8>,
}

impl SampleConversionPolicy {
    fn for_components(components: &[ComponentData]) -> Result<Self> {
        let Some(first) = components.first() else {
            bail!(DecodingError::CodeBlockDecodeFailure);
        };
        let uniform_bit_depth = components
            .iter()
            .all(|component| component.bit_depth == first.bit_depth)
            .then_some(first.bit_depth);
        Ok(Self { uniform_bit_depth })
    }

    const fn uses_direct_8_bit(self) -> bool {
        matches!(self.uniform_bit_depth, Some(8))
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "pixel samples are rounded and intentionally quantized to the stable 8-bit output format"
    )]
    fn quantize(self, component: &ComponentData, sample: f32) -> u8 {
        if self.uses_direct_8_bit() {
            return round_to_u8(sample);
        }
        let source_max = ((1_u64 << u32::from(component.bit_depth)) - 1) as f32;
        math::round_f32((sample / source_max) * f32::from(u8::MAX)) as u8
    }
}

#[derive(Clone, Copy)]
struct SampleWindow {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    row_stride: usize,
}

impl SampleWindow {
    const fn full(sample_count: usize) -> Self {
        Self {
            x: 0,
            y: 0,
            width: sample_count,
            height: 1,
            row_stride: sample_count,
        }
    }

    fn region(image_width: usize, roi: (u32, u32, u32, u32)) -> Result<Self> {
        let (x, y, width, height) = roi;
        let x = x as usize;
        let y = y as usize;
        let width = width as usize;
        let height = height as usize;
        let x_end = x.checked_add(width).ok_or(ValidationError::ImageTooLarge)?;
        if x_end > image_width {
            bail!(DecodingError::CodeBlockDecodeFailure);
        }
        Ok(Self {
            x,
            y,
            width,
            height,
            row_stride: image_width,
        })
    }

    fn required_output_len(self, channels: usize) -> Result<usize> {
        self.width
            .checked_mul(self.height)
            .and_then(|samples| samples.checked_mul(channels))
            .ok_or(ValidationError::ImageTooLarge.into())
    }

    fn validate(self, components: &[ComponentData], output_len: usize) -> Result<()> {
        if output_len < self.required_output_len(components.len())? {
            bail!(DecodingError::OutputBufferTooSmall);
        }
        let row_end = self
            .y
            .checked_add(self.height)
            .ok_or(ValidationError::ImageTooLarge)?;
        let required_samples = if self.width == 0 || self.height == 0 {
            0
        } else {
            row_end
                .checked_sub(1)
                .and_then(|row| row.checked_mul(self.row_stride))
                .and_then(|base| base.checked_add(self.x))
                .and_then(|start| start.checked_add(self.width))
                .ok_or(ValidationError::ImageTooLarge)?
        };
        if components
            .iter()
            .any(|component| component.container.truncated().len() < required_samples)
        {
            bail!(DecodingError::CodeBlockDecodeFailure);
        }
        Ok(())
    }
}

fn interleave_window(
    components: &[ComponentData],
    policy: SampleConversionPolicy,
    window: SampleWindow,
    buf: &mut [u8],
) -> Result<()> {
    window.validate(components, buf.len())?;
    let mut output = buf.iter_mut();
    let row_end = window
        .y
        .checked_add(window.height)
        .ok_or(ValidationError::ImageTooLarge)?;
    let column_end = window
        .x
        .checked_add(window.width)
        .ok_or(ValidationError::ImageTooLarge)?;
    for row in window.y..row_end {
        let row_base = row
            .checked_mul(window.row_stride)
            .ok_or(ValidationError::ImageTooLarge)?;
        for column in window.x..column_end {
            let index = row_base
                .checked_add(column)
                .ok_or(ValidationError::ImageTooLarge)?;
            for component in components {
                let destination = output.next().ok_or(DecodingError::OutputBufferTooSmall)?;
                *destination = policy.quantize(component, component.container[index]);
            }
        }
    }
    Ok(())
}

/// [`interleave_and_convert`] after a decode that left its inverse MCT to
/// the pack: 8-bit RGB fuses it with the pack, other outputs apply it first.
pub(crate) fn interleave_and_convert_after_mct(
    image: &mut DecodedImage<'_, '_>,
    buf: &mut [u8],
    mct: Option<DeferredMct>,
    parallel: bool,
) -> Result<()> {
    let Some(mct) = mct else {
        return interleave_and_convert(image, buf);
    };
    let components = &mut *image.decoded_components;
    let policy = SampleConversionPolicy::for_components(components)?;
    let fused = components.len() == 3 && policy.uses_direct_8_bit();
    let len = components[0].container.truncated().len();
    if fused {
        SampleWindow::full(len).validate(components, buf.len())?;
    }
    let [s0, s1, s2, ..] = components.as_mut_slice() else {
        bail!(ColorError::Mct);
    };
    if fused {
        let planes = [s0, s1, s2].map(|component| &mut component.container[..len]);
        mct.pack_rgb8(planes, &mut buf[..3 * len], parallel, pack_rgb8);
        return Ok(());
    }
    mct.apply(
        (
            &mut s0.container[..],
            &mut s1.container[..],
            &mut s2.container[..],
        ),
        parallel,
    );
    interleave_and_convert(image, buf)
}

pub(crate) fn interleave_and_convert(
    image: &mut DecodedImage<'_, '_>,
    buf: &mut [u8],
) -> Result<()> {
    let components = &mut *image.decoded_components;
    let num_components = components.len();
    let policy = SampleConversionPolicy::for_components(components)?;
    let sample_count = components[0].container.truncated().len();
    let window = SampleWindow::full(sample_count);
    window.validate(components, buf.len())?;

    let mut output_iter = buf.iter_mut();

    if policy.uses_direct_8_bit() && num_components <= 4 {
        // Fast path for the common case.
        match num_components {
            // Gray-scale.
            1 => {
                for (output, input) in output_iter.zip(
                    components[0]
                        .container
                        .iter()
                        .map(|sample| policy.quantize(&components[0], *sample)),
                ) {
                    *output = input;
                }
            }
            // Gray-scale with alpha.
            2 => {
                let c0 = &components[0];
                let c1 = &components[1];

                let c0 = &c0.container[..sample_count];
                let c1 = &c1.container[..sample_count];

                for i in 0..sample_count {
                    *output_iter.next().unwrap() = policy.quantize(&components[0], c0[i]);
                    *output_iter.next().unwrap() = policy.quantize(&components[1], c1[i]);
                }
            }
            // RGB
            3 => {
                let c0 = &components[0];
                let c1 = &components[1];
                let c2 = &components[2];

                let c0 = &c0.container[..sample_count];
                let c1 = &c1.container[..sample_count];
                let c2 = &c2.container[..sample_count];

                pack_rgb8([c0, c1, c2], buf);
            }
            // RGBA or CMYK.
            4 => {
                let c0 = &components[0];
                let c1 = &components[1];
                let c2 = &components[2];
                let c3 = &components[3];

                let c0 = &c0.container[..sample_count];
                let c1 = &c1.container[..sample_count];
                let c2 = &c2.container[..sample_count];
                let c3 = &c3.container[..sample_count];

                for i in 0..sample_count {
                    *output_iter.next().unwrap() = policy.quantize(&components[0], c0[i]);
                    *output_iter.next().unwrap() = policy.quantize(&components[1], c1[i]);
                    *output_iter.next().unwrap() = policy.quantize(&components[2], c2[i]);
                    *output_iter.next().unwrap() = policy.quantize(&components[3], c3[i]);
                }
            }
            _ => bail!(ValidationError::TooManyChannels),
        }
    } else {
        interleave_window(components, policy, window, buf)?;
    }

    Ok(())
}

/// Pack 8-bit RGB planes as interleaved bytes, rounded and saturated.
fn pack_rgb8(planes: [&[f32]; 3], output: &mut [u8]) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        use fearless_simd::{dispatch, Level};
        dispatch!(Level::new(), simd => pack_rgb8_simd(simd, planes, output));
    }
    // An indexed loop over whole pixels, which LLVM vectorizes on NEON.
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    {
        let [r, g, b] = planes;
        let pixels = output.as_chunks_mut::<3>().0;
        let len = pixels.len().min(r.len()).min(g.len()).min(b.len());
        for i in 0..len {
            pixels[i] = [round_to_u8(r[i]), round_to_u8(g[i]), round_to_u8(b[i])];
        }
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "8-bit samples are rounded and saturated to the stable 8-bit output format"
)]
fn round_to_u8(sample: f32) -> u8 {
    math::round_f32(sample) as u8
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline(always)]
fn pack_rgb8_simd<S: fearless_simd::Simd>(simd: S, planes: [&[f32]; 3], output: &mut [u8]) {
    use fearless_simd::{f32x8, u32x8, u8x16, Bytes, Select, SimdBase, SimdFloat};

    let zero = f32x8::splat(simd, 0.0);
    let max = f32x8::splat(simd, 255.0);
    let half = f32x8::splat(simd, 0.5);
    let one = f32x8::splat(simd, 1.0);
    let indices = u8x16::from_slice(simd, &[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 0, 0, 0, 0]);
    let vector_len = planes[0].len() / 8 * 8;
    for start in (0..vector_len).step_by(8) {
        let channels = planes.map(|plane| {
            let value = f32x8::from_slice(simd, &plane[start..start + 8]);
            // Match Rust's saturating float-to-u8 cast, including NaN -> 0.
            let value = value.simd_ge(zero).select(value, zero).min(max);
            let integral = value.floor();
            // Adding 0.5 before rounding would incorrectly round the float
            // immediately below 0.5 up to 1. Compare the fractional part.
            let rounded = integral + (value - integral).simd_ge(half).select(one, zero);
            simd.cvt_u32_f32x8(rounded)
        });
        let packed: u32x8<S> = channels[0] | (channels[1] << 8) | (channels[2] << 16);
        let (first, second) = simd.split_u32x8(packed);
        for (offset, words) in [(0, first), (12, second)] {
            let bytes = words.bitcast::<u8x16<S>>().swizzle_dyn(indices).to_array();
            output[start * 3 + offset..start * 3 + offset + 12].copy_from_slice(&bytes[..12]);
        }
    }
    for index in vector_len..planes[0].len() {
        for channel in 0..3 {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "pixel samples are rounded and saturated to the 8-bit output format"
            )]
            {
                output[index * 3 + channel] = math::round_f32(planes[channel][index]) as u8;
            }
        }
    }
}

pub(crate) fn interleave_and_convert_region(
    image: &mut DecodedImage<'_, '_>,
    image_width: usize,
    roi: (u32, u32, u32, u32),
    buf: &mut [u8],
) -> Result<()> {
    let components = &mut *image.decoded_components;
    let policy = SampleConversionPolicy::for_components(components)?;
    interleave_window(
        components,
        policy,
        SampleWindow::region(image_width, roi)?,
        buf,
    )
}
