// SPDX-License-Identifier: MIT OR Apache-2.0

//! libjpeg-turbo DCT scaling for 12-bit sequential and progressive images.
//!
//! Each component is inverse-transformed with the reduced IDCT libjpeg-turbo
//! picks for it ([`ScaledSampling`]) into a plane at its scaled resolution;
//! output rows then upsample every plane with libjpeg-turbo's upsampler for
//! that component and convert the color. Full-size decodes take this route
//! only when libjpeg-turbo replicates a narrow 2:1 component instead of
//! smoothing it; every other full-size decode keeps the direct writers.

use alloc::vec::Vec;

use super::super::{
    checked_scratch_len, decode_block_with_activity, decode_progressive_dct_blocks, finish_scan,
    merged_warnings, scaled_rect_covering, try_clone_warnings, Backend, BitReader, BlockActivity,
    CoefficientBlock, ColorSpace, DecodeOutcome, Decoder, DownscaleFactor, JpegError, Rect,
    SofKind,
};
use super::planes::{dequantize_progressive12_block, ensure_progressive12_coefficient_capacities};
use super::state::Extended12RestartTracker;
use super::writers::{Extended12Output, Extended12RgbProjection};
use crate::allocation::try_reserve_for_len_with_live_budget;
use crate::color::scaled_sampling::{ScaledComponent, ScaledSampling, Upsample};
use crate::idct::downscale::{idct_islow_12bit_1x1, idct_islow_12bit_2x2, idct_islow_12bit_4x4};

/// How 12-bit component samples become output pixels.
#[derive(Clone, Copy, Debug)]
pub(super) enum Extended12Color {
    /// One component written as `Gray16`, or as `Rgb16` with equal channels.
    Gray(Extended12Output),
    /// Three components written as `Rgb16`.
    Rgb(Extended12RgbProjection),
    /// Inverted CMYK or YCCK written as `Rgb16`.
    FourComponent(ColorSpace),
}

impl Extended12Color {
    fn components(self) -> usize {
        match self {
            Self::Gray(_) => 1,
            Self::Rgb(_) => 3,
            Self::FourComponent(_) => 4,
        }
    }
}

/// Whether a 12-bit decode needs libjpeg-turbo's scaled geometry: any reduced
/// scale, or a full-size 2:1 component that libjpeg-turbo replicates because
/// it is at most two samples wide.
pub(super) fn needs_scaled_extended12_route(
    scaled: &ScaledSampling,
    downscale: DownscaleFactor,
) -> bool {
    downscale != DownscaleFactor::Full
        || (0..scaled.effective.len()).any(|index| {
            let component = scaled.component(index);
            component.upsample == Upsample::Replicate && component.h_ratio == 2
        })
}

/// A component plane at its scaled resolution, padded to whole blocks.
struct ScaledPlane {
    samples: Vec<u16>,
    stride: usize,
    component: ScaledComponent,
}

impl ScaledPlane {
    fn row(&self, y: usize) -> &[u16] {
        let y = y.min(self.component.height.saturating_sub(1) as usize);
        let start = y * self.stride;
        &self.samples[start..start + self.component.width as usize]
    }

    fn deposit(&mut self, x: usize, y: usize, size: usize, block: &[u16]) {
        for row in 0..size {
            let dst = (y + row) * self.stride + x;
            self.samples[dst..dst + size].copy_from_slice(&block[row * size..(row + 1) * size]);
        }
    }
}

/// At most four component planes, held inline so only sample storage is
/// charged to the decode budget.
struct ScaledPlanes {
    planes: [ScaledPlane; 4],
    count: usize,
}

impl ScaledPlanes {
    fn as_slice(&self) -> &[ScaledPlane] {
        &self.planes[..self.count]
    }

    fn get_mut(&mut self, index: usize) -> Option<&mut ScaledPlane> {
        self.planes[..self.count].get_mut(index)
    }

    fn sample_bytes(&self) -> Result<usize, JpegError> {
        self.as_slice().iter().try_fold(0usize, |total, plane| {
            let bytes = checked_scratch_len(&[plane.samples.capacity(), 2])?;
            checked_scratch_len(&[1, total]).map(|total| total.saturating_add(bytes))
        })
    }
}

/// Allocates one plane per component, `blocks[i]` (columns, rows) blocks of
/// the component's IDCT size, charging `cap`.
fn allocate_scaled_planes(
    scaled: &ScaledSampling,
    blocks: &[(usize, usize)],
    initial_live_bytes: usize,
    cap: usize,
) -> Result<ScaledPlanes, JpegError> {
    let mut live_bytes = initial_live_bytes;
    let mut planes: [ScaledPlane; 4] = core::array::from_fn(|index| ScaledPlane {
        samples: Vec::new(),
        stride: 0,
        component: scaled.component(index.min(blocks.len().saturating_sub(1))),
    });
    for (index, &(cols, rows)) in blocks.iter().enumerate() {
        let component = scaled.component(index);
        let size = component.idct_size as usize;
        let stride = checked_scratch_len(&[cols, size])?;
        let len = checked_scratch_len(&[stride, rows, size])?;
        let mut samples = Vec::new();
        try_reserve_for_len_with_live_budget(&mut samples, len, &mut live_bytes, cap)?;
        samples.resize(len, 0);
        planes[index] = ScaledPlane {
            samples,
            stride,
            component,
        };
    }
    Ok(ScaledPlanes {
        planes,
        count: blocks.len(),
    })
}

/// Reduced IDCT of one dequantized block into `pixels` (`size * size`
/// samples); `dc_only` lets the full-size transform take its DC shortcut.
fn idct_12bit_block(
    backend: Backend,
    coefficients: &[i16; 64],
    dc_only: bool,
    size: usize,
    pixels: &mut [u16; 64],
) {
    match size {
        8 if dc_only => pixels.fill(crate::idct::idct_islow_12bit_dc_only_sample(
            coefficients[0],
        )),
        8 => backend.idct_12bit(coefficients, pixels),
        4 => {
            let mut block = [0u16; 16];
            idct_islow_12bit_4x4(coefficients, &mut block);
            pixels[..16].copy_from_slice(&block);
        }
        2 => {
            let mut block = [0u16; 4];
            idct_islow_12bit_2x2(coefficients, &mut block);
            pixels[..4].copy_from_slice(&block);
        }
        _ => pixels[0] = idct_islow_12bit_1x1(coefficients),
    }
}

impl Decoder<'_> {
    /// Decodes `roi` of a 12-bit image at `downscale` with libjpeg-turbo's
    /// per-component reduced IDCTs and upsamplers.
    pub(super) fn decode_extended12_scaled_region_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        downscale: DownscaleFactor,
        color: Extended12Color,
    ) -> Result<DecodeOutcome, JpegError> {
        let output_rect = scaled_rect_covering(roi, downscale)?;
        let (planes, warnings, cap) = if self.info.sof_kind == SofKind::Progressive12 {
            let planes = self.render_progressive12_scaled_planes(downscale, color)?;
            let cap = self
                .progressive_plan
                .as_ref()
                .map_or(0, |plan| plan.scratch_bytes);
            (planes, try_clone_warnings(&self.warnings)?, cap)
        } else {
            let (planes, scan_warnings) = self.decode_extended12_scaled_planes(downscale, color)?;
            let warnings = merged_warnings(&self.warnings, scan_warnings)?;
            (planes, warnings, self.plan.scratch_bytes)
        };
        let live_bytes = planes.sample_bytes()?;
        let write = ScaledWrite {
            output_rect,
            output_width: self.info.dimensions.0.div_ceil(downscale.denominator()) as usize,
            live_bytes,
            cap,
        };
        write_scaled_planes_region(out, stride, write, planes.as_slice(), color)?;
        Ok(DecodeOutcome {
            decoded: roi,
            warnings,
        })
    }

    fn decode_extended12_scaled_planes(
        &self,
        downscale: DownscaleFactor,
        color: Extended12Color,
    ) -> Result<(ScaledPlanes, Vec<crate::error::Warning>), JpegError> {
        let plan = &self.plan;
        let sof = self.info.sof_kind;
        if plan.components.len() != color.components() {
            return Err(JpegError::NotImplemented { sof });
        }
        let scaled = ScaledSampling::new(plan.sampling, downscale, plan.dimensions);
        let max_h = u32::from(plan.sampling.max_h);
        let max_v = u32::from(plan.sampling.max_v);
        let mcu_cols = plan.dimensions.0.div_ceil(max_h * 8);
        let mcu_rows = plan.dimensions.1.div_ceil(max_v * 8);
        let blocks: Vec<(usize, usize)> = plan
            .sampling
            .iter()
            .map(|(h, v)| {
                (
                    mcu_cols as usize * usize::from(h),
                    mcu_rows as usize * usize::from(v),
                )
            })
            .collect();
        let mut planes = allocate_scaled_planes(&scaled, &blocks, 0, plan.scratch_bytes)?;

        let mut br = BitReader::new(&self.bytes[plan.scan_offset..]);
        let mut prev_dc = [0i32; 4];
        let mut coeff = CoefficientBlock::default();
        let mut pixels = [0u16; 64];
        let mut restart_tracker =
            Extended12RestartTracker::new(plan.restart_interval, mcu_cols * mcu_rows);
        for mcu_y in 0..mcu_rows {
            for mcu_x in 0..mcu_cols {
                if restart_tracker.begin_mcu(&mut br, mcu_y * mcu_cols + mcu_x)? {
                    prev_dc.fill(0);
                }
                for component in &plan.components {
                    let index = component.output_index;
                    let resolved = plan.resolve_component(component)?;
                    let plane = planes
                        .get_mut(index)
                        .ok_or(JpegError::NotImplemented { sof })?;
                    let size = plane.component.idct_size as usize;
                    let (h, v) = (u32::from(component.h), u32::from(component.v));
                    for by in 0..v {
                        for bx in 0..h {
                            let activity = decode_block_with_activity(
                                &mut br,
                                resolved.dc_table,
                                resolved.ac_table,
                                &mut prev_dc[index],
                                resolved.quant,
                                &mut coeff,
                            )?;
                            idct_12bit_block(
                                self.backend,
                                coeff.coefficients(),
                                activity == BlockActivity::DcOnly,
                                size,
                                &mut pixels,
                            );
                            plane.deposit(
                                (mcu_x * h + bx) as usize * size,
                                (mcu_y * v + by) as usize * size,
                                size,
                                &pixels,
                            );
                        }
                    }
                }
                restart_tracker.finish_mcu();
            }
        }
        let scan_warnings = finish_scan(&mut br, true)?;
        Ok((planes, scan_warnings))
    }

    fn render_progressive12_scaled_planes(
        &self,
        downscale: DownscaleFactor,
        color: Extended12Color,
    ) -> Result<ScaledPlanes, JpegError> {
        let sof = self.info.sof_kind;
        let plan = self
            .progressive_plan
            .as_ref()
            .ok_or(JpegError::NotImplemented { sof })?;
        if plan.components.len() != color.components() {
            return Err(JpegError::NotImplemented { sof });
        }
        let scaled = ScaledSampling::new(plan.sampling, downscale, plan.dimensions);
        let dct_blocks = decode_progressive_dct_blocks(plan, self.bytes, 0)?;
        ensure_progressive12_coefficient_capacities(&dct_blocks, plan.scratch_bytes)?;
        let mut blocks = [(0usize, 0usize); 4];
        for component in &plan.components {
            let slot = blocks
                .get_mut(component.output_index)
                .ok_or(JpegError::NotImplemented { sof })?;
            *slot = (component.block_cols as usize, component.block_rows as usize);
        }
        let mut planes = allocate_scaled_planes(
            &scaled,
            &blocks[..plan.components.len()],
            dct_blocks.capacity_bytes()?,
            plan.scratch_bytes,
        )?;
        let mut dequant = [0i16; 64];
        let mut pixels = [0u16; 64];
        for (component, coeffs) in plan.components.iter().zip(&dct_blocks.quantized) {
            let plane = planes
                .get_mut(component.output_index)
                .ok_or(JpegError::NotImplemented { sof })?;
            let size = plane.component.idct_size as usize;
            let cols = component.block_cols as usize;
            for (block_index, block) in coeffs.iter().enumerate() {
                dequantize_progressive12_block(block, &component.quant, &mut dequant);
                let dc_only = dequant[1..].iter().all(|&coefficient| coefficient == 0);
                idct_12bit_block(self.backend, &dequant, dc_only, size, &mut pixels);
                plane.deposit(
                    (block_index % cols) * size,
                    (block_index / cols) * size,
                    size,
                    &pixels,
                );
            }
        }
        Ok(planes)
    }
}

/// One component's output row `y`, upsampled across the full scaled width.
fn upsample_row(plane: &ScaledPlane, y: usize, out: &mut [u16]) {
    let ScaledComponent {
        h_ratio,
        v_ratio,
        upsample,
        ..
    } = plane.component;
    match upsample {
        Upsample::None => out.copy_from_slice(&plane.row(y)[..out.len()]),
        Upsample::FancyH2V1 => {
            let row = plane.row(y);
            for (x, dst) in out.iter_mut().enumerate() {
                *dst = super::upsample::upsample_extended12_h2v1_at(row, x);
            }
        }
        Upsample::FancyH1V2 => {
            // h1v2_fancy_upsample: 3/4 nearer row + 1/4 further row, biased
            // +1 for the upper output row and +2 for the lower.
            let sample_y = y / 2;
            let curr = plane.row(sample_y);
            let (near, bias) = if y.is_multiple_of(2) {
                (plane.row(sample_y.saturating_sub(1)), 1)
            } else {
                (plane.row(sample_y + 1), 2)
            };
            for (x, dst) in out.iter_mut().enumerate() {
                let sum = 3 * u32::from(curr[x]) + u32::from(near[x]) + bias;
                *dst = u16::try_from(sum >> 2).expect("weighted mean of 12-bit samples fits u16");
            }
        }
        Upsample::FancyH2V2 => {
            let sample_y = y / 2;
            let prev = plane.row(sample_y.saturating_sub(1));
            let curr = plane.row(sample_y);
            let next = plane.row(sample_y + 1);
            for (x, dst) in out.iter_mut().enumerate() {
                *dst = super::upsample::upsample_h2v2_u16_rows_at(prev, curr, next, x, y % 2 == 1);
            }
        }
        Upsample::Replicate => {
            let row = plane.row(y / v_ratio as usize);
            let h_ratio = h_ratio as usize;
            for (x, dst) in out.iter_mut().enumerate() {
                *dst = row[(x / h_ratio).min(row.len() - 1)];
            }
        }
    }
}

/// Output geometry and memory budget for writing scaled planes.
#[derive(Clone, Copy)]
struct ScaledWrite {
    output_rect: Rect,
    /// Width of the scaled image; every upsampled row spans it.
    output_width: usize,
    /// Bytes already live (the planes) and the cap they share with the rows.
    live_bytes: usize,
    cap: usize,
}

fn write_scaled_planes_region(
    out: &mut [u8],
    stride: usize,
    write: ScaledWrite,
    planes: &[ScaledPlane],
    color: Extended12Color,
) -> Result<(), JpegError> {
    let ScaledWrite {
        output_rect,
        output_width,
        mut live_bytes,
        cap,
    } = write;
    let x0 = output_rect.x as usize;
    let x1 = x0 + output_rect.w as usize;
    let mut rows: [Vec<u16>; 4] = Default::default();
    for row in &mut rows[..planes.len()] {
        try_reserve_for_len_with_live_budget(row, output_width, &mut live_bytes, cap)?;
        row.resize(output_width, 0u16);
    }
    for y in output_rect.y..output_rect.y + output_rect.h {
        for (plane, row) in planes.iter().zip(rows.iter_mut()) {
            upsample_row(plane, y as usize, row);
        }
        let dst_row = &mut out[(y - output_rect.y) as usize * stride..];
        for (dst_x, x) in (x0..x1).enumerate() {
            match color {
                Extended12Color::Gray(Extended12Output::Gray16) => {
                    dst_row[dst_x * 2..dst_x * 2 + 2].copy_from_slice(&rows[0][x].to_le_bytes());
                }
                Extended12Color::Gray(Extended12Output::Rgb16) => {
                    let sample = rows[0][x];
                    write_rgb16_le(
                        &mut dst_row[dst_x * 6..dst_x * 6 + 6],
                        (sample, sample, sample),
                    );
                }
                Extended12Color::Rgb(projection) => {
                    let (r, g, b) = match projection {
                        Extended12RgbProjection::Identity => (rows[0][x], rows[1][x], rows[2][x]),
                        Extended12RgbProjection::YCbCr => crate::color::ycbcr::ycbcr12_to_rgb16(
                            rows[0][x], rows[1][x], rows[2][x],
                        ),
                    };
                    write_rgb16_le(&mut dst_row[dst_x * 6..dst_x * 6 + 6], (r, g, b));
                }
                Extended12Color::FourComponent(color_space) => {
                    let samples = (rows[0][x], rows[1][x], rows[2][x], rows[3][x]);
                    let rgb = if color_space == ColorSpace::Cmyk {
                        crate::color::cmyk::inverted_cmyk12_to_rgb16(
                            samples.0, samples.1, samples.2, samples.3,
                        )
                    } else {
                        crate::color::cmyk::ycck12_to_rgb16(
                            samples.0, samples.1, samples.2, samples.3,
                        )
                    };
                    write_rgb16_le(&mut dst_row[dst_x * 6..dst_x * 6 + 6], rgb);
                }
            }
        }
    }
    Ok(())
}

fn write_rgb16_le(dst: &mut [u8], (r, g, b): (u16, u16, u16)) {
    dst[0..2].copy_from_slice(&r.to_le_bytes());
    dst[2..4].copy_from_slice(&g.to_le_bytes());
    dst[4..6].copy_from_slice(&b.to_le_bytes());
}
