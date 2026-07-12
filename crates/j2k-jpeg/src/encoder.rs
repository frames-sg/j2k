// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::borrow::Cow;
use alloc::vec::Vec;
use core::f64::consts::PI;

use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;
use thiserror::Error;

use crate::adapter::{
    assemble_jpeg_baseline_frame, baseline_encode_tables, checked_cpu_encode_live_bytes,
    checked_encode_host_live_bytes, cpu_owned_plane_capacity_limit,
    jpeg_baseline_entropy_capacity_bytes, validate_jpeg_baseline_dimensions,
    validate_jpeg_baseline_restart_interval, JpegBaselineHuffmanTable, JpegBaselineSampling,
    JPEG_BASELINE_ZIGZAG,
};
use crate::encoded_output::{checked_jpeg_baseline_frame_capacity, CappedBytes};
use crate::profile::{emit_jpeg_profile_fields, jpeg_profile_stages_enabled, ProfileField};
use std::time::{Duration, Instant};

mod entropy;
use self::entropy::{encode_entropy, entropy_host_workspace_bytes};
#[cfg(test)]
use self::entropy::{
    encode_entropy_restart_segments, encode_entropy_serial, parallel_entropy_chunk_count,
    MAX_PARALLEL_ENTROPY_CHUNKS,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// JPEG encoder backend selector.
pub enum JpegBackend {
    /// Choose the best available backend for the platform.
    Auto,
    /// Use the portable CPU encoder.
    Cpu,
    /// Use a Metal encoder when called through the Metal integration.
    Metal,
    /// Use a CUDA encoder when called through the CUDA integration.
    Cuda,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// JPEG baseline chroma subsampling mode.
pub enum JpegSubsampling {
    /// Single-component grayscale.
    Gray,
    /// Three-component YBR/RGB 4:4:4 sampling.
    Ybr444,
    /// Three-component YBR/RGB 4:2:2 sampling.
    Ybr422,
    /// Three-component YBR/RGB 4:2:0 sampling.
    Ybr420,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Options controlling baseline JPEG encoding.
pub struct JpegEncodeOptions {
    /// JPEG quality in the conventional 1..=100 range.
    pub quality: u8,
    /// Output component sampling.
    pub subsampling: JpegSubsampling,
    /// Optional restart interval in MCUs.
    pub restart_interval: Option<u16>,
    /// Requested encoder backend.
    pub backend: JpegBackend,
}

impl Default for JpegEncodeOptions {
    fn default() -> Self {
        Self {
            quality: 90,
            subsampling: JpegSubsampling::Ybr422,
            restart_interval: None,
            backend: JpegBackend::Auto,
        }
    }
}

#[derive(Debug, Clone, Copy)]
/// Borrowed input samples for baseline JPEG encoding.
pub enum JpegSamples<'a> {
    /// Interleaved 8-bit grayscale samples.
    Gray8 {
        /// Pixel data, one byte per pixel.
        data: &'a [u8],
        /// Image width in pixels.
        width: u32,
        /// Image height in pixels.
        height: u32,
    },
    /// Interleaved 8-bit RGB samples.
    Rgb8 {
        /// Pixel data, three bytes per pixel.
        data: &'a [u8],
        /// Image width in pixels.
        width: u32,
        /// Image height in pixels.
        height: u32,
    },
}

#[derive(Debug, PartialEq, Eq)]
/// Encoded baseline JPEG bytes and the backend that produced them.
///
/// The retained codestream can approach the shared host-allocation cap, so
/// this owner is intentionally move-only rather than exposing infallible
/// full-payload cloning.
pub struct EncodedJpeg {
    /// Complete JPEG codestream.
    pub data: Vec<u8>,
    /// Backend used to encode the codestream.
    pub backend: JpegBackend,
}

#[derive(Clone, Debug, Error)]
/// Errors produced by baseline JPEG encoding.
pub enum JpegEncodeError {
    #[error("JPEG encode requires nonzero dimensions")]
    /// Width or height was zero.
    EmptyDimensions,
    #[error("JPEG baseline dimensions must fit in u16, got {width}x{height}")]
    /// JPEG baseline SOF dimensions exceed the 16-bit marker fields.
    DimensionsTooLarge {
        /// Requested width in pixels.
        width: u32,
        /// Requested height in pixels.
        height: u32,
    },
    #[error("JPEG sample buffer length mismatch: expected {expected}, got {actual}")]
    /// Input sample buffer length does not match width, height, and format.
    SampleLength {
        /// Required byte count.
        expected: usize,
        /// Supplied byte count.
        actual: usize,
    },
    #[error("JPEG host buffer requires {requested} bytes, exceeding the {cap}-byte cap")]
    /// A sample layout or encoded output exceeds the shared host allocation cap.
    MemoryCapExceeded {
        /// Required byte count, saturated when arithmetic overflows.
        requested: usize,
        /// Maximum accepted host allocation size.
        cap: usize,
    },
    #[error("JPEG host allocation failed for {bytes} bytes")]
    /// A required host buffer could not reserve its capacity.
    HostAllocationFailed {
        /// Requested allocation size in bytes.
        bytes: usize,
    },
    #[error("JPEG subsampling {subsampling:?} is incompatible with {samples}")]
    /// Requested subsampling is incompatible with the supplied sample format.
    IncompatibleSubsampling {
        /// Requested output sampling.
        subsampling: JpegSubsampling,
        /// Human-readable sample format name.
        samples: &'static str,
    },
    #[error("JPEG restart interval must be nonzero when provided")]
    /// Restart interval was explicitly set to zero.
    InvalidRestartInterval,
    #[error("JPEG encode backend {backend:?} is unavailable in j2k-jpeg CPU crate")]
    /// Requested backend is not available in this crate.
    UnsupportedBackend {
        /// Requested backend.
        backend: JpegBackend,
    },
    #[error("JPEG encoded marker segment is too large: {name}")]
    /// A marker segment would exceed the JPEG 16-bit length field.
    SegmentTooLarge {
        /// Marker segment name.
        name: &'static str,
    },
    #[error("JPEG entropy symbol has no Huffman code: {symbol}")]
    /// Encoder attempted to emit a symbol absent from the active Huffman table.
    MissingHuffmanCode {
        /// Missing entropy symbol.
        symbol: u8,
    },
    #[error("invalid JPEG DCT image: {reason}")]
    /// Caller-supplied coefficient-domain input cannot be re-emitted as baseline JPEG.
    InvalidDctImage {
        /// Typed invalid-input reason.
        #[source]
        reason: crate::transcode::JpegDctImageError,
    },
    #[error("JPEG encode internal invariant failed: {reason}")]
    /// A heap-free diagnostic for an impossible encoder state.
    InternalInvariant {
        /// Static invariant description.
        reason: &'static str,
    },
}

pub(crate) struct BitWriter {
    bytes: CappedBytes,
    current: u8,
    used: u8,
}

#[derive(Default)]
struct JpegEncodeProfile {
    validation: Duration,
    setup: Duration,
    planes: Duration,
    header: Duration,
    entropy: Duration,
}

struct CpuEncodeCapacityPlan {
    entropy_capacity: usize,
    entropy_workspace_bytes: usize,
    plane_capacity_limit: usize,
}

impl BitWriter {
    pub(crate) fn try_with_max_bytes(max_bytes: usize) -> Result<Self, JpegEncodeError> {
        Ok(Self {
            bytes: CappedBytes::try_with_capacity(max_bytes, max_bytes)?,
            current: 0,
            used: 0,
        })
    }

    fn write_bits(&mut self, code: u32, len: u8) -> Result<(), JpegEncodeError> {
        for bit_idx in (0..len).rev() {
            let bit = u8::from(((code >> bit_idx) & 1) != 0);
            self.current = (self.current << 1) | bit;
            self.used += 1;
            if self.used == 8 {
                self.push_byte(self.current)?;
                self.current = 0;
                self.used = 0;
            }
        }
        Ok(())
    }

    fn align_with_ones(&mut self) -> Result<(), JpegEncodeError> {
        if self.used == 0 {
            return Ok(());
        }
        let remaining = 8 - self.used;
        self.current <<= remaining;
        self.current |= (1u8 << remaining) - 1;
        self.push_byte(self.current)?;
        self.current = 0;
        self.used = 0;
        Ok(())
    }

    pub(crate) fn into_bytes(mut self) -> Result<Vec<u8>, JpegEncodeError> {
        self.align_with_ones()?;
        Ok(self.bytes.into_vec())
    }

    pub(crate) fn capacity_bytes(&self) -> usize {
        self.bytes.capacity()
    }

    fn write_restart_marker(&mut self, marker: u8) -> Result<(), JpegEncodeError> {
        self.align_with_ones()?;
        self.bytes.push(0xFF)?;
        self.bytes.push(marker)
    }

    fn push_byte(&mut self, byte: u8) -> Result<(), JpegEncodeError> {
        self.bytes.push(byte)?;
        if byte == 0xFF {
            self.bytes.push(0x00)?;
        }
        Ok(())
    }
}

/// Encode grayscale or RGB samples as a baseline JPEG codestream.
///
/// # Errors
///
/// Returns an error for invalid dimensions, sample layout, quality, restart
/// configuration, or an unavailable explicitly requested backend.
pub fn encode_jpeg_baseline(
    samples: JpegSamples<'_>,
    options: JpegEncodeOptions,
) -> Result<EncodedJpeg, JpegEncodeError> {
    match options.backend {
        JpegBackend::Auto | JpegBackend::Cpu => encode_jpeg_baseline_cpu(samples, options),
        JpegBackend::Metal | JpegBackend::Cuda => Err(JpegEncodeError::UnsupportedBackend {
            backend: options.backend,
        }),
    }
}

fn encode_jpeg_baseline_cpu(
    samples: JpegSamples<'_>,
    options: JpegEncodeOptions,
) -> Result<EncodedJpeg, JpegEncodeError> {
    let profile_enabled = jpeg_profile_stages_enabled();
    let total_start = profile_enabled.then(Instant::now);
    let mut profile = JpegEncodeProfile::default();

    let validation_start = profile_enabled.then(Instant::now);
    validate_jpeg_baseline_restart_interval(options.restart_interval)?;
    let (width, height) = samples.dimensions();
    let sample_format = samples.name();
    validate_jpeg_baseline_dimensions(width, height)?;
    let expected_sample_len = samples.validate_layout(options.subsampling)?;
    if let Some(start) = validation_start {
        profile.validation = start.elapsed();
    }

    let setup_start = profile_enabled.then(Instant::now);
    let tables = baseline_encode_tables(options)?;
    let sampling = tables.sampling;
    let capacity_plan = checked_cpu_encode_capacity_plan(
        samples,
        sampling,
        expected_sample_len,
        options.restart_interval,
    )?;
    if samples.data_len() != expected_sample_len {
        return Err(JpegEncodeError::SampleLength {
            expected: expected_sample_len,
            actual: samples.data_len(),
        });
    }
    let cosine = cosine_table();
    if let Some(start) = setup_start {
        profile.setup = start.elapsed();
    }

    let planes_start = profile_enabled.then(Instant::now);
    let planes = component_planes(
        samples,
        options.subsampling,
        capacity_plan.plane_capacity_limit,
    )?;
    let plane_live_bytes = component_plane_capacity_bytes(planes.capacity(), &planes)?;
    checked_encode_host_live_bytes([plane_live_bytes, capacity_plan.entropy_workspace_bytes])?;
    if let Some(start) = planes_start {
        profile.planes = start.elapsed();
    }

    let entropy_start = profile_enabled.then(Instant::now);
    let entropy = encode_entropy(
        &planes,
        width,
        height,
        sampling,
        &tables.q_luma,
        &tables.q_chroma,
        [&tables.huff_dc_luma, &tables.huff_dc_chroma],
        [&tables.huff_ac_luma, &tables.huff_ac_chroma],
        &cosine,
        options.restart_interval,
        capacity_plan.entropy_capacity,
        plane_live_bytes,
    )?;
    if let Some(start) = entropy_start {
        profile.entropy = start.elapsed();
    }
    let header_start = profile_enabled.then(Instant::now);
    let frame_capacity = checked_jpeg_baseline_frame_capacity(entropy.len())?;
    checked_encode_host_live_bytes([plane_live_bytes, entropy.capacity(), frame_capacity])?;
    let encoded =
        assemble_jpeg_baseline_frame(&entropy, width, height, &tables, options, JpegBackend::Cpu)?;
    checked_encode_host_live_bytes([
        plane_live_bytes,
        entropy.capacity(),
        encoded.data.capacity(),
    ])?;
    if let Some(start) = header_start {
        profile.header = start.elapsed();
    }
    drop(entropy);
    drop(planes);

    if let Some(start) = total_start {
        emit_cpu_encode_profile(
            start,
            &profile,
            (width, height),
            sample_format,
            options,
            sampling,
            &encoded,
        );
    }

    Ok(encoded)
}

fn emit_cpu_encode_profile(
    start: Instant,
    profile: &JpegEncodeProfile,
    (width, height): (u32, u32),
    sample_format: &str,
    options: JpegEncodeOptions,
    sampling: JpegBaselineSampling,
    encoded: &EncodedJpeg,
) {
    emit_jpeg_profile_fields("jpeg_cpu_encode_fields", "encode", "cpu", || {
        Ok([
            ProfileField::metric_with_summary("sample", sample_format, false)?,
            ProfileField::metric_with_summary("width", width, false)?,
            ProfileField::metric_with_summary("height", height, false)?,
            ProfileField::metric_with_summary("components", sampling.components, false)?,
            ProfileField::metric_with_summary("quality", options.quality, false)?,
            ProfileField::metric_with_summary(
                "subsampling",
                format_args!("{:?}", options.subsampling),
                false,
            )?,
            ProfileField::metric_with_summary(
                "restart_interval",
                options.restart_interval.unwrap_or(0),
                false,
            )?,
            ProfileField::metric("validation_us", profile.validation.as_micros())?,
            ProfileField::metric("setup_us", profile.setup.as_micros())?,
            ProfileField::metric("planes_us", profile.planes.as_micros())?,
            ProfileField::metric("header_us", profile.header.as_micros())?,
            ProfileField::metric("entropy_us", profile.entropy.as_micros())?,
            ProfileField::metric("total_us", start.elapsed().as_micros())?,
            ProfileField::metric_with_summary("output_bytes", encoded.data.len(), false)?,
            ProfileField::metric_with_summary(
                "rayon_threads",
                rayon::current_num_threads(),
                false,
            )?,
        ])
    });
}

impl JpegSamples<'_> {
    fn name(self) -> &'static str {
        match self {
            Self::Gray8 { .. } => "Gray8",
            Self::Rgb8 { .. } => "Rgb8",
        }
    }

    fn dimensions(self) -> (u32, u32) {
        match self {
            Self::Gray8 { width, height, .. } | Self::Rgb8 { width, height, .. } => (width, height),
        }
    }

    fn data_len(self) -> usize {
        match self {
            Self::Gray8 { data, .. } | Self::Rgb8 { data, .. } => data.len(),
        }
    }

    fn validate_layout(self, subsampling: JpegSubsampling) -> Result<usize, JpegEncodeError> {
        let (width, height, components, name) = match self {
            Self::Gray8 { width, height, .. } => (width, height, 1usize, "Gray8"),
            Self::Rgb8 { width, height, .. } => (width, height, 3usize, "Rgb8"),
        };
        let expected = checked_sample_byte_len(width, height, components)?;
        match (name, subsampling) {
            ("Gray8", JpegSubsampling::Gray)
            | (
                "Rgb8",
                JpegSubsampling::Ybr444 | JpegSubsampling::Ybr422 | JpegSubsampling::Ybr420,
            ) => Ok(expected),
            _ => Err(JpegEncodeError::IncompatibleSubsampling {
                subsampling,
                samples: name,
            }),
        }
    }
}

fn checked_cpu_encode_capacity_plan(
    samples: JpegSamples<'_>,
    sampling: JpegBaselineSampling,
    expected_sample_len: usize,
    restart_interval: Option<u16>,
) -> Result<CpuEncodeCapacityPlan, JpegEncodeError> {
    let (width, height) = samples.dimensions();
    let entropy_capacity =
        jpeg_baseline_entropy_capacity_bytes(width, height, sampling, restart_interval)?;
    let entropy_workspace_bytes =
        entropy_host_workspace_bytes(width, height, sampling, restart_interval, entropy_capacity)?;
    let owned_plane_bytes = match samples {
        JpegSamples::Gray8 { .. } => 0,
        JpegSamples::Rgb8 { .. } => expected_sample_len,
    };
    checked_cpu_encode_live_bytes(
        owned_plane_bytes,
        usize::from(sampling.components),
        entropy_capacity,
        entropy_workspace_bytes,
    )?;
    let plane_capacity_limit =
        cpu_owned_plane_capacity_limit(entropy_capacity, entropy_workspace_bytes)?;
    Ok(CpuEncodeCapacityPlan {
        entropy_capacity,
        entropy_workspace_bytes,
        plane_capacity_limit,
    })
}

fn checked_sample_byte_len(
    width: u32,
    height: u32,
    components: usize,
) -> Result<usize, JpegEncodeError> {
    let requested = usize::try_from(width)
        .ok()
        .and_then(|width| {
            usize::try_from(height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|pixels| pixels.checked_mul(components))
        .ok_or(JpegEncodeError::MemoryCapExceeded {
            requested: usize::MAX,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        })?;
    if requested > DEFAULT_MAX_HOST_ALLOCATION_BYTES {
        return Err(JpegEncodeError::MemoryCapExceeded {
            requested,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        });
    }
    Ok(requested)
}

fn try_vec_with_live_budget<T>(
    capacity: usize,
    live_bytes: &mut usize,
    cap: usize,
) -> Result<Vec<T>, JpegEncodeError> {
    let requested_bytes = capacity.checked_mul(core::mem::size_of::<T>()).ok_or(
        JpegEncodeError::MemoryCapExceeded {
            requested: usize::MAX,
            cap,
        },
    )?;
    let projected =
        live_bytes
            .checked_add(requested_bytes)
            .ok_or(JpegEncodeError::MemoryCapExceeded {
                requested: usize::MAX,
                cap,
            })?;
    if projected > cap {
        return Err(JpegEncodeError::MemoryCapExceeded {
            requested: projected,
            cap,
        });
    }
    let mut plane = Vec::new();
    plane
        .try_reserve_exact(capacity)
        .map_err(|_| JpegEncodeError::HostAllocationFailed {
            bytes: requested_bytes,
        })?;
    let actual_bytes = plane
        .capacity()
        .checked_mul(core::mem::size_of::<T>())
        .ok_or(JpegEncodeError::MemoryCapExceeded {
            requested: usize::MAX,
            cap,
        })?;
    let actual_live =
        live_bytes
            .checked_add(actual_bytes)
            .ok_or(JpegEncodeError::MemoryCapExceeded {
                requested: usize::MAX,
                cap,
            })?;
    if actual_live > cap {
        return Err(JpegEncodeError::MemoryCapExceeded {
            requested: actual_live,
            cap,
        });
    }
    *live_bytes = actual_live;
    Ok(plane)
}

fn component_plane_capacity_bytes(
    outer_capacity: usize,
    planes: &[Cow<'_, [u8]>],
) -> Result<usize, JpegEncodeError> {
    let outer = outer_capacity
        .checked_mul(core::mem::size_of::<Cow<'_, [u8]>>())
        .ok_or(JpegEncodeError::MemoryCapExceeded {
            requested: usize::MAX,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        })?;
    let owned = planes
        .iter()
        .filter_map(|plane| match plane {
            Cow::Borrowed(_) => None,
            Cow::Owned(samples) => Some(samples.capacity()),
        })
        .try_fold(0usize, usize::checked_add)
        .ok_or(JpegEncodeError::MemoryCapExceeded {
            requested: usize::MAX,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        })?;
    checked_encode_host_live_bytes([outer, owned])
}

fn component_planes(
    samples: JpegSamples<'_>,
    subsampling: JpegSubsampling,
    plane_capacity_limit: usize,
) -> Result<Vec<Cow<'_, [u8]>>, JpegEncodeError> {
    let mut live_bytes = 0;
    match samples {
        JpegSamples::Gray8 {
            data,
            width,
            height,
        } => {
            checked_sample_byte_len(width, height, 1)?;
            let mut planes = try_vec_with_live_budget(1, &mut live_bytes, plane_capacity_limit)?;
            planes.push(Cow::Borrowed(data));
            Ok(planes)
        }
        JpegSamples::Rgb8 {
            data,
            width,
            height,
        } => {
            if subsampling == JpegSubsampling::Gray {
                return Err(JpegEncodeError::IncompatibleSubsampling {
                    subsampling,
                    samples: "Rgb8",
                });
            }
            let sample_bytes = checked_sample_byte_len(width, height, 3)?;
            let pixels = sample_bytes / 3;
            let logical_plane_bytes = core::mem::size_of::<Cow<'_, [u8]>>()
                .checked_mul(3)
                .and_then(|metadata| metadata.checked_add(sample_bytes))
                .ok_or(JpegEncodeError::MemoryCapExceeded {
                    requested: usize::MAX,
                    cap: plane_capacity_limit,
                })?;
            if logical_plane_bytes > plane_capacity_limit {
                return Err(JpegEncodeError::MemoryCapExceeded {
                    requested: logical_plane_bytes,
                    cap: plane_capacity_limit,
                });
            }
            let mut planes = try_vec_with_live_budget(3, &mut live_bytes, plane_capacity_limit)?;
            let mut y_plane =
                try_vec_with_live_budget(pixels, &mut live_bytes, plane_capacity_limit)?;
            let mut cb_plane =
                try_vec_with_live_budget(pixels, &mut live_bytes, plane_capacity_limit)?;
            let mut cr_plane =
                try_vec_with_live_budget(pixels, &mut live_bytes, plane_capacity_limit)?;
            for rgb in data.chunks_exact(3) {
                let (y, cb, cr) = rgb_to_ycbcr(rgb[0], rgb[1], rgb[2]);
                y_plane.push(y);
                cb_plane.push(cb);
                cr_plane.push(cr);
            }
            planes.push(Cow::Owned(y_plane));
            planes.push(Cow::Owned(cb_plane));
            planes.push(Cow::Owned(cr_plane));
            Ok(planes)
        }
    }
}

fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let r = i32::from(r);
    let g = i32::from(g);
    let b = i32::from(b);
    let y = (19_595 * r + 38_470 * g + 7_471 * b + 32_768) >> 16;
    let cb = (-11_059 * r - 21_709 * g + 32_768 * b + 8_421_376) >> 16;
    let cr = (32_768 * r - 27_439 * g - 5_329 * b + 8_421_376) >> 16;
    (clamp_u8(y), clamp_u8(cb), clamp_u8(cr))
}

#[expect(
    clippy::cast_sign_loss,
    reason = "RGB-to-YCbCr arithmetic is clamped to the u8 sample range before conversion"
)]
fn clamp_u8(value: i32) -> u8 {
    value.clamp(0, 255) as u8
}

#[expect(
    clippy::too_many_arguments,
    reason = "private JPEG sample hot path keeps scalar arguments for optimized codegen"
)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "edge-replicated source coordinates address validated u8 sample planes"
)]
fn sample_block(
    planes: &[Cow<'_, [u8]>],
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
    component: usize,
    mcu_x: u32,
    mcu_y: u32,
    block_x: u8,
    block_y: u8,
) -> [u8; 64] {
    let mut out = [0u8; 64];
    let max_h = u32::from(sampling.max_h);
    let max_v = u32::from(sampling.max_v);
    let comp_h = u32::from(sampling.h[component]);
    let comp_v = u32::from(sampling.v[component]);
    let x_scale = max_h / comp_h;
    let y_scale = max_v / comp_v;
    let mcu_origin_x = mcu_x * max_h * 8;
    let mcu_origin_y = mcu_y * max_v * 8;
    for y in 0..8u32 {
        for x in 0..8u32 {
            let value = if component == 0 {
                let sx = (mcu_origin_x + u32::from(block_x) * 8 + x).min(width - 1);
                let sy = (mcu_origin_y + u32::from(block_y) * 8 + y).min(height - 1);
                planes[component][(sy as usize * width as usize) + sx as usize]
            } else {
                let mut sum = 0u32;
                for dy in 0..y_scale {
                    for dx in 0..x_scale {
                        let sx = (mcu_origin_x + (u32::from(block_x) * 8 + x) * x_scale + dx)
                            .min(width - 1);
                        let sy = (mcu_origin_y + (u32::from(block_y) * 8 + y) * y_scale + dy)
                            .min(height - 1);
                        sum += u32::from(
                            planes[component][sy as usize * width as usize + sx as usize],
                        );
                    }
                }
                (sum / (x_scale * y_scale)) as u8
            };
            out[(y * 8 + x) as usize] = value;
        }
    }
    out
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "rounded baseline DCT coefficients are bounded by the encoder's validated eight-bit input domain"
)]
fn fdct_quantize(block: &[u8; 64], quant: &[u8; 64], cosine: &[[f64; 8]; 8]) -> [i32; 64] {
    let mut coeffs = [0i32; 64];
    for v in 0..8 {
        for u in 0..8 {
            let mut sum = 0.0;
            for y in 0..8 {
                for x in 0..8 {
                    let sample = f64::from(block[y * 8 + x]) - 128.0;
                    sum += sample * cosine[u][x] * cosine[v][y];
                }
            }
            let cu = if u == 0 {
                core::f64::consts::FRAC_1_SQRT_2
            } else {
                1.0
            };
            let cv = if v == 0 {
                core::f64::consts::FRAC_1_SQRT_2
            } else {
                1.0
            };
            let natural = v * 8 + u;
            let transformed = 0.25 * cu * cv * sum;
            coeffs[natural] = (transformed / f64::from(quant[natural])).round() as i32;
        }
    }
    coeffs
}

pub(crate) fn encode_block(
    coeffs: &[i32; 64],
    prev_dc: &mut i32,
    dc_table: &JpegBaselineHuffmanTable,
    ac_table: &JpegBaselineHuffmanTable,
    writer: &mut BitWriter,
) -> Result<(), JpegEncodeError> {
    let diff = coeffs[0] - *prev_dc;
    *prev_dc = coeffs[0];
    let (dc_size, dc_bits) = magnitude(diff);
    write_huffman_symbol(dc_table, dc_size, writer)?;
    if dc_size > 0 {
        writer.write_bits(dc_bits, dc_size)?;
    }

    let mut zero_run = 0u8;
    for k in 1..64 {
        let coeff = coeffs[JPEG_BASELINE_ZIGZAG[k] as usize];
        if coeff == 0 {
            zero_run = zero_run.saturating_add(1);
            continue;
        }
        while zero_run >= 16 {
            write_huffman_symbol(ac_table, 0xF0, writer)?;
            zero_run -= 16;
        }
        let (size, bits) = magnitude(coeff);
        let symbol = (zero_run << 4) | size;
        write_huffman_symbol(ac_table, symbol, writer)?;
        writer.write_bits(bits, size)?;
        zero_run = 0;
    }
    if zero_run > 0 {
        write_huffman_symbol(ac_table, 0, writer)?;
    }
    Ok(())
}

fn write_huffman_symbol(
    table: &JpegBaselineHuffmanTable,
    symbol: u8,
    writer: &mut BitWriter,
) -> Result<(), JpegEncodeError> {
    let len = table.lens[symbol as usize];
    if len == 0 {
        return Err(JpegEncodeError::MissingHuffmanCode { symbol });
    }
    writer.write_bits(u32::from(table.codes[symbol as usize]), len)
}

pub(crate) fn magnitude(value: i32) -> (u8, u32) {
    if value == 0 {
        return (0, 0);
    }
    let magnitude = value.unsigned_abs();
    let mut remaining = magnitude;
    let mut size = 0u8;
    while remaining > 0 {
        size += 1;
        remaining >>= 1;
    }
    let category_mask = u32::MAX >> (u32::BITS - u32::from(size));
    let bits = if value >= 0 {
        magnitude
    } else {
        category_mask - magnitude
    };
    (size, bits)
}

fn cosine_table() -> [[f64; 8]; 8] {
    let mut table = [[0.0; 8]; 8];
    for (u, row) in (0u32..8).zip(&mut table) {
        for (x, value) in (0u32..8).zip(row) {
            *value = ((f64::from(2 * x + 1) * f64::from(u) * PI) / 16.0).cos();
        }
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_rejects_geometry_above_host_cap_before_length_check() {
        let error = encode_jpeg_baseline(
            JpegSamples::Rgb8 {
                data: &[],
                width: u32::from(u16::MAX),
                height: u32::from(u16::MAX),
            },
            JpegEncodeOptions {
                subsampling: JpegSubsampling::Ybr444,
                backend: JpegBackend::Cpu,
                ..JpegEncodeOptions::default()
            },
        )
        .expect_err("maximum baseline RGB geometry must exceed the host cap");

        assert!(matches!(
            error,
            JpegEncodeError::MemoryCapExceeded { requested, cap }
                if requested > cap && cap == DEFAULT_MAX_HOST_ALLOCATION_BYTES
        ));
    }

    #[test]
    fn restart_one_rejects_cap_valid_geometry_before_sample_or_entropy_allocation() {
        let width = 8_225;
        let height = 65_273;
        assert!(
            usize::try_from(width).unwrap() * usize::try_from(height).unwrap()
                <= DEFAULT_MAX_HOST_ALLOCATION_BYTES
        );

        let error = encode_jpeg_baseline(
            JpegSamples::Gray8 {
                data: &[],
                width,
                height,
            },
            JpegEncodeOptions {
                quality: 100,
                subsampling: JpegSubsampling::Gray,
                restart_interval: Some(1),
                backend: JpegBackend::Cpu,
            },
        )
        .expect_err("conservative encoded output exceeds the shared host cap");

        assert!(matches!(
            error,
            JpegEncodeError::MemoryCapExceeded { requested, cap }
                if requested > cap && cap == DEFAULT_MAX_HOST_ALLOCATION_BYTES
        ));
    }

    #[test]
    fn grayscale_rejects_entropy_and_frame_live_peak_before_sample_allocation() {
        let error = encode_jpeg_baseline(
            JpegSamples::Gray8 {
                data: &[],
                width: 4_096,
                height: 8_192,
            },
            JpegEncodeOptions {
                subsampling: JpegSubsampling::Gray,
                backend: JpegBackend::Cpu,
                ..JpegEncodeOptions::default()
            },
        )
        .expect_err("entropy plus frame capacity exceeds the shared live cap");

        assert!(matches!(
            error,
            JpegEncodeError::MemoryCapExceeded { requested, cap }
                if requested > cap && cap == DEFAULT_MAX_HOST_ALLOCATION_BYTES
        ));
    }

    #[test]
    fn grayscale_component_plane_borrows_the_input() {
        let samples = [3u8, 7, 11, 19];
        let planes = component_planes(
            JpegSamples::Gray8 {
                data: &samples,
                width: 2,
                height: 2,
            },
            JpegSubsampling::Gray,
            DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )
        .expect("grayscale planes");

        assert!(
            matches!(planes.as_slice(), [Cow::Borrowed(data)] if core::ptr::eq(*data, samples.as_slice()))
        );
    }

    #[test]
    fn magnitude_represents_the_full_i32_domain() {
        assert_eq!(magnitude(0), (0, 0));
        assert_eq!(magnitude(5), (3, 5));
        assert_eq!(magnitude(-5), (3, 2));
        assert_eq!(magnitude(i32::MAX), (31, i32::MAX.unsigned_abs()));
        assert_eq!(magnitude(i32::MIN), (32, i32::MAX.unsigned_abs()));
    }

    fn patterned_rgb(width: u32, height: u32) -> Vec<u8> {
        let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
        for y in 0..height {
            for x in 0..width {
                pixels.push(((x * 17 + y * 3) & 0xFF) as u8);
                pixels.push(((x * 5 + y * 11 + 40) & 0xFF) as u8);
                pixels.push(((x * 13 + y * 7 + 90) & 0xFF) as u8);
            }
        }
        pixels
    }

    fn assert_restart_entropy_matches_serial(restart_interval: u16) {
        let width = 160;
        let height = 80;
        let tables = baseline_encode_tables(JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr422,
            restart_interval: Some(restart_interval),
            backend: JpegBackend::Cpu,
        })
        .unwrap();
        let sampling = tables.sampling;
        let cosine = cosine_table();
        let pixels = patterned_rgb(width, height);
        let planes = component_planes(
            JpegSamples::Rgb8 {
                data: &pixels,
                width,
                height,
            },
            JpegSubsampling::Ybr422,
            DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )
        .unwrap();
        let plane_live_bytes = component_plane_capacity_bytes(planes.capacity(), &planes).unwrap();
        let entropy_capacity =
            jpeg_baseline_entropy_capacity_bytes(width, height, sampling, Some(restart_interval))
                .unwrap();

        let serial = encode_entropy_serial(
            &planes,
            width,
            height,
            sampling,
            &tables.q_luma,
            &tables.q_chroma,
            [&tables.huff_dc_luma, &tables.huff_dc_chroma],
            [&tables.huff_ac_luma, &tables.huff_ac_chroma],
            &cosine,
            Some(restart_interval),
            entropy_capacity,
            plane_live_bytes,
        )
        .unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let segmented = pool
            .install(|| {
                encode_entropy_restart_segments(
                    &planes,
                    width,
                    height,
                    sampling,
                    &tables.q_luma,
                    &tables.q_chroma,
                    [&tables.huff_dc_luma, &tables.huff_dc_chroma],
                    [&tables.huff_ac_luma, &tables.huff_ac_chroma],
                    &cosine,
                    restart_interval,
                    entropy_capacity,
                    plane_live_bytes,
                )
            })
            .unwrap();

        assert_eq!(segmented, serial);
        assert!(segmented.windows(2).any(|window| window == [0xFF, 0xD0]));
    }

    #[test]
    fn restart_entropy_segments_match_serial_entropy() {
        assert_restart_entropy_matches_serial(64);
    }

    #[test]
    fn restart_one_entropy_chunks_match_serial_entropy() {
        assert_restart_entropy_matches_serial(1);
    }

    #[test]
    fn restart_segment_fanout_is_bounded_by_chunk_policy() {
        let chunk_count = parallel_entropy_chunk_count(u32::MAX).unwrap();
        assert_eq!(chunk_count, MAX_PARALLEL_ENTROPY_CHUNKS);
        assert_eq!(parallel_entropy_chunk_count(1).unwrap(), 1);
    }

    #[test]
    fn restart_segment_fanout_keeps_work_stealing_granularity() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();

        assert_eq!(
            pool.install(|| parallel_entropy_chunk_count(16)).unwrap(),
            16
        );
    }
}
