// SPDX-License-Identifier: MIT OR Apache-2.0

//! Safe owned codec state. Parsing, planning, and pixel reconstruction stay in J2K.

#![forbid(unsafe_code)]

use crate::{
    abi::{
        J2kCComponentInfo, J2kCDecodeOptions, J2kCImageInfo, J2kCOpenOptions, J2kCOutputInfo,
        J2K_ALLOCATION_FAILED, J2K_ALPHA_NONE, J2K_ALPHA_PREMULTIPLIED, J2K_ALPHA_STRAIGHT,
        J2K_I16_LE, J2K_INVALID_INPUT, J2K_OPEN_FLAGS, J2K_OPEN_LENIENT, J2K_U16_LE, J2K_U8,
        J2K_UNSUPPORTED, J2K_WARNING_LENIENT_RECOVERY,
    },
    error::{Error, Result},
};
use j2k::{
    BatchAlpha, BatchDecodeOptions, BatchLayout, CpuBatchDecoder, CpuBatchSamples, DecodeRequest,
    DecodeSettings, Downscale, EncodedImage, J2kDecodeWarning, J2kDecoder, J2kReferenceGrid,
    J2kSupportInfo, NativeSampleType, PreparedBatch, Rect,
};
use std::{num::NonZeroUsize, sync::Arc};

const DEFAULT_LIMIT: u64 = 512 * 1024 * 1024;
/// Most decode threads a decoder's own pool may have.
const MAX_WORKERS: u32 = 64;

pub(crate) struct Decoder {
    pub(crate) bytes: Arc<[u8]>,
    pub(crate) settings: DecodeSettings,
    lenient: bool,
    support: J2kSupportInfo,
    grid: J2kReferenceGrid,
    session: CpuBatchDecoder,
    /// The decoder's own threads when `workers` asked for them; otherwise the
    /// codec's shared pool runs its work.
    pub(crate) pool: Option<rayon::ThreadPool>,
    prepared: Option<(DecodeRequest, PreparedBatch, J2kCOutputInfo)>,
    max_output: u64,
    pub(crate) poisoned: bool,
}

// The C contract permits serial ownership transfer between threads.
const _: fn() = || {
    fn require_send<T: Send>() {}
    require_send::<Decoder>();
};

impl Decoder {
    pub(crate) fn input_limit(options: J2kCOpenOptions) -> Result<u64> {
        if options.flags & !J2K_OPEN_FLAGS != 0 || options.workers > MAX_WORKERS {
            return Err(Error::argument("unknown open flags, or workers above 64"));
        }
        Ok(if options.max_input_bytes == 0 {
            DEFAULT_LIMIT
        } else {
            options.max_input_bytes
        })
    }

    pub(crate) fn open(input: &[u8], options: J2kCOpenOptions) -> Result<Self> {
        if input.len() as u64 > Self::input_limit(options)? {
            return Err(Error::limit("encoded input exceeds max_input_bytes"));
        }
        if !is_jpeg2000(input) {
            return Err(Error::new(
                J2K_INVALID_INPUT,
                "input is not a JP2/JPH file or a raw JPEG 2000 codestream",
            ));
        }
        let settings = if options.flags & J2K_OPEN_LENIENT == 0 {
            DecodeSettings::strict()
        } else {
            DecodeSettings::lenient()
        };
        let support = J2kDecoder::inspect_support_with_settings(input, settings)?;
        let grid = J2kDecoder::inspect_reference_grid(input)?;
        let pool = match options.workers {
            0 => None,
            threads => Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads as usize)
                    .thread_name(|index| format!("j2k-c-{index}"))
                    .build()
                    .map_err(|_| {
                        Error::new(
                            J2K_ALLOCATION_FAILED,
                            "could not start the decoder's threads",
                        )
                    })?,
            ),
        };
        let mut batch_options = BatchDecodeOptions::default();
        batch_options.layout = BatchLayout::Nhwc;
        batch_options.settings = settings;
        // The codec spreads a lone image over the rayon pool it runs in whatever the
        // worker count, so more batch workers would only be allocated and left idle.
        batch_options.workers = NonZeroUsize::new(1);
        Ok(Self {
            // A single infallible allocation: exhaustion aborts, as documented in docs/api.md.
            bytes: Arc::from(input),
            settings,
            lenient: options.flags & J2K_OPEN_LENIENT != 0,
            support,
            grid,
            session: CpuBatchDecoder::new(batch_options),
            pool,
            prepared: None,
            max_output: if options.max_output_bytes == 0 {
                DEFAULT_LIMIT
            } else {
                options.max_output_bytes
            },
            poisoned: false,
        })
    }

    pub(crate) fn info(&self) -> J2kCImageInfo {
        J2kCImageInfo {
            width: self.support.info.dimensions.0,
            height: self.support.info.dimensions.1,
            components: self.support.info.components.into(),
            resolution_levels: self.support.info.resolution_levels.into(),
            has_signed_components: self.support.has_signed_components().into(),
            has_icc_profile: self
                .support
                .file_metadata
                .as_ref()
                .is_some_and(j2k::J2kFileMetadata::has_icc_profile)
                .into(),
            image_x0: self.grid.image_origin.0,
            image_y0: self.grid.image_origin.1,
            tile_x0: self.grid.tile_origin.0,
            tile_y0: self.grid.tile_origin.1,
            tile_width: self.grid.tile_size.0,
            tile_height: self.grid.tile_size.1,
            ..Default::default()
        }
    }

    /// The source image's width and height.
    pub(crate) fn image_size(&self) -> (u32, u32) {
        self.support.info.dimensions
    }

    pub(crate) fn component(&self, index: u32) -> Result<J2kCComponentInfo> {
        let value = self
            .support
            .components
            .get(index as usize)
            .ok_or_else(|| Error::argument("component index is outside the codestream"))?;
        Ok(J2kCComponentInfo {
            precision: value.bit_depth.into(),
            is_signed: value.signed.into(),
            sampling_x: value.x_rsiz.into(),
            sampling_y: value.y_rsiz.into(),
            ..Default::default()
        })
    }

    pub(crate) fn prepare(&mut self, options: J2kCDecodeOptions) -> Result<J2kCOutputInfo> {
        let request = decode_request(options)?;
        if let Some((cached, _, info)) = &self.prepared {
            if *cached == request {
                return Ok(*info);
            }
        }
        // Codec planning cost follows the declared geometry, so reject forged image or
        // tile dimensions before planning when even one byte per pixel exceeds the limit.
        self.check_output_limit(self.minimum_output_bytes(options))?;
        // Keep at most one plan; release it before preparing a different request.
        self.prepared = None;
        let input = Vec::from([EncodedImage::new(Arc::clone(&self.bytes), request)]);
        let Self { session, pool, .. } = self;
        let batch =
            in_pool(pool.as_ref(), || session.prepare(input)).map_err(Error::infrastructure)?;
        if let Some(error) = batch.errors().first() {
            return Err(Error::item(&error.source));
        }
        let [group] = batch.groups() else {
            return Err(Error::internal("expected one prepared image group"));
        };
        let [image] = group.images() else {
            return Err(Error::internal("expected one prepared image"));
        };
        let info = group.info();
        let (sample_type, width) = match info.sample_type {
            NativeSampleType::U8 => (J2K_U8, 1_u64),
            NativeSampleType::U16 => (J2K_U16_LE, 2),
            NativeSampleType::I16 => (J2K_I16_LE, 2),
            _ => {
                return Err(Error::new(
                    J2K_UNSUPPORTED,
                    "unsupported native sample type",
                ))
            }
        };
        let alpha = match info.alpha {
            BatchAlpha::None => J2K_ALPHA_NONE,
            BatchAlpha::Straight => J2K_ALPHA_STRAIGHT,
            BatchAlpha::Premultiplied => J2K_ALPHA_PREMULTIPLIED,
            _ => {
                return Err(Error::new(
                    J2K_UNSUPPORTED,
                    "unsupported alpha interpretation",
                ))
            }
        };
        let rect = image.plan().output_rect();
        if rect.w == 0 || rect.h == 0 {
            return Err(Error::internal("codec planned an empty output rectangle"));
        }
        let channels = u32::try_from(info.color.channels())
            .map_err(|_| Error::internal("channel count does not fit u32"))?;
        let row_bytes = u64::from(rect.w) * u64::from(channels) * width;
        let buffer_bytes = row_bytes
            .checked_mul(u64::from(rect.h))
            .ok_or_else(|| Error::limit("output size overflow"))?;
        self.check_output_limit(buffer_bytes)?;
        let output = J2kCOutputInfo {
            x: rect.x,
            y: rect.y,
            width: rect.w,
            height: rect.h,
            channels,
            sample_type,
            precision: info.precision.into(),
            alpha,
            row_bytes,
            buffer_bytes,
            warnings: if image.used_lenient_metadata_recovery() {
                J2K_WARNING_LENIENT_RECOVERY
            } else {
                0
            },
            ..Default::default()
        };
        self.prepared = Some((request, batch, output));
        Ok(output)
    }

    pub(crate) fn output_span(&self, info: J2kCOutputInfo, stride: usize) -> Result<usize> {
        if (stride as u64) < info.row_bytes {
            return Err(Error::argument("stride is smaller than row_bytes"));
        }
        let required = (stride as u64)
            .checked_mul(u64::from(info.height) - 1)
            .and_then(|prefix| prefix.checked_add(info.row_bytes))
            .ok_or_else(|| Error::limit("strided output size overflow"))?;
        self.check_output_limit(required)?;
        usize::try_from(required)
            .map_err(|_| Error::limit("output does not fit the host address space"))
    }

    /// One byte per pixel of the requested area, clipped to the source, at the requested
    /// resolution. Never exceeds the prepared `buffer_bytes` of a valid request.
    fn minimum_output_bytes(&self, options: J2kCDecodeOptions) -> u64 {
        let (width, height) = self.support.info.dimensions;
        let span = |start: u32, length: u32, end: u32| {
            u64::from(start.saturating_add(length).min(end) - start.min(end))
        };
        let (columns, rows) = if [options.x, options.y, options.width, options.height] == [0; 4] {
            (u64::from(width), u64::from(height))
        } else {
            (
                span(options.x, options.width, width),
                span(options.y, options.height, height),
            )
        };
        (columns >> options.reduction) * (rows >> options.reduction)
    }

    fn check_output_limit(&self, bytes: u64) -> Result<()> {
        if bytes > self.max_output || bytes > isize::MAX as u64 {
            return Err(Error::limit(
                "output exceeds max_output_bytes or the host address space",
            ));
        }
        Ok(())
    }

    pub(crate) fn decode_prepared(&mut self, output: &mut [u8], stride: usize) -> Result<()> {
        let (_, prepared, info) = self
            .prepared
            .as_ref()
            .ok_or_else(|| Error::internal("decode requires a prepared request"))?;
        let Self { session, pool, .. } = self;
        let result = in_pool(pool.as_ref(), || session.decode_prepared(prepared))
            .map_err(Error::infrastructure)?;
        if let Some(error) = result.errors().first() {
            return Err(Error::item(&error.source));
        }
        let [group] = result.groups() else {
            return Err(Error::internal("expected one decoded image group"));
        };
        let expected_rect = Rect {
            x: info.x,
            y: info.y,
            w: info.width,
            h: info.height,
        };
        if group.source_indices() != [0]
            || group.decoded_rects() != [expected_rect]
            || group.info() != prepared.groups()[0].info()
            || !group
                .warnings()
                .iter()
                .flatten()
                .all(|warning| self.allows(*warning))
        {
            return Err(Error::internal(
                "decoded output disagrees with the prepared contract",
            ));
        }
        let row_samples = info.width as usize * info.channels as usize;
        if group.samples().len() != row_samples * info.height as usize {
            return Err(Error::internal(
                "decoded sample count disagrees with output geometry",
            ));
        }
        // No writes occur until the complete codec result and layout have succeeded.
        match (group.samples(), info.sample_type) {
            (CpuBatchSamples::U8(samples), J2K_U8) => {
                for (src, dst) in samples
                    .chunks_exact(row_samples)
                    .zip(output.chunks_mut(stride))
                {
                    dst[..src.len()].copy_from_slice(src);
                }
            }
            (CpuBatchSamples::U16(samples), J2K_U16_LE) => {
                for (src, dst) in samples
                    .chunks_exact(row_samples)
                    .zip(output.chunks_mut(stride))
                {
                    for (value, bytes) in src.iter().zip(dst.as_chunks_mut::<2>().0) {
                        bytes.copy_from_slice(&value.to_le_bytes());
                    }
                }
            }
            (CpuBatchSamples::I16(samples), J2K_I16_LE) => {
                for (src, dst) in samples
                    .chunks_exact(row_samples)
                    .zip(output.chunks_mut(stride))
                {
                    for (value, bytes) in src.iter().zip(dst.as_chunks_mut::<2>().0) {
                        bytes.copy_from_slice(&value.to_le_bytes());
                    }
                }
            }
            _ => {
                return Err(Error::internal(
                    "decoded sample type disagrees with output layout",
                ))
            }
        }
        Ok(())
    }
}

impl Decoder {
    /// Whether this decoder's settings permit a decode to report `warning`:
    /// a lenient decoder its metadata recoveries, a strict one nothing.
    pub(crate) fn allows(&self, warning: J2kDecodeWarning) -> bool {
        self.lenient && matches!(warning, J2kDecodeWarning::LenientMetadataRecovery)
    }
}

/// Run `operation` on `pool` when the decoder has its own threads, and on the
/// caller's thread (with the codec's shared pool) otherwise.
pub(crate) fn in_pool<T: Send>(
    pool: Option<&rayon::ThreadPool>,
    operation: impl FnOnce() -> T + Send,
) -> T {
    match pool {
        Some(pool) => pool.install(operation),
        None => operation(),
    }
}

/// `J2K_WARNING_*` bits of decode warnings.
pub(crate) fn warning_bits(warnings: &[J2kDecodeWarning]) -> u32 {
    warnings.iter().fold(0, |bits, warning| match warning {
        J2kDecodeWarning::LenientMetadataRecovery => bits | J2K_WARNING_LENIENT_RECOVERY,
        _ => bits,
    })
}

/// JP2/JPH signature box, or the SOC and SIZ markers that open a raw codestream.
fn is_jpeg2000(input: &[u8]) -> bool {
    const JP2_SIGNATURE: [u8; 12] = [0, 0, 0, 12, b'j', b'P', b' ', b' ', 0x0D, 0x0A, 0x87, 0x0A];
    input.starts_with(&JP2_SIGNATURE) || input.starts_with(&[0xFF, 0x4F, 0xFF, 0x51])
}

fn decode_request(options: J2kCDecodeOptions) -> Result<DecodeRequest> {
    if options.reserved != 0 {
        return Err(Error::argument("reserved decode options must be zero"));
    }
    let scale = match options.reduction {
        0 => Downscale::None,
        1 => Downscale::Half,
        2 => Downscale::Quarter,
        3 => Downscale::Eighth,
        _ => {
            return Err(Error::new(
                J2K_UNSUPPORTED,
                "reduction must be between zero and three",
            ))
        }
    };
    let full = [options.x, options.y, options.width, options.height] == [0; 4];
    let roi = Rect {
        x: options.x,
        y: options.y,
        w: options.width,
        h: options.height,
    };
    if !full && (roi.w == 0 || roi.h == 0) {
        return Err(Error::argument("region width and height must be positive"));
    }
    Ok(match (full, options.reduction) {
        (true, 0) => DecodeRequest::Full,
        (true, _) => DecodeRequest::Reduced { scale },
        (false, 0) => DecodeRequest::Region { roi },
        (false, _) => DecodeRequest::RegionReduced { roi, scale },
    })
}
