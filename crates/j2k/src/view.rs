// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    backend::{
        image as backend_image, inspect_info, inspect_info_from_image, DecodeSettings, Image,
    },
    context::J2kContext,
    decode::{
        decode_image_into_with_native_context, decode_image_region_into_with_native_context,
        decode_warnings_for_settings, validate_buffer, validate_region, J2kDecodeOutcome,
        J2kDecodeWarning, J2kDecodedComponents, J2kDecodedNativeComponents,
    },
    parse::{parse_image_info, parse_info},
    scratch::J2kScratchPool,
    CpuDecodeParallelism, J2kError, J2kSupportInfo,
};
use alloc::vec::Vec;
use j2k_core::{
    BufferError, CompressedPayloadKind, CompressedTransferSyntax, DecodeRowsError, DecoderContext,
    Downscale, ImageCodec, ImageDecode, ImageDecodeRows, Info, PassthroughCandidate, PixelFormat,
    Rect, RowSink, TileBatchDecode, TileRegionScaledDecodeJob, DEFAULT_MAX_HOST_ALLOCATION_BYTES,
};

/// Borrowed parse result for a JP2 or raw JPEG 2000 / HTJ2K codestream.
///
/// Use this when a caller wants to inspect metadata once and build a decoder
/// later without copying compressed tile bytes.
pub struct J2kView<'a> {
    bytes: &'a [u8],
    info: Info,
    support_info: Option<J2kSupportInfo>,
    image: Option<Image<'a>>,
    passthrough: Option<(CompressedTransferSyntax, CompressedPayloadKind)>,
}

impl<'a> J2kView<'a> {
    /// Parse container/codestream metadata into a borrowed view.
    ///
    /// # Errors
    /// Returns [`J2kError`] when the input is not a supported JP2/J2C/HTJ2K
    /// stream or when backend inspection rejects the codestream.
    pub fn parse(input: &'a [u8]) -> Result<Self, J2kError> {
        let (info, support_info, passthrough) = match parse_image_info(input) {
            Ok(parsed) => {
                let support_info = parsed.into_support_info();
                let passthrough = Some((support_info.transfer_syntax, support_info.payload_kind));
                (support_info.info.clone(), Some(support_info), passthrough)
            }
            Err(error) if should_retry_with_backend(&error) => (inspect_info(input)?, None, None),
            Err(error) => return Err(error),
        };
        let image = Some(backend_image(input, DecodeSettings::default())?);
        Ok(Self {
            bytes: input,
            info,
            support_info,
            image,
            passthrough,
        })
    }

    /// Header-derived image metadata.
    pub fn info(&self) -> &Info {
        &self.info
    }

    /// Full JPEG 2000 / HTJ2K support metadata when header parsing classified
    /// the payload without falling back to backend-only inspection.
    pub fn support_info(&self) -> Option<&J2kSupportInfo> {
        self.support_info.as_ref()
    }

    /// Original compressed bytes backing this view.
    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Return a byte-preserving passthrough candidate when the native parser
    /// classified the compressed syntax and payload shape.
    pub fn passthrough_candidate(&self) -> Option<PassthroughCandidate<'a>> {
        self.passthrough.map(|(transfer_syntax, payload_kind)| {
            PassthroughCandidate::new(self.bytes, transfer_syntax, payload_kind, self.info.clone())
        })
    }
}

/// JPEG 2000 / HTJ2K decoder with full-frame, ROI, and scaled output methods.
///
/// The decoder borrows compressed tile bytes and owns reusable native decode
/// context so repeated operations can avoid reparsing backend state.
pub struct J2kDecoder<'a> {
    bytes: &'a [u8],
    info: Info,
    image: Option<Image<'a>>,
    native_context: j2k_native::DecoderContext<'a>,
}

/// Options for bounded J2K row decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct J2kRowDecodeOptions {
    max_rows_per_stripe: u32,
    max_stripe_bytes: usize,
}

impl J2kRowDecodeOptions {
    /// Create row decode options with the requested maximum stripe height.
    ///
    /// A zero value is normalized to one row.
    pub const fn new(max_rows_per_stripe: u32) -> Self {
        Self {
            max_rows_per_stripe,
            max_stripe_bytes: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        }
    }

    /// Create row decode options with explicit stripe height and byte caps.
    pub const fn new_with_max_stripe_bytes(
        max_rows_per_stripe: u32,
        max_stripe_bytes: usize,
    ) -> Self {
        Self {
            max_rows_per_stripe,
            max_stripe_bytes,
        }
    }

    /// Return a copy of these options with an explicit stripe byte cap.
    #[must_use]
    pub const fn with_max_stripe_bytes(mut self, max_stripe_bytes: usize) -> Self {
        self.max_stripe_bytes = max_stripe_bytes;
        self
    }

    /// Maximum number of decoded rows held per bounded row-decode stripe.
    pub const fn max_rows_per_stripe(self) -> u32 {
        if self.max_rows_per_stripe == 0 {
            1
        } else {
            self.max_rows_per_stripe
        }
    }

    /// Maximum number of packed bytes held per bounded row-decode stripe.
    pub const fn max_stripe_bytes(self) -> usize {
        self.max_stripe_bytes
    }
}

impl Default for J2kRowDecodeOptions {
    fn default() -> Self {
        Self::new(64)
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
/// Marker type used by generic tile-batch decode traits.
pub struct J2kCodec;

impl<'a> J2kDecoder<'a> {
    /// Inspect JP2/J2C/HTJ2K metadata without decoding pixels.
    ///
    /// # Errors
    /// Returns [`J2kError`] when the input cannot be parsed or inspected as a
    /// supported JPEG 2000 / HTJ2K image.
    pub fn inspect(input: &'a [u8]) -> Result<Info, J2kError> {
        match parse_info(input) {
            Ok(info) => Ok(info),
            Err(error) if should_retry_with_backend(&error) => inspect_info(input),
            Err(error) => Err(error),
        }
    }

    /// Inspect full JPEG 2000 / HTJ2K support metadata without decoding pixels.
    ///
    /// # Errors
    /// Returns [`J2kError`] when the input cannot be parsed as a JP2/JPH file
    /// or raw JPEG 2000 / HTJ2K codestream.
    pub fn inspect_support(input: &'a [u8]) -> Result<J2kSupportInfo, J2kError> {
        parse_image_info(input).map(crate::parse::ParsedImageInfo::into_support_info)
    }

    /// Create a decoder from compressed bytes.
    ///
    /// # Errors
    /// Returns [`J2kError`] for unsupported or malformed input.
    pub fn new(input: &'a [u8]) -> Result<Self, J2kError> {
        Self::from_view(J2kView::parse(input)?)
    }

    /// Create a decoder from a previously parsed [`J2kView`].
    ///
    /// # Errors
    /// Returns [`J2kError`] if the parsed view cannot be promoted to a decoder.
    pub fn from_view(view: J2kView<'a>) -> Result<Self, J2kError> {
        Ok(Self {
            bytes: view.bytes,
            info: view.info,
            image: view.image,
            native_context: j2k_native::DecoderContext::default(),
        })
    }

    /// Header-derived image metadata.
    pub fn info(&self) -> &Info {
        &self.info
    }

    /// Decode the full image into borrowed component planes.
    ///
    /// This exposes the native component-plane path for callers that need
    /// arbitrary component counts, per-component bit depths, or non-RGB
    /// component interpretation. The returned planes borrow this decoder's
    /// reusable native decode context.
    ///
    /// # Errors
    /// Returns [`J2kError`] when the codestream cannot be decoded.
    pub fn decode_components(&mut self) -> Result<J2kDecodedComponents<'_>, J2kError> {
        self.ensure_image()?;
        let (Some(image), native_context) = (self.image.as_ref(), &mut self.native_context) else {
            return Err(J2kError::internal_backend("internal image cache missing"));
        };
        image
            .decode_components_with_context(native_context)
            .map(|decoded| J2kDecodedComponents::from_native(&decoded))
            .map_err(J2kError::from_native_decode_error)
    }

    /// Decode a source-coordinate region into borrowed component planes.
    ///
    /// # Errors
    /// Returns [`J2kError`] when the region is invalid or the codestream cannot
    /// be decoded.
    pub fn decode_region_components(
        &mut self,
        roi: Rect,
    ) -> Result<J2kDecodedComponents<'_>, J2kError> {
        validate_region(roi, self.info.dimensions)?;
        self.ensure_image()?;
        let (Some(image), native_context) = (self.image.as_ref(), &mut self.native_context) else {
            return Err(J2kError::internal_backend("internal image cache missing"));
        };
        image
            .decode_region_components_with_context((roi.x, roi.y, roi.w, roi.h), native_context)
            .map(|decoded| J2kDecodedComponents::from_native(&decoded))
            .map_err(J2kError::from_native_decode_error)
    }

    /// Decode the full image into owned native-bit-depth component planes.
    ///
    /// This preserves per-component bit depth, signedness, sampling, and byte
    /// width for callers that cannot use a single interleaved packed bitmap.
    ///
    /// # Errors
    /// Returns [`J2kError`] when the codestream cannot be decoded.
    pub fn decode_native_components(&mut self) -> Result<J2kDecodedNativeComponents, J2kError> {
        self.ensure_image()?;
        let (Some(image), native_context) = (self.image.as_ref(), &mut self.native_context) else {
            return Err(J2kError::internal_backend("internal image cache missing"));
        };
        image
            .decode_native_components_with_context(native_context)
            .map(|decoded| J2kDecodedNativeComponents::from_native(&decoded))
            .map_err(J2kError::from_native_decode_error)
    }

    /// Decode a source-coordinate region into owned native-bit-depth component
    /// planes.
    ///
    /// # Errors
    /// Returns [`J2kError`] when the region is invalid or the codestream cannot
    /// be decoded.
    pub fn decode_native_region_components(
        &mut self,
        roi: Rect,
    ) -> Result<J2kDecodedNativeComponents, J2kError> {
        validate_region(roi, self.info.dimensions)?;
        self.ensure_image()?;
        let (Some(image), native_context) = (self.image.as_ref(), &mut self.native_context) else {
            return Err(J2kError::internal_backend("internal image cache missing"));
        };
        image
            .decode_native_region_components_with_context(
                (roi.x, roi.y, roi.w, roi.h),
                native_context,
            )
            .map(|decoded| J2kDecodedNativeComponents::from_native(&decoded))
            .map_err(J2kError::from_native_decode_error)
    }

    /// Return the CPU decode parallelism policy for this decoder.
    pub fn cpu_decode_parallelism(&self) -> CpuDecodeParallelism {
        CpuDecodeParallelism::from_native(self.native_context.cpu_decode_parallelism())
    }

    /// Set the CPU decode parallelism policy for this decoder.
    pub fn set_cpu_decode_parallelism(&mut self, parallelism: CpuDecodeParallelism) {
        self.native_context
            .set_cpu_decode_parallelism(parallelism.to_native());
    }

    /// Decode the full image into `out` using `stride` bytes per output row.
    ///
    /// # Errors
    /// Returns [`J2kError`] when the format is unsupported, the output buffer
    /// is too small, or the codestream fails during decode.
    pub fn decode_into(
        &mut self,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<J2kDecodeOutcome, J2kError> {
        self.decode_into_cached(out, stride, fmt)
    }

    /// Decode the full image with caller-owned scratch.
    ///
    /// The current native full-frame path writes directly into the caller's
    /// output buffer; the pool is accepted to satisfy the shared codec trait
    /// and is used by reduced-resolution and row-bounded paths.
    ///
    /// # Errors
    /// Same as [`Self::decode_into`].
    pub fn decode_into_with_scratch(
        &mut self,
        _pool: &mut J2kScratchPool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<J2kDecodeOutcome, J2kError> {
        self.decode_into_cached(out, stride, fmt)
    }

    fn decode_into_cached(
        &mut self,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<J2kDecodeOutcome, J2kError> {
        validate_buffer(self.info.dimensions, out.len(), stride, fmt)?;
        self.ensure_image()?;
        let (Some(image), native_context) = (self.image.as_ref(), &mut self.native_context) else {
            return Err(J2kError::internal_backend("internal image cache missing"));
        };
        decode_image_into_with_native_context(image, native_context, out, stride, fmt)?;
        Ok(j2k_core::DecodeOutcome::new(
            Rect::full(self.info.dimensions),
            decode_warnings_for_settings(DecodeSettings::default()),
        ))
    }

    /// Decode a source-coordinate region into `out`.
    ///
    /// `roi` is expressed in full-resolution source pixels. The output buffer
    /// must hold `roi.w * roi.h * fmt.bytes_per_pixel()` bytes with the
    /// provided row stride.
    ///
    /// # Errors
    /// Returns [`J2kError`] when the region is out of bounds, the output buffer
    /// is too small, the format is unsupported, or decode fails.
    pub fn decode_region_into(
        &mut self,
        _pool: &mut J2kScratchPool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<J2kDecodeOutcome, J2kError> {
        self.decode_region_into_cached(out, stride, fmt, roi)
    }

    fn decode_region_into_cached(
        &mut self,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<J2kDecodeOutcome, J2kError> {
        validate_region(roi, self.info.dimensions)?;
        validate_buffer((roi.w, roi.h), out.len(), stride, fmt)?;
        self.ensure_image()?;
        let (Some(image), native_context) = (self.image.as_ref(), &mut self.native_context) else {
            return Err(J2kError::internal_backend("internal image cache missing"));
        };
        decode_image_region_into_with_native_context(image, native_context, out, stride, fmt, roi)?;
        Ok(j2k_core::DecodeOutcome::new(
            roi,
            decode_warnings_for_settings(DecodeSettings::default()),
        ))
    }

    /// Decode the full image at a reduced resolution.
    ///
    /// `scale` uses the shared [`Downscale`] contract; `Downscale::None`
    /// delegates to full-resolution decode.
    ///
    /// # Errors
    /// Returns [`J2kError`] when the format or scale request is unsupported,
    /// the output buffer is too small, or decode fails.
    pub fn decode_scaled_into(
        &mut self,
        pool: &mut J2kScratchPool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        scale: Downscale,
    ) -> Result<J2kDecodeOutcome, J2kError> {
        if scale == Downscale::None {
            return self.decode_into_with_scratch(pool, out, stride, fmt);
        }
        let settings = DecodeSettings {
            target_resolution: Some(self.scaled_target_dims(scale)),
            ..DecodeSettings::default()
        };
        let warnings = decode_warnings_for_settings(settings);
        let image = backend_image(self.bytes, settings)?;
        let image_dims = (image.width(), image.height());
        validate_buffer(image_dims, out.len(), stride, fmt)?;
        let mut native_context = self.scaled_decode_native_context();
        decode_image_into_with_native_context(&image, &mut native_context, out, stride, fmt)?;
        Ok(j2k_core::DecodeOutcome::new(
            Rect::full(image_dims),
            warnings,
        ))
    }

    /// Decode a source-coordinate region at a reduced resolution.
    ///
    /// `roi` is expressed in full-resolution source pixels. The decoded output
    /// covers `roi.scaled_covering(scale)` in reduced-resolution coordinates.
    ///
    /// # Errors
    /// Returns [`J2kError`] when the region is out of bounds, the scale or
    /// pixel format is unsupported, the output buffer is too small, or decode
    /// fails.
    pub fn decode_region_scaled_into(
        &mut self,
        pool: &mut J2kScratchPool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
    ) -> Result<J2kDecodeOutcome, J2kError> {
        if scale == Downscale::None {
            return self.decode_region_into(pool, out, stride, fmt, roi);
        }
        validate_region(roi, self.info.dimensions)?;
        let scaled_roi = roi.scaled_covering(scale);
        validate_buffer((scaled_roi.w, scaled_roi.h), out.len(), stride, fmt)?;
        let settings = DecodeSettings {
            target_resolution: Some(self.scaled_target_dims(scale)),
            ..DecodeSettings::default()
        };
        let warnings = decode_warnings_for_settings(settings);
        let image = backend_image(self.bytes, settings)?;
        let image_dims = (image.width(), image.height());
        validate_region(scaled_roi, image_dims)?;
        let mut native_context = self.scaled_decode_native_context();
        decode_image_region_into_with_native_context(
            &image,
            &mut native_context,
            out,
            stride,
            fmt,
            scaled_roi,
        )?;
        Ok(j2k_core::DecodeOutcome::new(scaled_roi, warnings))
    }

    fn ensure_image(&mut self) -> Result<(), J2kError> {
        if self.image.is_none() {
            self.image = Some(backend_image(self.bytes, DecodeSettings::default())?);
            if self.info.tile_layout.is_none() {
                self.info = inspect_info_from_image(self.cached_image()?);
            }
        }
        Ok(())
    }

    fn cached_image(&self) -> Result<&Image<'a>, J2kError> {
        self.image
            .as_ref()
            .ok_or_else(|| J2kError::internal_backend("internal image cache missing"))
    }

    fn scaled_target_dims(&self, scale: Downscale) -> (u32, u32) {
        (
            self.info.dimensions.0.div_ceil(scale.denominator()),
            self.info.dimensions.1.div_ceil(scale.denominator()),
        )
    }

    fn scaled_decode_native_context(&self) -> j2k_native::DecoderContext<'a> {
        let mut native_context = j2k_native::DecoderContext::default();
        native_context.set_cpu_decode_parallelism(self.native_context.cpu_decode_parallelism());
        native_context
    }

    /// Decode rows into a `u8` row sink while bounding host output scratch to
    /// at most `options.max_rows_per_stripe()` rows.
    ///
    /// # Errors
    /// Returns a decode error for unsupported formats or malformed input, and
    /// forwards sink errors without converting them to successful decodes.
    pub fn decode_rows_u8_bounded<R: RowSink<u8>>(
        &mut self,
        sink: &mut R,
        options: J2kRowDecodeOptions,
    ) -> Result<J2kDecodeOutcome, DecodeRowsError<J2kError, R::Error>> {
        let fmt = row_format_u8(self.info()).map_err(DecodeRowsError::Decode)?;
        let row_bytes = row_bytes_for(self.info(), fmt).map_err(DecodeRowsError::Decode)?;
        let width = self.info.dimensions.0;
        let height = self.info.dimensions.1;
        let (stripe_rows, max_stripe_len) = bounded_row_stripe_layout(row_bytes, height, options)
            .map_err(DecodeRowsError::Decode)?;
        let mut pool = J2kScratchPool::new();
        let mut y = 0_u32;
        while y < height {
            let rows = stripe_rows.min(height - y);
            let stripe_len = row_bytes.checked_mul(rows as usize).ok_or_else(|| {
                DecodeRowsError::Decode(J2kError::Buffer(BufferError::SizeOverflow {
                    what: "J2K bounded row decode stripe buffer",
                }))
            })?;
            let stripe = pool.packed_bytes(max_stripe_len);
            self.decode_region_into_cached(
                &mut stripe[..stripe_len],
                row_bytes,
                fmt,
                Rect {
                    x: 0,
                    y,
                    w: width,
                    h: rows,
                },
            )
            .map_err(DecodeRowsError::Decode)?;
            for row_index in 0..rows {
                let start = row_index as usize * row_bytes;
                sink.write_row(y + row_index, &stripe[start..start + row_bytes])
                    .map_err(DecodeRowsError::Sink)?;
            }
            y += rows;
        }
        Ok(j2k_core::DecodeOutcome::new(
            Rect::full(self.info.dimensions),
            Vec::new(),
        ))
    }

    /// Decode rows into a `u16` row sink while bounding host output scratch to
    /// at most `options.max_rows_per_stripe()` rows.
    ///
    /// # Errors
    /// Returns a decode error for unsupported formats or malformed input, and
    /// forwards sink errors without converting them to successful decodes.
    pub fn decode_rows_u16_bounded<R: RowSink<u16>>(
        &mut self,
        sink: &mut R,
        options: J2kRowDecodeOptions,
    ) -> Result<J2kDecodeOutcome, DecodeRowsError<J2kError, R::Error>> {
        let fmt = row_format_u16(self.info()).map_err(DecodeRowsError::Decode)?;
        let row_bytes = row_bytes_for(self.info(), fmt).map_err(DecodeRowsError::Decode)?;
        let samples_per_row = row_samples_for(self.info(), fmt).map_err(DecodeRowsError::Decode)?;
        let width = self.info.dimensions.0;
        let height = self.info.dimensions.1;
        let (stripe_rows, max_stripe_len) = bounded_row_stripe_layout(row_bytes, height, options)
            .map_err(DecodeRowsError::Decode)?;
        let mut pool = J2kScratchPool::new();
        let mut y = 0_u32;
        while y < height {
            let rows = stripe_rows.min(height - y);
            let stripe_len = row_bytes.checked_mul(rows as usize).ok_or_else(|| {
                DecodeRowsError::Decode(J2kError::Buffer(BufferError::SizeOverflow {
                    what: "J2K bounded row decode stripe buffer",
                }))
            })?;
            let (packed, row) = pool.packed_bytes_and_row_u16(max_stripe_len, samples_per_row);
            self.decode_region_into_cached(
                &mut packed[..stripe_len],
                row_bytes,
                fmt,
                Rect {
                    x: 0,
                    y,
                    w: width,
                    h: rows,
                },
            )
            .map_err(DecodeRowsError::Decode)?;
            for row_index in 0..rows {
                let start = row_index as usize * row_bytes;
                let packed_row = &packed[start..start + row_bytes];
                for (dst, src) in row.iter_mut().zip(packed_row.chunks_exact(2)) {
                    *dst = u16::from_le_bytes([src[0], src[1]]);
                }
                sink.write_row(y + row_index, row)
                    .map_err(DecodeRowsError::Sink)?;
            }
            y += rows;
        }
        Ok(j2k_core::DecodeOutcome::new(
            Rect::full(self.info.dimensions),
            Vec::new(),
        ))
    }
}

#[doc(hidden)]
impl ImageCodec for J2kDecoder<'_> {
    type Error = J2kError;
    type Warning = J2kDecodeWarning;
    type Pool = J2kScratchPool;
}

#[doc(hidden)]
impl<'a> ImageDecode<'a> for J2kDecoder<'a> {
    type View = J2kView<'a>;

    fn inspect(input: &'a [u8]) -> Result<Info, Self::Error> {
        Self::inspect(input)
    }

    fn parse(input: &'a [u8]) -> Result<Self::View, Self::Error> {
        J2kView::parse(input)
    }

    fn from_view(view: Self::View) -> Result<Self, Self::Error> {
        Self::from_view(view)
    }

    fn decode_into(
        &mut self,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<j2k_core::DecodeOutcome<Self::Warning>, Self::Error> {
        J2kDecoder::decode_into(self, out, stride, fmt)
    }

    fn decode_into_with_scratch(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<j2k_core::DecodeOutcome<Self::Warning>, Self::Error> {
        J2kDecoder::decode_into_with_scratch(self, pool, out, stride, fmt)
    }

    fn decode_region_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<j2k_core::DecodeOutcome<Self::Warning>, Self::Error> {
        J2kDecoder::decode_region_into(self, pool, out, stride, fmt, roi)
    }

    fn decode_scaled_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        scale: Downscale,
    ) -> Result<j2k_core::DecodeOutcome<Self::Warning>, Self::Error> {
        J2kDecoder::decode_scaled_into(self, pool, out, stride, fmt, scale)
    }

    fn decode_region_scaled_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
    ) -> Result<j2k_core::DecodeOutcome<Self::Warning>, Self::Error> {
        J2kDecoder::decode_region_scaled_into(self, pool, out, stride, fmt, roi, scale)
    }
}

#[doc(hidden)]
impl<'a> ImageDecodeRows<'a, u8> for J2kDecoder<'a> {
    fn decode_rows<R: RowSink<u8>>(
        &mut self,
        sink: &mut R,
    ) -> Result<j2k_core::DecodeOutcome<Self::Warning>, DecodeRowsError<Self::Error, R::Error>>
    {
        self.decode_rows_u8_bounded(sink, J2kRowDecodeOptions::default())
    }
}

#[doc(hidden)]
impl<'a> ImageDecodeRows<'a, u16> for J2kDecoder<'a> {
    fn decode_rows<R: RowSink<u16>>(
        &mut self,
        sink: &mut R,
    ) -> Result<j2k_core::DecodeOutcome<Self::Warning>, DecodeRowsError<Self::Error, R::Error>>
    {
        self.decode_rows_u16_bounded(sink, J2kRowDecodeOptions::default())
    }
}

#[doc(hidden)]
impl ImageCodec for J2kCodec {
    type Error = J2kError;
    type Warning = J2kDecodeWarning;
    type Pool = J2kScratchPool;
}

#[doc(hidden)]
impl TileBatchDecode for J2kCodec {
    type Context = J2kContext;

    fn decode_tile(
        ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &[u8],
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<j2k_core::DecodeOutcome<Self::Warning>, Self::Error> {
        let mut decoder = J2kDecoder::new(input)?;
        decoder.set_cpu_decode_parallelism(ctx.codec().cpu_decode_parallelism());
        decoder.decode_into_with_scratch(pool, out, stride, fmt)
    }

    fn decode_tile_region(
        ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &[u8],
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<j2k_core::DecodeOutcome<Self::Warning>, Self::Error> {
        let mut decoder = J2kDecoder::new(input)?;
        decoder.set_cpu_decode_parallelism(ctx.codec().cpu_decode_parallelism());
        decoder.decode_region_into(pool, out, stride, fmt, roi)
    }

    fn decode_tile_scaled(
        ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &[u8],
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        scale: Downscale,
    ) -> Result<j2k_core::DecodeOutcome<Self::Warning>, Self::Error> {
        let mut decoder = J2kDecoder::new(input)?;
        decoder.set_cpu_decode_parallelism(ctx.codec().cpu_decode_parallelism());
        decoder.decode_scaled_into(pool, out, stride, fmt, scale)
    }

    fn decode_tile_region_scaled(
        ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        fmt: PixelFormat,
        job: TileRegionScaledDecodeJob<'_, '_>,
    ) -> Result<j2k_core::DecodeOutcome<Self::Warning>, Self::Error> {
        let TileRegionScaledDecodeJob {
            input,
            out,
            stride,
            roi,
            scale,
        } = job;
        let mut decoder = J2kDecoder::new(input)?;
        decoder.set_cpu_decode_parallelism(ctx.codec().cpu_decode_parallelism());
        decoder.decode_region_scaled_into(pool, out, stride, fmt, roi, scale)
    }
}

fn row_format_u8(info: &Info) -> Result<PixelFormat, J2kError> {
    match info.components {
        1 => Ok(PixelFormat::Gray8),
        3 => Ok(PixelFormat::Rgb8),
        4 => Ok(PixelFormat::Rgba8),
        _ => Err(j2k_core::Unsupported {
            what: "row decode only supports Gray/RGB/RGBA images in J2K-M2",
        }
        .into()),
    }
}

fn row_format_u16(info: &Info) -> Result<PixelFormat, J2kError> {
    match info.components {
        1 => Ok(PixelFormat::Gray16),
        3 => Ok(PixelFormat::Rgb16),
        4 => Ok(PixelFormat::Rgba16),
        _ => Err(j2k_core::Unsupported {
            what: "row decode only supports Gray/RGB/RGBA images in J2K-M2",
        }
        .into()),
    }
}

fn bounded_row_stripe_layout(
    row_bytes: usize,
    height: u32,
    options: J2kRowDecodeOptions,
) -> Result<(u32, usize), J2kError> {
    let cap = options.max_stripe_bytes();
    if row_bytes > cap {
        return Err(J2kError::Buffer(BufferError::AllocationTooLarge {
            requested: row_bytes,
            cap,
            what: "J2K bounded row decode stripe buffer",
        }));
    }

    let max_rows = options.max_rows_per_stripe();
    let image_rows = height.max(1);
    let rows_by_cap = cap.checked_div(row_bytes).map_or(max_rows, |capped_rows| {
        u32::try_from(capped_rows).unwrap_or(u32::MAX).max(1)
    });
    let stripe_rows = max_rows.min(image_rows).min(rows_by_cap);
    let max_stripe_len = row_bytes
        .checked_mul(stripe_rows as usize)
        .ok_or(J2kError::Buffer(BufferError::SizeOverflow {
            what: "J2K bounded row decode stripe buffer",
        }))?;

    Ok((stripe_rows, max_stripe_len))
}

fn row_bytes_for(info: &Info, fmt: PixelFormat) -> Result<usize, J2kError> {
    (info.dimensions.0 as usize)
        .checked_mul(fmt.bytes_per_pixel())
        .ok_or(J2kError::DimensionOverflow {
            width: info.dimensions.0,
            height: info.dimensions.1,
        })
}

fn row_samples_for(info: &Info, fmt: PixelFormat) -> Result<usize, J2kError> {
    (info.dimensions.0 as usize)
        .checked_mul(fmt.channels())
        .ok_or(J2kError::DimensionOverflow {
            width: info.dimensions.0,
            height: info.dimensions.1,
        })
}

fn should_retry_with_backend(error: &J2kError) -> bool {
    matches!(
        error,
        J2kError::InvalidMarker {
            marker: 0x50
                | 0x53
                | 0x55
                | 0x57
                | 0x58
                | 0x59
                | 0x5C
                | 0x5D
                | 0x5E
                | 0x5F
                | 0x60
                | 0x61
                | 0x63
                | 0x64
                | 0x91
                | 0x92,
            ..
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scaled_decode_native_context_preserves_configured_parallelism() {
        let mut decoder = J2kDecoder {
            bytes: &[],
            info: Info {
                dimensions: (1, 1),
                components: 1,
                colorspace: j2k_core::Colorspace::SGray,
                bit_depth: 8,
                tile_layout: None,
                coded_unit_layout: None,
                restart_interval: None,
                resolution_levels: 1,
            },
            image: None,
            native_context: j2k_native::DecoderContext::default(),
        };
        decoder.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);

        let native_context = decoder.scaled_decode_native_context();

        assert_eq!(
            native_context.cpu_decode_parallelism(),
            CpuDecodeParallelism::Serial.to_native()
        );
    }

    #[test]
    fn bounded_row_stripe_layout_clamps_rows_to_byte_cap() {
        let options = J2kRowDecodeOptions::new_with_max_stripe_bytes(100, 1_024);

        let (rows, bytes) =
            bounded_row_stripe_layout(100, 50, options).expect("stripe layout should fit");

        assert_eq!(rows, 10);
        assert_eq!(bytes, 1_000);
    }

    #[test]
    fn bounded_row_stripe_layout_rejects_single_row_over_cap() {
        let options = J2kRowDecodeOptions::new_with_max_stripe_bytes(100, 99);

        let err =
            bounded_row_stripe_layout(100, 50, options).expect_err("single row should exceed cap");

        assert!(matches!(
            err,
            J2kError::Buffer(BufferError::AllocationTooLarge {
                requested: 100,
                cap: 99,
                what: "J2K bounded row decode stripe buffer",
            })
        ));
    }
}
