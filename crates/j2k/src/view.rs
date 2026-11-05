// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    backend::{
        image as backend_image, image_at_scale as backend_image_at_scale,
        image_with_reduction as backend_image_with_reduction, inspect_info,
        inspect_info_from_image, Image,
    },
    decode::{
        decode_full_image_on_route, decode_image_region_into_with_native_context,
        decode_native_components_with_native_context, decode_warnings_for_image, full_image_route,
        native_components_route, native_region_components_route, validate_buffer, validate_region,
        J2kDecodeOutcome, J2kDecodedComponents, J2kDecodedNativeComponents, NativeRoute,
        PlanesBudget, RouteRequest,
    },
    parse::{parse_image_info, parse_image_info_with_settings, parse_info},
    scratch::J2kScratchPool,
    CpuDecodeParallelism, DecodeSettings, J2kError, J2kReferenceGrid, J2kSupportInfo,
};
use j2k_core::{
    CompressedPayloadKind, CompressedTransferSyntax, Downscale, Info, PassthroughCandidate,
    PixelFormat, Rect,
};

mod deep_scale;
mod plan;
mod rows;
pub use plan::{J2kDecodePlan, J2kDecodeRoute, J2kStreamingBlocker};
mod srgb8;
mod traits;

/// Borrowed parse result for a JP2 or raw JPEG 2000 / HTJ2K codestream.
///
/// Use this when a caller wants to inspect metadata once and build a decoder
/// later without copying compressed tile bytes.
pub struct J2kView<'a> {
    bytes: &'a [u8],
    info: Info,
    reference_grid: J2kReferenceGrid,
    support_info: Option<J2kSupportInfo>,
    image: Option<Image<'a>>,
    settings: DecodeSettings,
    passthrough: Option<(CompressedTransferSyntax, CompressedPayloadKind)>,
}

impl<'a> J2kView<'a> {
    /// Parse container/codestream metadata into a borrowed view.
    ///
    /// # Errors
    /// Returns [`J2kError`] when the input is not a supported JP2/J2C/HTJ2K
    /// stream or when backend inspection rejects the codestream.
    pub fn parse(input: &'a [u8]) -> Result<Self, J2kError> {
        Self::parse_with_settings(input, DecodeSettings::default())
    }

    /// Parse container/codestream metadata with a per-image validation policy.
    ///
    /// [`DecodeSettings::default`] is strict. Explicit lenient mode is limited
    /// to the JP2/JPH metadata recoveries documented by
    /// [`DecodeSettings::lenient`].
    ///
    /// # Errors
    /// Returns [`J2kError`] when the input is unsupported or malformed outside
    /// the selected policy's documented recovery boundary.
    pub fn parse_with_settings(
        input: &'a [u8],
        settings: DecodeSettings,
    ) -> Result<Self, J2kError> {
        let image = backend_image(input, settings, None)?;
        // The settings' recoveries apply to this parse as to the image's, so
        // a recovered file has the metadata its decodes use. Its bytes are
        // not a passthrough candidate: they hold what was recovered from.
        let (info, support_info, passthrough) =
            match parse_image_info_with_settings(input, settings) {
                Ok(parsed) => {
                    let support_info = parsed.into_support_info();
                    let passthrough = (!image.used_lenient_metadata_recovery())
                        .then_some((support_info.transfer_syntax, support_info.payload_kind));
                    (support_info.info.clone(), Some(support_info), passthrough)
                }
                Err(error)
                    if should_fallback_to_backend_after_parse_error(
                        &error,
                        image.used_lenient_metadata_recovery(),
                    ) =>
                {
                    (inspect_info_from_image(&image), None, None)
                }
                Err(error) => return Err(error),
            };
        Ok(Self {
            bytes: input,
            reference_grid: J2kReferenceGrid::of_native(&image, info.dimensions),
            info,
            support_info,
            image: Some(image),
            settings,
            passthrough,
        })
    }

    /// Header-derived image metadata.
    pub fn info(&self) -> &Info {
        &self.info
    }

    /// Where the image and its tiles lie on the JPEG 2000 reference grid.
    #[must_use]
    pub const fn reference_grid(&self) -> J2kReferenceGrid {
        self.reference_grid
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
    reference_grid: J2kReferenceGrid,
    image: Option<Image<'a>>,
    settings: DecodeSettings,
    native_context: j2k_native::DecoderContext<'a>,
    /// Set by [`Self::set_full_decode_planes_budget`]; see
    /// [`Self::full_decode_planes_budget`].
    full_decode_planes_budget: Option<usize>,
}

/// Options for bounded J2K row decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct J2kRowDecodeOptions {
    max_rows_per_stripe: u32,
    max_stripe_bytes: usize,
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

    /// [`Self::inspect_support`] with the JP2/JPH metadata recoveries
    /// `settings` allows a decode: the metadata a decoder made by
    /// [`Self::new_with_settings`] with the same settings decodes with.
    /// Raw codestream inspection is strict under every setting.
    ///
    /// # Errors
    /// Returns [`J2kError`] when the input cannot be parsed as a JP2/JPH file
    /// or raw JPEG 2000 / HTJ2K codestream within the settings' recoveries.
    pub fn inspect_support_with_settings(
        input: &'a [u8],
        settings: DecodeSettings,
    ) -> Result<J2kSupportInfo, J2kError> {
        crate::parse::parse_image_info_with_settings(input, settings)
            .map(crate::parse::ParsedImageInfo::into_support_info)
    }

    /// Inspect where the image and its tiles lie on the JPEG 2000 reference
    /// grid from the codestream's main header, without building a decoder:
    /// what [`Self::reference_grid`] reports for the same input.
    ///
    /// # Errors
    /// Returns [`J2kError`] when the input is not a raw codestream or JP2/JPH
    /// file, or its main header is malformed.
    pub fn inspect_reference_grid(input: &'a [u8]) -> Result<J2kReferenceGrid, J2kError> {
        crate::parse::parse_reference_grid(input)
    }

    /// Create a decoder from compressed bytes.
    ///
    /// # Errors
    /// Returns [`J2kError`] for unsupported or malformed input.
    pub fn new(input: &'a [u8]) -> Result<Self, J2kError> {
        Self::from_view(J2kView::parse(input)?)
    }

    /// Create a decoder with a per-image validation policy.
    ///
    /// The policy is retained by this decoder and reused by full-resolution,
    /// region, and scaled decode jobs. It does not mutate [`crate::J2kContext`]
    /// or leak into another decoder.
    ///
    /// # Errors
    /// Returns [`J2kError`] for unsupported or malformed input.
    pub fn new_with_settings(input: &'a [u8], settings: DecodeSettings) -> Result<Self, J2kError> {
        Self::from_view(J2kView::parse_with_settings(input, settings)?)
    }

    /// Create a decoder from a previously parsed [`J2kView`].
    ///
    /// # Errors
    /// Returns [`J2kError`] if the parsed view cannot be promoted to a decoder.
    pub fn from_view(view: J2kView<'a>) -> Result<Self, J2kError> {
        Ok(Self {
            bytes: view.bytes,
            info: view.info,
            reference_grid: view.reference_grid,
            image: view.image,
            settings: view.settings,
            native_context: j2k_native::DecoderContext::default(),
            full_decode_planes_budget: None,
        })
    }

    /// Header-derived image metadata.
    pub fn info(&self) -> &Info {
        &self.info
    }

    /// Where the image and its tiles lie on the JPEG 2000 reference grid.
    ///
    /// Reduced-resolution decodes cover the reduced samples whose
    /// reference-grid footprints overlap the request; build device plans
    /// with [`crate::DeviceDecodePlan::for_reference_grid`] and this grid.
    #[must_use]
    pub const fn reference_grid(&self) -> J2kReferenceGrid {
        self.reference_grid
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
        let decoded = image
            .decode_components_with_context(native_context)
            .map_err(J2kError::from_native_decode_error)?;
        let retained_image_bytes = component_handoff_image_bytes(image)?;
        J2kDecodedComponents::try_from_native(decoded, retained_image_bytes)
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
        let decoded = image
            .decode_region_components_with_context((roi.x, roi.y, roi.w, roi.h), native_context)
            .map_err(J2kError::from_native_decode_error)?;
        let retained_image_bytes = component_handoff_image_bytes(image)?;
        J2kDecodedComponents::try_from_native(decoded, retained_image_bytes)
    }

    /// Decode the full image into owned native-bit-depth component planes.
    ///
    /// This preserves per-component bit depth, signedness, sampling, and byte
    /// width for callers that cannot use a single interleaved packed bitmap.
    /// A decode whose `f32` component planes would pass the native decode
    /// budget runs as a region decode of the whole image, which packs rows
    /// into the returned planes as they are finished.
    ///
    /// # Errors
    /// Returns [`J2kError`] when the codestream cannot be decoded.
    pub fn decode_native_components(&mut self) -> Result<J2kDecodedNativeComponents, J2kError> {
        self.ensure_image()?;
        let planes_budget = self.full_decode_planes_budget();
        let (Some(image), native_context) = (self.image.as_ref(), &mut self.native_context) else {
            return Err(J2kError::internal_backend("internal image cache missing"));
        };
        let planes_budget = match planes_budget {
            PlanesBudget::Default => j2k_native::DEFAULT_MAX_DECODE_BYTES,
            PlanesBudget::Fixed(bytes) => bytes,
        };
        let route = native_components_route(image, planes_budget);
        let decoded =
            decode_native_components_with_native_context(image, native_context, None, route);
        let retained_image_bytes = component_handoff_image_bytes(image);
        if route == NativeRoute::Rows {
            self.native_context = self.fresh_native_context();
        }
        J2kDecodedNativeComponents::try_from_native(decoded?, retained_image_bytes?)
    }

    /// Extract selected codestream wavelet bands on the CPU, without inverse
    /// DWT, inverse color transformation, ICC conversion, or pixel packing.
    ///
    /// Reversible 5/3 bands use exact I64 storage; irreversible 9/7 bands are
    /// dequantized F32. Tile boundaries and component metadata are preserved.
    /// See [`crate::WaveletSelection`] for selection and memory semantics.
    pub fn decode_wavelets(
        &mut self,
        selection: &crate::WaveletSelection,
    ) -> Result<crate::WaveletImage, J2kError> {
        self.ensure_image()?;
        let (Some(image), context) = (self.image.as_ref(), &mut self.native_context) else {
            return Err(J2kError::internal_backend("internal image cache missing"));
        };
        image
            .decode_wavelets_with_context(context, selection)
            .map_err(J2kError::from_native_decode_error)
    }

    /// Decode a source-coordinate region into owned native-bit-depth component
    /// planes.
    ///
    /// Like 16-bit region decodes, it packs rows into the returned planes as
    /// they are finished, without region-sized `f32` planes, for RGB images
    /// without palette, channel definition, sYCC, `CIELab` or image and tile
    /// offsets whose components share one precision, and for other such
    /// images of up to four components when the planes would pass the
    /// native decode budget.
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
        let route = native_region_components_route(image, roi);
        let decoded =
            decode_native_components_with_native_context(image, native_context, Some(roi), route);
        let retained_image_bytes = component_handoff_image_bytes(image);
        if route == NativeRoute::Rows {
            self.native_context = self.fresh_native_context();
        }
        J2kDecodedNativeComponents::try_from_native(decoded?, retained_image_bytes?)
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
    /// A decode whose component planes would pass the native decode budget
    /// runs as a region decode of the whole image, which writes rows as they
    /// are finished instead of holding the planes: at 8 bits, and at 16 bits
    /// for images whose components share one precision of at most 24 bits.
    /// A 16-bit output holds each sample's value clamped to the 16-bit range
    /// (two's complement when signed); samples of at most 8 bits are widened
    /// across it.
    ///
    /// # Errors
    /// Returns [`J2kError`] when the format is unsupported, the output buffer
    /// is too small, or the codestream fails during decode. A decode that
    /// runs as a region decode and fails partway may already have written
    /// some rows of `out`.
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
    /// Decodes exactly as [`Self::decode_into`], which writes directly into
    /// the caller's output buffer. The pool is accepted to satisfy the shared
    /// codec trait; no decode path currently draws on it. Native decode
    /// allocations are reused through this decoder's own context instead.
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

    /// Decode like [`Self::decode_into`], borrowing the decoded-component,
    /// Tier-1, and IDWT allocations of a retained worker workspace.
    pub(crate) fn decode_into_with_workspace(
        &mut self,
        workspace: &mut j2k_native::DecoderWorkspace,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<J2kDecodeOutcome, J2kError> {
        let parallelism = self.native_context.cpu_decode_parallelism();
        self.native_context =
            j2k_native::DecoderContext::from_workspace(core::mem::take(workspace));
        self.native_context.set_cpu_decode_parallelism(parallelism);
        let result = self.decode_into_cached(out, stride, fmt);
        // Parsed graphs borrow this image; into_workspace releases them even
        // after a failed decode and retains only lifetime-free allocations.
        *workspace = core::mem::take(&mut self.native_context).into_workspace();
        result
    }

    fn decode_into_cached(
        &mut self,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<J2kDecodeOutcome, J2kError> {
        validate_buffer(self.info.dimensions, out.len(), stride, fmt)?;
        self.ensure_image()?;
        let budget = self.full_decode_planes_budget();
        let (Some(image), native_context) = (self.image.as_ref(), &mut self.native_context) else {
            return Err(J2kError::internal_backend("internal image cache missing"));
        };
        let request = RouteRequest {
            fmt,
            budget,
            parallelism: native_context.cpu_decode_parallelism(),
            components: self.info.components,
            coded_bytes: self.bytes.len(),
            scale: Downscale::None,
        };
        let route = full_image_route(image, request);
        let decoded =
            decode_full_image_on_route(image, native_context, out, stride, request, &route);
        let warnings = decode_warnings_for_image(image);
        if route.streams() {
            // As after a region decode, release the decode buffers rather
            // than hold them while the decoder is idle.
            self.native_context = self.fresh_native_context();
        }
        decoded?;
        Ok(j2k_core::DecodeOutcome::new(
            Rect::full(self.info.dimensions),
            warnings,
        ))
    }

    /// Decode a source-coordinate region into `out`.
    ///
    /// `roi` is expressed in full-resolution source pixels. The output buffer
    /// must hold `roi.w * roi.h * fmt.bytes_per_pixel()` bytes with the
    /// provided row stride.
    ///
    /// Reduced-resolution and region decoding validate the packets and
    /// structural dependencies needed for the requested output and may leave
    /// other packets unexamined; detected errors in required data remain
    /// fatal, success does not establish whole-codestream validity, and
    /// acceptance is independent of execution route or thread count.
    ///
    /// # Errors
    /// Returns [`J2kError`] when the region is out of bounds, the output buffer
    /// is too small, the format is unsupported, or decode fails.
    ///
    /// A decode that fails partway may already have written some rows of
    /// `out`.
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
        let decoded = decode_image_region_into_with_native_context(
            image,
            native_context,
            out,
            stride,
            fmt,
            roi,
        );
        let warnings = decode_warnings_for_image(image);
        // Region planes are zeroed before every region decode, so keeping
        // them would only save their allocation. Release them and the other
        // decode buffers rather than hold them while the decoder is idle.
        self.native_context = self.fresh_native_context();
        decoded?;
        Ok(j2k_core::DecodeOutcome::new(roi, warnings))
    }

    /// Decode the full image at a reduced resolution.
    ///
    /// `scale` uses the shared [`Downscale`] contract; `Downscale::None`
    /// delegates to full-resolution decode. As with [`Self::decode_into`],
    /// a decode whose component planes would pass the native decode budget
    /// runs as a region decode of the whole reduced image.
    ///
    /// Reduced-resolution and region decoding validate the packets and
    /// structural dependencies needed for the requested output and may leave
    /// other packets unexamined; detected errors in required data remain
    /// fatal, success does not establish whole-codestream validity, and
    /// acceptance is independent of execution route or thread count.
    ///
    /// # Errors
    /// Returns [`J2kError`] when the format or scale request is unsupported,
    /// including a scale deeper than the codestream's resolution ladder, the
    /// output buffer is too small, or decode fails. A decode that runs as a
    /// region decode and fails partway may already have written some rows of
    /// `out`.
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
        let image = backend_image_at_scale(self.bytes, self.settings, scale)?;
        let warnings = decode_warnings_for_image(&image);
        let image_dims = (image.width(), image.height());
        validate_buffer(image_dims, out.len(), stride, fmt)?;
        let mut native_context = self.fresh_native_context();
        let request = RouteRequest {
            fmt,
            budget: self.full_decode_planes_budget(),
            parallelism: native_context.cpu_decode_parallelism(),
            components: self.info.components,
            coded_bytes: self.bytes.len(),
            scale,
        };
        let route = full_image_route(&image, request);
        decode_full_image_on_route(&image, &mut native_context, out, stride, request, &route)?;
        Ok(j2k_core::DecodeOutcome::new(
            Rect::full(image_dims),
            warnings,
        ))
    }

    /// Decode a source-coordinate region at a reduced resolution.
    ///
    /// `roi` is expressed in full-resolution source pixels. The decoded output
    /// covers the reduced-resolution samples whose reference-grid footprints
    /// overlap `roi`: `roi.scaled_covering(scale)` when the image's
    /// reference-grid origin is a multiple of the scale's denominator. Another
    /// origin shifts the reduced grid against the image's own, as the
    /// standard defines it, and the returned outcome reports the covered
    /// rectangle in reduced-resolution coordinates.
    ///
    /// Reduced-resolution and region decoding validate the packets and
    /// structural dependencies needed for the requested output and may leave
    /// other packets unexamined; detected errors in required data remain
    /// fatal, success does not establish whole-codestream validity, and
    /// acceptance is independent of execution route or thread count.
    ///
    /// # Errors
    /// Returns [`J2kError`] when the region is out of bounds, the scale or
    /// pixel format is unsupported (including a scale deeper than the
    /// codestream's resolution ladder), the output buffer is too small, or
    /// decode fails.
    ///
    /// A decode that fails partway may already have written some rows of
    /// `out`.
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
        let image = backend_image_at_scale(self.bytes, self.settings, scale)?;
        let scaled_roi =
            deep_scale::image_covering(&image, self.info.dimensions, roi, scale.denominator());
        validate_buffer((scaled_roi.w, scaled_roi.h), out.len(), stride, fmt)?;
        let warnings = decode_warnings_for_image(&image);
        let image_dims = (image.width(), image.height());
        validate_region(scaled_roi, image_dims)?;
        let mut native_context = self.fresh_native_context();
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
            self.image = Some(backend_image(self.bytes, self.settings, None)?);
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

    /// What a full-image decode's component planes may charge: the native
    /// decode budget, past which the planes decode is refused, with
    /// streaming wherever it was measured faster, or on one thread nearly as
    /// fast in far less memory, unless a test set a fixed budget.
    fn full_decode_planes_budget(&self) -> PlanesBudget {
        self.full_decode_planes_budget
            .map_or(PlanesBudget::Default, PlanesBudget::Fixed)
    }

    /// Test hook: hold full-image decodes' component planes to `bytes` and
    /// drop the preference for streaming, so that a decode takes the
    /// streamed region route exactly when its planes would pass `bytes`: 0
    /// streams every decode that can stream, `usize::MAX` keeps every decode
    /// the planes can finish on the planes. The native decode
    /// budget itself is unchanged.
    #[doc(hidden)]
    pub fn set_full_decode_planes_budget(&mut self, bytes: usize) {
        self.full_decode_planes_budget = Some(bytes);
    }

    fn fresh_native_context(&self) -> j2k_native::DecoderContext<'a> {
        let mut native_context = j2k_native::DecoderContext::default();
        native_context.set_cpu_decode_parallelism(self.native_context.cpu_decode_parallelism());
        native_context
    }
}

fn component_handoff_image_bytes(image: &Image<'_>) -> Result<usize, J2kError> {
    image
        .retained_allocation_bytes()
        .map_err(J2kError::from_native_decode_error)
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

fn should_fallback_to_backend_after_parse_error(
    error: &J2kError,
    used_lenient_metadata_recovery: bool,
) -> bool {
    should_retry_with_backend(error) || used_lenient_metadata_recovery
}

#[cfg(test)]
mod tests;
