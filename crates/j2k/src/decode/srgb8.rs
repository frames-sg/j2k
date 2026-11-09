// SPDX-License-Identifier: MIT OR Apache-2.0

//! Explicit 8-bit sRGB output normalization.

use super::{decode_warnings_for_image, J2kDecodeWarning};
use crate::{backend::ColorSpace, backend::Image, BatchAlpha, J2kDecodeControl, J2kError};
use alloc::vec::Vec;
use j2k_core::{
    ensure_allocation_within_cap, try_host_vec_filled, BufferError, HostAllocationError,
    PixelFormat, Rect, DEFAULT_MAX_HOST_ALLOCATION_BYTES,
};
use moxcms::{CmsError, ColorProfile, DataColorSpace, Layout, ParsingOptions, TransformOptions};

const SRGB8_OUTPUT_WHAT: &str = "J2K sRGB8 normalized output";

/// Interleaved sample layout returned by [`J2kSrgb8Image`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum J2kSrgb8Layout {
    /// One 8-bit sRGB-grey sample per pixel.
    Gray,
    /// Three interleaved 8-bit sRGB samples per pixel.
    Rgb,
    /// Three interleaved 8-bit sRGB samples followed by alpha per pixel.
    Rgba,
}

impl J2kSrgb8Layout {
    const fn channels(self) -> usize {
        match self {
            Self::Gray => 1,
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }

    pub(crate) const fn format(self) -> PixelFormat {
        match self {
            Self::Gray => PixelFormat::Gray8,
            Self::Rgb => PixelFormat::Rgb8,
            Self::Rgba => PixelFormat::Rgba8,
        }
    }
}

/// Owned, tightly packed 8-bit sRGB decode result, with straight alpha in RGBA.
#[derive(Debug, PartialEq, Eq)]
pub struct J2kSrgb8Image {
    dimensions: (u32, u32),
    layout: J2kSrgb8Layout,
    data: Vec<u8>,
    warnings: Vec<J2kDecodeWarning>,
}

impl J2kSrgb8Image {
    /// Alpha association of the normalized samples. RGBA output always has
    /// straight opacity; gray and RGB output have no alpha channel.
    #[must_use]
    pub const fn alpha(&self) -> BatchAlpha {
        match self.layout {
            J2kSrgb8Layout::Rgba => BatchAlpha::Straight,
            J2kSrgb8Layout::Gray | J2kSrgb8Layout::Rgb => BatchAlpha::None,
        }
    }

    /// Non-fatal warnings from the decode, such as a lenient metadata
    /// recovery the decoder's settings allowed.
    #[must_use]
    pub fn warnings(&self) -> &[J2kDecodeWarning] {
        &self.warnings
    }

    /// Decoded image dimensions.
    #[must_use]
    pub const fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    /// Interleaved sample layout.
    #[must_use]
    pub const fn layout(&self) -> J2kSrgb8Layout {
        self.layout
    }

    /// Tightly packed 8-bit image samples.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Consume the image and return its tightly packed sample storage.
    #[must_use]
    pub fn into_data(self) -> Vec<u8> {
        self.data
    }

    /// Consume the image and return its samples as channel planes: every
    /// pixel's first channel, then every pixel's second, and so on, as planar
    /// (NCHW) layouts store them. Gray output is already one plane.
    ///
    /// The packed samples and their planar copy are both live while the layout
    /// converts, within `control`'s working limit.
    ///
    /// # Errors
    /// Returns a resource-limit error when both copies exceed the working
    /// limit, or a buffer error when the planar copy cannot be allocated.
    pub fn into_planar_data_with_control(
        self,
        control: &J2kDecodeControl,
    ) -> Result<Vec<u8>, J2kError> {
        let channels = self.layout.channels();
        if channels == 1 {
            return Ok(self.data);
        }
        let budget = control
            .to_native()
            .beside_retained_bytes(self.data.capacity())
            .map_err(J2kError::from_native_decode_error)?;
        budget
            .clone()
            .beside_retained_bytes(self.data.len())
            .map_err(J2kError::from_native_decode_error)?;
        let mut planar = Vec::new();
        planar.try_reserve_exact(self.data.len()).map_err(|_| {
            BufferError::HostAllocationFailed {
                bytes: self.data.len(),
                what: "J2K planar sRGB8 output",
            }
        })?;
        budget
            .beside_retained_bytes(planar.capacity())
            .map_err(J2kError::from_native_decode_error)?;
        for channel in 0..channels {
            planar.extend(self.data.iter().skip(channel).step_by(channels).copied());
        }
        Ok(planar)
    }

    fn with_warnings(mut self, warnings: Vec<J2kDecodeWarning>) -> Self {
        self.warnings = warnings;
        self
    }
}

/// The output of an image whose sRGB8 samples are the samples
/// `decode_into` writes in one of its own pixel formats: a Gray, RGB or
/// RGBA image without a primary ICC profile. Such an image decodes on
/// `decode_into`'s full-image route, straight into the result.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RoutedSrgb8 {
    dimensions: (u32, u32),
    layout: J2kSrgb8Layout,
    retained_image_bytes: usize,
    premultiplied: bool,
}

impl RoutedSrgb8 {
    /// The routed output of `image`, or `None` when its samples need an ICC
    /// conversion, a grey-alpha expansion or source-precision unassociation
    /// first.
    pub(crate) fn plan(image: &Image<'_>) -> Result<Option<Self>, J2kError> {
        if image.primary_icc_profile().is_some()
            || image
                .unassociates_alpha_in_source_values()
                .map_err(J2kError::from_native_decode_error)?
        {
            return Ok(None);
        }
        let layout = match (image.color_space(), image.has_alpha()) {
            (ColorSpace::Gray, false) => J2kSrgb8Layout::Gray,
            (ColorSpace::RGB, false) => J2kSrgb8Layout::Rgb,
            (ColorSpace::RGB, true) => J2kSrgb8Layout::Rgba,
            _ => return Ok(None),
        };
        let retained_image_bytes = image
            .retained_allocation_bytes()
            .map_err(J2kError::from_native_decode_error)?;
        let premultiplied = image
            .srgb8_alpha_is_premultiplied()
            .map_err(J2kError::from_native_decode_error)?;
        Ok(Some(Self {
            dimensions: (image.width(), image.height()),
            layout,
            retained_image_bytes,
            premultiplied,
        }))
    }

    pub(crate) const fn format(self) -> PixelFormat {
        self.layout.format()
    }

    pub(crate) const fn stride(self) -> usize {
        self.dimensions.0 as usize * self.layout.channels()
    }

    /// The zeroed result buffer, within the host allocation cap and the
    /// control's output and working limits beside the image's retained
    /// metadata and already-live reusable context storage.
    pub(crate) fn allocate(
        self,
        retained_context_bytes: usize,
        control: &J2kDecodeControl,
    ) -> Result<Vec<u8>, J2kError> {
        control.check_output(expected_len(self.dimensions, self.layout)?)?;
        allocate_output(
            self.dimensions,
            self.layout,
            checked_peak_bytes(self.retained_image_bytes, retained_context_bytes, 0)?,
            control,
        )
    }

    pub(crate) fn finish(
        self,
        mut data: Vec<u8>,
        warnings: Vec<J2kDecodeWarning>,
        control: &J2kDecodeControl,
    ) -> Result<J2kSrgb8Image, J2kError> {
        if self.premultiplied {
            straighten_alpha(&mut data, 4, control)?;
        }
        finish(data, self.dimensions, self.layout).map(|image| image.with_warnings(warnings))
    }
}

/// Decode the whole of `image` into component planes, then normalize them
/// to 8-bit sRGB: the route of images [`RoutedSrgb8::plan`] does not route.
pub(crate) fn decode_image_srgb8_with_control<'a>(
    image: &Image<'a>,
    native_context: &mut j2k_native::DecoderContext<'a>,
    control: &J2kDecodeControl,
) -> Result<J2kSrgb8Image, J2kError> {
    check_icc_working_control(image, control)?;
    control.check_output(expected_len(
        (image.width(), image.height()),
        srgb8_output_layout(image)?,
    )?)?;
    let warnings = decode_warnings_for_image(image);
    decode_image_srgb8_planes(image, native_context, control)
        .map(|decoded| decoded.with_warnings(warnings))
}

/// Check an explicit working limit before any colour-management state exists.
///
/// The pinned CMS is bounded only for profiles that connect through XYZ
/// (see [`icc_conversion_working_bytes`]); others are refused under a limit.
/// The bound, beside the image's retained metadata, must fit the limit. It
/// also covers the profile parse that output planning performs.
pub(crate) fn check_icc_working_control(
    image: &Image<'_>,
    control: &J2kDecodeControl,
) -> Result<(), J2kError> {
    control.check_cancelled()?;
    if control.max_working_memory_bytes.is_none() {
        return Ok(());
    }
    let Some(profile) = image
        .primary_icc_profile()
        .or_else(|| match image.color_space() {
            ColorSpace::Icc { profile, .. } => Some(profile.as_slice()),
            _ => None,
        })
    else {
        return Ok(());
    };
    if profile.get(20..24) != Some(b"XYZ ".as_slice()) {
        return Err(j2k_core::Unsupported {
            what: "working-memory limits require ICC profiles with an XYZ connection space",
        }
        .into());
    }
    let retained = image
        .retained_allocation_bytes()
        .map_err(J2kError::from_native_decode_error)?;
    checked_conversion_peak(
        retained,
        icc_conversion_working_bytes(profile.len()),
        control,
    )
    .map(|_| ())
}

/// Heap the pinned colour-management crate (`moxcms` 0.8.1) holds for one
/// restricted conversion of `profile_len` bytes, after
/// [`restricted_profile_tags`]: the parsed copy, up to three tone curves of at
/// most 65,536 entries each (none larger than the profile), and the fixed
/// 8-bit linearization, gamma and fused tables of a matrix/TRC or gray-TRC
/// transform with an XYZ connection space, with slack for its small owners.
/// Transforming pixels allocates nothing. `icc_text_tags` measures it.
fn icc_conversion_working_bytes(profile_len: usize) -> usize {
    const CURVE_BYTES: usize = 65_536 * 2;
    const TRANSFORM_BYTES: usize = 200 << 10;
    profile_len
        .saturating_add(profile_len.min(CURVE_BYTES).saturating_mul(3))
        .saturating_add(TRANSFORM_BYTES)
}

/// Whether a parsed profile takes the bounded matrix/TRC or gray-TRC path.
fn has_bounded_icc_transform(profile: &ColorProfile) -> bool {
    profile.pcs == DataColorSpace::Xyz
        && match profile.color_space {
            DataColorSpace::Rgb => profile.is_matrix_shaper(),
            DataColorSpace::Gray => profile.gray_trc.is_some(),
            _ => false,
        }
}

/// Output classification shared by explicit region planning and execution.
/// Unlike ordinary packed output, restricted non-sRGB ICC input is transformed.
pub(crate) fn srgb8_output_layout(image: &Image<'_>) -> Result<J2kSrgb8Layout, J2kError> {
    image
        .srgb8_alpha_is_premultiplied()
        .map_err(J2kError::from_native_decode_error)?;
    let profile = image
        .primary_icc_profile()
        .or_else(|| match image.color_space() {
            ColorSpace::Icc { profile, .. } => Some(profile.as_slice()),
            _ => None,
        });
    let rgb =
        if let Some(profile) = profile {
            let rgb = match parse_restricted_profile(profile)?.color_space {
                DataColorSpace::Rgb => true,
                DataColorSpace::Gray => false,
                _ => return Err(j2k_core::Unsupported {
                    what: "decode_srgb8 supports restricted ICC RGB and monochrome input profiles",
                }
                .into()),
            };
            if image.color_space().num_channels() != if rgb { 3 } else { 1 } {
                return Err(j2k_core::Unsupported {
                    what: "restricted ICC profile channels do not match decoded samples",
                }
                .into());
            }
            rgb
        } else {
            match image.color_space() {
                ColorSpace::RGB => true,
                ColorSpace::Gray => false,
                _ => return Err(j2k_core::Unsupported {
                    what: "decode_srgb8 requires grayscale, RGB, or a restricted ICC input profile",
                }
                .into()),
            }
        };
    Ok(if image.has_alpha() {
        J2kSrgb8Layout::Rgba
    } else if rgb {
        J2kSrgb8Layout::Rgb
    } else {
        J2kSrgb8Layout::Gray
    })
}

/// Reconstruct only the native region and its required support window, then
/// apply the same normalization used by whole-image explicit sRGB output.
pub(crate) fn decode_image_region_srgb8<'a>(
    image: &Image<'a>,
    native_context: &mut j2k_native::DecoderContext<'a>,
    roi: Rect,
    retained_capacity: usize,
    control: &J2kDecodeControl,
) -> Result<J2kSrgb8Image, J2kError> {
    check_icc_working_control(image, control)?;
    let layout = srgb8_output_layout(image)?;
    control.check_output(expected_len((roi.w, roi.h), layout)?)?;
    let bitmap = native_context
        .with_straight_alpha_output(|context| {
            image.decode_region_with_context_and_retained_capacity(
                (roi.x, roi.y, roi.w, roi.h),
                context,
                retained_capacity,
            )
        })
        .map_err(J2kError::from_native_decode_error)?;
    let retained_image_bytes = checked_peak_bytes(
        retained_capacity,
        image
            .retained_allocation_bytes()
            .map_err(J2kError::from_native_decode_error)?,
        native_context
            .retained_allocation_bytes()
            .map_err(J2kError::from_native_decode_error)?,
    )?;
    normalize_bitmap(image, bitmap, retained_image_bytes, control)
        .map(|decoded| decoded.with_warnings(decode_warnings_for_image(image)))
}

fn decode_image_srgb8_planes<'a>(
    image: &Image<'a>,
    native_context: &mut j2k_native::DecoderContext<'a>,
    control: &J2kDecodeControl,
) -> Result<J2kSrgb8Image, J2kError> {
    let bitmap = native_context
        .with_straight_alpha_output(|context| image.decode_with_context(context))
        .map_err(J2kError::from_native_decode_error)?;
    let retained_image_bytes = checked_peak_bytes(
        image
            .retained_allocation_bytes()
            .map_err(J2kError::from_native_decode_error)?,
        native_context
            .retained_allocation_bytes()
            .map_err(J2kError::from_native_decode_error)?,
        0,
    )?;
    normalize_bitmap(image, bitmap, retained_image_bytes, control)
}

fn normalize_bitmap(
    image: &Image<'_>,
    mut bitmap: crate::backend::Bitmap,
    retained_image_bytes: usize,
    control: &J2kDecodeControl,
) -> Result<J2kSrgb8Image, J2kError> {
    control.check_cancelled()?;
    let primary_icc = image.primary_icc_profile();
    let dimensions = (bitmap.width, bitmap.height);
    let has_alpha = bitmap.has_alpha;
    let bitmap_profile_bytes = match &bitmap.color_space {
        ColorSpace::Icc { profile, .. } => profile.capacity(),
        _ => 0,
    };
    let bitmap_bytes = checked_peak_bytes(bitmap.data.capacity(), bitmap_profile_bytes, 0)?;
    let retained_bytes = checked_conversion_peak(retained_image_bytes, bitmap_bytes, control)?;

    // The native decode already unassociated alpha in source code values
    // when it could; otherwise the accepted eight-bit samples are
    // unassociated here, still before any colour transform.
    if image
        .srgb8_alpha_is_premultiplied()
        .map_err(J2kError::from_native_decode_error)?
        && !image
            .unassociates_alpha_in_source_values()
            .map_err(J2kError::from_native_decode_error)?
    {
        let channels = usize::from(bitmap.color_space.num_channels()) + 1;
        straighten_alpha(&mut bitmap.data, channels, control)?;
    }

    if let Some(profile) = primary_icc {
        return convert_icc_data(
            &bitmap.data,
            profile,
            dimensions,
            has_alpha,
            retained_bytes,
            control,
        );
    }

    match bitmap.color_space {
        ColorSpace::Gray if has_alpha => {
            expand_gray_alpha(&bitmap.data, dimensions, retained_bytes, control)
        }
        ColorSpace::Gray => finish(bitmap.data, dimensions, J2kSrgb8Layout::Gray),
        ColorSpace::RGB => finish(
            bitmap.data,
            dimensions,
            if has_alpha {
                J2kSrgb8Layout::Rgba
            } else {
                J2kSrgb8Layout::Rgb
            },
        ),
        ColorSpace::Icc { profile, .. } => convert_icc_data(
            &bitmap.data,
            &profile,
            dimensions,
            has_alpha,
            retained_bytes,
            control,
        ),
        ColorSpace::CMYK | ColorSpace::Unknown { .. } => Err(j2k_core::Unsupported {
            what: "decode_srgb8 requires grayscale, RGB, or a restricted ICC input profile",
        }
        .into()),
    }
}

// Undo association in source-profile code values before applying its transfer
// curve or colour matrix. At zero opacity the straight colour is defined as zero.
fn straighten_alpha(
    data: &mut [u8],
    channels: usize,
    control: &J2kDecodeControl,
) -> Result<(), J2kError> {
    if !matches!(channels, 2 | 4) || !data.len().is_multiple_of(channels) {
        return Err(j2k_core::Unsupported {
            what: "sRGB normalization requires gray-alpha or RGBA samples",
        }
        .into());
    }
    for chunk in data.chunks_mut(4096 * channels) {
        control.check_cancelled()?;
        for pixel in chunk.chunks_exact_mut(channels) {
            let alpha = u16::from(pixel[channels - 1]);
            for sample in &mut pixel[..channels - 1] {
                *sample = (u16::from(*sample) * 255 + alpha / 2)
                    .checked_div(alpha)
                    .unwrap_or(0)
                    .min(255) as u8;
            }
        }
    }
    Ok(())
}

fn convert_icc_data(
    source: &[u8],
    profile: &[u8],
    dimensions: (u32, u32),
    has_alpha: bool,
    retained_bytes: usize,
    control: &J2kDecodeControl,
) -> Result<J2kSrgb8Image, J2kError> {
    // Charge the colour-management state beside the decoded samples before
    // it is parsed; the output is then allocated beside both.
    let retained_bytes = checked_conversion_peak(
        retained_bytes,
        icc_conversion_working_bytes(profile.len()),
        control,
    )?;
    let source_profile = parse_restricted_profile(profile)?;
    if control.max_working_memory_bytes.is_some() && !has_bounded_icc_transform(&source_profile) {
        return Err(j2k_core::Unsupported {
            what: "working-memory limits require matrix/TRC or gray-TRC ICC profiles",
        }
        .into());
    }

    match source_profile.color_space {
        DataColorSpace::Rgb => transform_rgb(
            source,
            dimensions,
            has_alpha,
            retained_bytes,
            &source_profile,
            control,
        ),
        DataColorSpace::Gray => transform_gray(
            source,
            dimensions,
            has_alpha,
            retained_bytes,
            &source_profile,
            control,
        ),
        _ => Err(j2k_core::Unsupported {
            what: "decode_srgb8 supports restricted ICC RGB and monochrome input profiles",
        }
        .into()),
    }
}

fn parse_restricted_profile(profile: &[u8]) -> Result<ColorProfile, J2kError> {
    let profile = restricted_profile_tags(profile)?;
    let max_profile_size = profile
        .len()
        .checked_add(1)
        .ok_or_else(output_size_overflow)?;
    let source_profile = ColorProfile::new_from_slice_with_options(
        &profile,
        ParsingOptions {
            max_profile_size,
            max_allowed_clut_size: 0,
            max_allowed_trc_size: 65_536,
        },
    )
    .map_err(|error| map_profile_error(&error))?;
    reject_non_restricted_profile(&source_profile)?;
    Ok(source_profile)
}

/// Restricted RGB matrix/TRC profiles may use ordinary eight-bit output only
/// when both their linear matrix and their transformed ramps stay near sRGB.
/// Ramp checks alone miss negative crosstalk that clips on primary ramps and
/// cancels on neutral ramps, while moving mixed colours by many code values.
pub(in crate::decode) fn is_srgb_equivalent_icc_profile(profile: &[u8]) -> bool {
    let Ok(source) = parse_restricted_profile(profile) else {
        return false;
    };
    // Equivalence is decided from the matrix and tone curves, so only a
    // matrix/TRC profile through XYZ can qualify, and only its transform is
    // within the bounded colour-management path.
    if source.color_space != DataColorSpace::Rgb || !has_bounded_icc_transform(&source) {
        return false;
    }
    let srgb = ColorProfile::new_srgb();
    if !is_near_identity(&source.transform_matrix(&srgb).v) {
        return false;
    }
    let Ok(transform) =
        source.create_transform_8bit(Layout::Rgb, &srgb, Layout::Rgb, TransformOptions::default())
    else {
        return false;
    };
    let mut probe = [0_u8; 256 * 4 * 3];
    for (value, pixels) in (0..=u8::MAX).zip(probe.as_chunks_mut::<12>().0) {
        *pixels = [value, 0, 0, 0, value, 0, 0, 0, value, value, value, value];
    }
    let mut converted = [0_u8; 256 * 4 * 3];
    transform.transform(&probe, &mut converted).is_ok()
        && probe
            .iter()
            .zip(&converted)
            .all(|(&expected, &actual)| expected.abs_diff(actual) <= 1)
}

// One code value at the sRGB toe is about 3.0e-4 of linear light. Bound
// leakage below that and diagonal error near 0.1 code value at full intensity.
const SRGB_MATRIX_OFF_DIAGONAL_TOLERANCE: f64 = 2.5e-4;
const SRGB_MATRIX_DIAGONAL_TOLERANCE: f64 = 1e-3;

fn is_near_identity(matrix: &[[f64; 3]; 3]) -> bool {
    matrix.iter().enumerate().all(|(row, entries)| {
        let off_diagonal: f64 = entries
            .iter()
            .enumerate()
            .filter(|&(column, _)| column != row)
            .map(|(_, entry)| entry.abs())
            .sum();
        (entries[row] - 1.0).abs() <= SRGB_MATRIX_DIAGONAL_TOLERANCE
            && off_diagonal <= SRGB_MATRIX_OFF_DIAGONAL_TOLERANCE
    })
}

/// Description, copyright, viewing-condition, device and target text: never
/// read by a colour transform.
const ICC_TEXT_TAGS: [[u8; 4]; 6] = [*b"cprt", *b"desc", *b"vued", *b"dmnd", *b"dmdd", *b"targ"];

/// Lookup-table transforms, which restricted matrix/TRC profiles do not use.
const ICC_LUT_TAGS: [[u8; 4]; 15] = [
    *b"A2B0", *b"A2B1", *b"A2B2", *b"B2A0", *b"B2A1", *b"B2A2", *b"gamt", *b"D2B0", *b"D2B1",
    *b"D2B2", *b"D2B3", *b"B2D0", *b"B2D1", *b"B2D2", *b"B2D3",
];

/// A real profile has a few dozen tags; more is not a restricted profile.
const MAX_ICC_TAGS: usize = 256;

/// Copy `profile` keeping only tags a restricted transform can read.
///
/// The colour-management parser decodes every text tag, and one multi-
/// localized Unicode tag whose records all name the same string costs memory
/// in proportion to the square of its size (and the six text signatures may
/// all name one tag). Text never affects colour, so its tag-table entries are
/// dropped before parsing. Lookup-table tags are refused before their tables
/// are allocated, and duplicate signatures, which ICC forbids, are invalid.
/// Tag offsets are absolute, so the kept entries keep their data in place.
fn restricted_profile_tags(profile: &[u8]) -> Result<Vec<u8>, J2kError> {
    const HEADER_BYTES: usize = 128;
    const TAG_ENTRY_BYTES: usize = 12;
    let table = HEADER_BYTES + 4;
    let count = profile
        .get(HEADER_BYTES..table)
        .map(|bytes| u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize)
        .ok_or(J2kError::InvalidIccProfile)?;
    if count > MAX_ICC_TAGS {
        return Err(j2k_core::Unsupported {
            what: "decode_srgb8 supports ICC profiles of at most 256 tags",
        }
        .into());
    }
    let entries = profile
        .get(table..table + count * TAG_ENTRY_BYTES)
        .ok_or(J2kError::InvalidIccProfile)?;
    let mut copy = Vec::new();
    copy.try_reserve_exact(profile.len())
        .map_err(|_| BufferError::HostAllocationFailed {
            bytes: profile.len(),
            what: "restricted ICC profile",
        })?;
    copy.extend_from_slice(profile);
    let mut kept = 0_usize;
    let (entries, _) = entries.as_chunks::<TAG_ENTRY_BYTES>();
    for (index, entry) in entries.iter().enumerate() {
        let signature: [u8; 4] = [entry[0], entry[1], entry[2], entry[3]];
        if ICC_LUT_TAGS.contains(&signature) {
            return Err(j2k_core::Unsupported {
                what: "decode_srgb8 supports restricted matrix/TRC ICC profiles only",
            }
            .into());
        }
        if entries[..index]
            .iter()
            .any(|earlier| earlier[..4] == signature)
        {
            return Err(J2kError::InvalidIccProfile);
        }
        if ICC_TEXT_TAGS.contains(&signature) {
            continue;
        }
        let destination = table + kept * TAG_ENTRY_BYTES;
        copy[destination..destination + TAG_ENTRY_BYTES].copy_from_slice(entry);
        kept += 1;
    }
    let kept = u32::try_from(kept).map_err(|_| J2kError::InvalidIccProfile)?;
    copy[HEADER_BYTES..table].copy_from_slice(&kept.to_be_bytes());
    Ok(copy)
}

fn reject_non_restricted_profile(profile: &ColorProfile) -> Result<(), J2kError> {
    if profile.lut_a_to_b_perceptual.is_some()
        || profile.lut_a_to_b_colorimetric.is_some()
        || profile.lut_a_to_b_saturation.is_some()
        || profile.lut_b_to_a_perceptual.is_some()
        || profile.lut_b_to_a_colorimetric.is_some()
        || profile.lut_b_to_a_saturation.is_some()
        || profile.gamut.is_some()
    {
        return Err(j2k_core::Unsupported {
            what: "decode_srgb8 supports restricted matrix/TRC ICC profiles only",
        }
        .into());
    }
    Ok(())
}

fn transform_rgb(
    source: &[u8],
    dimensions: (u32, u32),
    has_alpha: bool,
    retained_bytes: usize,
    source_profile: &ColorProfile,
    control: &J2kDecodeControl,
) -> Result<J2kSrgb8Image, J2kError> {
    let cms_layout = if has_alpha { Layout::Rgba } else { Layout::Rgb };
    let destination = ColorProfile::new_srgb();
    transform_icc(
        source,
        dimensions,
        retained_bytes,
        (source_profile, cms_layout),
        (&destination, cms_layout),
        control,
    )
}

fn transform_gray(
    source: &[u8],
    dimensions: (u32, u32),
    has_alpha: bool,
    retained_bytes: usize,
    source_profile: &ColorProfile,
    control: &J2kDecodeControl,
) -> Result<J2kSrgb8Image, J2kError> {
    if has_alpha {
        let destination = ColorProfile::new_srgb();
        return transform_icc(
            source,
            dimensions,
            retained_bytes,
            (source_profile, Layout::GrayAlpha),
            (&destination, Layout::Rgba),
            control,
        );
    }

    let mut destination = ColorProfile::new_gray_with_gamma(1.0);
    destination.gray_trc = ColorProfile::new_srgb().red_trc;
    transform_icc(
        source,
        dimensions,
        retained_bytes,
        (source_profile, Layout::Gray),
        (&destination, Layout::Gray),
        control,
    )
}

fn transform_icc(
    source: &[u8],
    dimensions: (u32, u32),
    retained_bytes: usize,
    (source_profile, source_layout): (&ColorProfile, Layout),
    (destination_profile, destination_layout): (&ColorProfile, Layout),
    control: &J2kDecodeControl,
) -> Result<J2kSrgb8Image, J2kError> {
    let source_len = (dimensions.0 as usize)
        .checked_mul(dimensions.1 as usize)
        .and_then(|pixels| pixels.checked_mul(source_layout.channels()))
        .ok_or_else(output_size_overflow)?;
    if source.len() != source_len {
        return Err(j2k_core::Unsupported {
            what: "restricted ICC profile channels do not match decoded samples",
        }
        .into());
    }
    let output_layout = match destination_layout {
        Layout::Gray => J2kSrgb8Layout::Gray,
        Layout::Rgb => J2kSrgb8Layout::Rgb,
        Layout::Rgba => J2kSrgb8Layout::Rgba,
        _ => {
            return Err(J2kError::InternalInvariant {
                what: "restricted ICC transform has a non-sRGB destination layout",
            });
        }
    };
    control.check_output(expected_len(dimensions, output_layout)?)?;
    let mut output = allocate_output(dimensions, output_layout, retained_bytes, control)?;
    let transform = source_profile
        .create_transform_8bit(
            source_layout,
            destination_profile,
            destination_layout,
            TransformOptions::default(),
        )
        .map_err(|error| map_transform_error(&error))?;
    for (input, destination) in source
        .chunks(4096 * source_layout.channels())
        .zip(output.chunks_mut(4096 * destination_layout.channels()))
    {
        control.check_cancelled()?;
        transform
            .transform(input, destination)
            .map_err(|error| map_transform_error(&error))?;
    }
    control.check_cancelled()?;
    finish(output, dimensions, output_layout)
}

fn expand_gray_alpha(
    source: &[u8],
    dimensions: (u32, u32),
    retained_bytes: usize,
    control: &J2kDecodeControl,
) -> Result<J2kSrgb8Image, J2kError> {
    control.check_output(expected_len(dimensions, J2kSrgb8Layout::Rgba)?)?;
    let mut output = allocate_output(dimensions, J2kSrgb8Layout::Rgba, retained_bytes, control)?;
    for (input, destination) in source.chunks(4096 * 2).zip(output.chunks_mut(4096 * 4)) {
        control.check_cancelled()?;
        for (gray, pixel) in input
            .as_chunks::<2>()
            .0
            .iter()
            .zip(destination.as_chunks_mut::<4>().0)
        {
            *pixel = [gray[0], gray[0], gray[0], gray[1]];
        }
    }
    control.check_cancelled()?;
    finish(output, dimensions, J2kSrgb8Layout::Rgba)
}

fn allocate_output(
    dimensions: (u32, u32),
    layout: J2kSrgb8Layout,
    retained_bytes: usize,
    control: &J2kDecodeControl,
) -> Result<Vec<u8>, J2kError> {
    let len = expected_len(dimensions, layout)?;
    checked_conversion_peak(retained_bytes, len, control)?;
    let output = try_host_vec_filled(len, 0_u8).map_err(host_allocation_error)?;
    checked_conversion_peak(retained_bytes, output.capacity(), control)?;
    Ok(output)
}

fn checked_conversion_peak(
    retained: usize,
    additional: usize,
    control: &J2kDecodeControl,
) -> Result<usize, J2kError> {
    let requested = retained
        .checked_add(additional)
        .ok_or_else(output_size_overflow)?;
    if let Some(cap) = control.max_working_memory_bytes {
        let cap = cap.get().min(j2k_native::DEFAULT_MAX_DECODE_BYTES);
        if requested > cap {
            return Err(J2kError::from_native_decode_error(
                j2k_native::DecodeError::AllocationTooLarge {
                    what: SRGB8_OUTPUT_WHAT,
                    requested,
                    cap,
                },
            ));
        }
    }
    checked_peak_bytes(requested, 0, 0)
}

fn finish(
    data: Vec<u8>,
    dimensions: (u32, u32),
    layout: J2kSrgb8Layout,
) -> Result<J2kSrgb8Image, J2kError> {
    let expected = expected_len(dimensions, layout)?;
    if data.len() != expected {
        return Err(J2kError::InternalInvariant {
            what: "normalized sRGB output length does not match its layout",
        });
    }
    Ok(J2kSrgb8Image {
        dimensions,
        layout,
        data,
        warnings: Vec::new(),
    })
}

fn expected_len(dimensions: (u32, u32), layout: J2kSrgb8Layout) -> Result<usize, J2kError> {
    (dimensions.0 as usize)
        .checked_mul(dimensions.1 as usize)
        .and_then(|pixels| pixels.checked_mul(layout.channels()))
        .ok_or_else(output_size_overflow)
}

fn checked_peak_bytes(first: usize, second: usize, third: usize) -> Result<usize, J2kError> {
    let requested = first
        .checked_add(second)
        .and_then(|bytes| bytes.checked_add(third))
        .ok_or_else(output_size_overflow)?;
    ensure_allocation_within_cap(
        requested,
        DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        SRGB8_OUTPUT_WHAT,
    )
    .map_err(Into::into)
}

fn output_size_overflow() -> J2kError {
    BufferError::SizeOverflow {
        what: SRGB8_OUTPUT_WHAT,
    }
    .into()
}

fn host_allocation_error(error: HostAllocationError) -> J2kError {
    BufferError::HostAllocationFailed {
        bytes: error.requested_bytes(),
        what: SRGB8_OUTPUT_WHAT,
    }
    .into()
}

fn map_profile_error(error: &CmsError) -> J2kError {
    match error {
        CmsError::OutOfMemory(bytes) => BufferError::HostAllocationFailed {
            bytes: *bytes,
            what: "restricted ICC profile",
        }
        .into(),
        _ => J2kError::InvalidIccProfile,
    }
}

fn map_transform_error(error: &CmsError) -> J2kError {
    match error {
        CmsError::OutOfMemory(bytes) => BufferError::HostAllocationFailed {
            bytes: *bytes,
            what: "restricted ICC transform",
        }
        .into(),
        _ => j2k_core::Unsupported {
            what: "restricted ICC profile cannot be transformed to sRGB",
        }
        .into(),
    }
}
