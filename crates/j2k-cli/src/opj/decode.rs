// SPDX-License-Identifier: MIT OR Apache-2.0

//! Decode one `opj_decompress` request through the `j2k` facade.
//!
//! The facade plans and decodes every region; this module only maps the
//! `-d` rectangle onto the facade's request, chooses the output
//! interpretation and keeps the samples `opj_decompress` would write.

use j2k::{
    J2kComponentInfo, J2kDecodedNativeComponents, J2kDecoder, J2kError, J2kNativeComponentPlane,
    J2kReferenceGrid, J2kScratchPool, J2kSrgb8Layout, NativeSampleType, Rect,
};

use super::args::DecodeArea;

/// Decoded samples ready for an output writer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DecodedImage {
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// Interleaved channels: 1 gray, 3 RGB or 4 RGBA.
    pub(crate) channels: usize,
    /// Significant bits per sample.
    pub(crate) precision: u8,
    pub(crate) signed: bool,
    /// Row-major interleaved sample values.
    pub(crate) samples: Vec<i32>,
}

/// Decode `area` (the whole image when `None`) after discarding `reduce`
/// resolution levels. Notes collects warnings for stderr.
///
/// Images with native Gray, RGB or RGBA samples keep their precision and
/// signedness; palette, ICC, sYCC, channel-definition and other colour
/// interpretations take the facade's explicit 8-bit sRGB output.
pub(crate) fn decode(
    bytes: &[u8],
    area: Option<DecodeArea>,
    reduce: u8,
    notes: &mut Vec<String>,
) -> Result<DecodedImage, String> {
    let header = |error: J2kError| format!("failed to read the header: {error}");
    check_components(
        &J2kDecoder::inspect_support(bytes)
            .map_err(header)?
            .components,
    )?;
    let mut decoder = J2kDecoder::new(bytes).map_err(header)?;
    let max_reduction = decoder
        .max_reduction_levels()
        .map_err(|error| format!("failed to read the tile headers: {error}"))?;
    if reduce > max_reduction {
        return Err(format!(
            "reduce factor {reduce} exceeds the {max_reduction} resolution levels every tile and \
             component can discard"
        ));
    }
    let grid = decoder.reference_grid();
    let region = source_region(grid, area, notes)?;
    let window = grid
        .reduced_area(region, reduce)
        .filter(|window| window.w > 0 && window.h > 0)
        .ok_or_else(|| {
            format!("the decoded area has no samples after discarding {reduce} resolution levels")
        })?;
    let (covering, image) = match decode_native(&mut decoder, region, reduce, notes) {
        Ok(native_image) => native_image,
        Err(native) => decode_srgb8(&mut decoder, region, reduce, notes)
            .map_err(|srgb8| format!("{native}; as 8-bit sRGB: {srgb8}"))?,
    };
    crop(image, covering, window)
}

/// Refuse the images `opj_decompress` writes component by component or not
/// at all: samples deeper than 16 bits, which its PNM, PNG and TIFF writers
/// refuse, and components that differ in sampling, precision or signedness,
/// which its PNM writer splits into one file per component and its PNG and
/// TIFF writers refuse.
fn check_components(components: &[J2kComponentInfo]) -> Result<(), String> {
    let Some(first) = components.first() else {
        return Err("the image has no components".into());
    };
    if let Some(deep) = components.iter().find(|component| component.bit_depth > 16) {
        return Err(format!(
            "precision {} is larger than 16: refused",
            deep.bit_depth
        ));
    }
    let shape = |component: &J2kComponentInfo| {
        (
            component.x_rsiz,
            component.y_rsiz,
            component.bit_depth,
            component.signed,
        )
    };
    if components
        .iter()
        .any(|component| shape(component) != shape(first))
    {
        return Err(
            "components differ in sampling, precision or signedness; this command writes \
             only images whose components share them"
                .into(),
        );
    }
    Ok(())
}

/// The full-resolution image rectangle `-d` selects, clamped to the image
/// area with a warning where `opj_set_decode_area` clamps it.
fn source_region(
    grid: J2kReferenceGrid,
    area: Option<DecodeArea>,
    notes: &mut Vec<String>,
) -> Result<Rect, String> {
    let (width, height) = grid.image_size;
    let Some(area) = area else {
        return Ok(Rect {
            x: 0,
            y: 0,
            w: width,
            h: height,
        });
    };
    let (origin_x, origin_y) = grid.image_origin;
    // The image area ends at Xsiz and Ysiz, which SIZ stores as 32-bit values.
    let extent = grid.extent();
    let (Ok(extent_x), Ok(extent_y)) = (u32::try_from(extent.0), u32::try_from(extent.1)) else {
        return Err("the image area ends beyond the 32-bit reference grid".into());
    };
    let (x0, x1) = area_axis("x", area.x0, area.x1, origin_x, extent_x, notes)?;
    let (y0, y1) = area_axis("y", area.y0, area.y1, origin_y, extent_y, notes)?;
    if x1 <= x0 || y1 <= y0 {
        return Err(format!(
            "the decoded area {},{},{},{} is empty",
            area.x0, area.y0, area.x1, area.y1
        ));
    }
    Ok(Rect {
        x: x0 - origin_x,
        y: y0 - origin_y,
        w: x1 - x0,
        h: y1 - y0,
    })
}

fn area_axis(
    axis: &str,
    start: u32,
    end: u32,
    origin: u32,
    extent: u32,
    notes: &mut Vec<String>,
) -> Result<(u32, u32), String> {
    if start > extent {
        return Err(format!(
            "the decoded area starts at {axis}={start}, after the image area ends at {extent}"
        ));
    }
    if end == 0 || end < origin {
        return Err(format!(
            "the decoded area ends at {axis}={end}, before the image area starts at {origin}"
        ));
    }
    let start = if start < origin {
        notes.push(format!(
            "warning: the decoded area starts at {axis}={start}, before the image area; using {origin}"
        ));
        origin
    } else {
        start
    };
    let end = if end > extent {
        notes.push(format!(
            "warning: the decoded area ends at {axis}={end}, after the image area; using {extent}"
        ));
        extent
    } else {
        end
    };
    Ok((start, end))
}

/// Native Gray, RGB or RGBA samples, and the reduced rectangle they cover.
fn decode_native(
    decoder: &mut J2kDecoder<'_>,
    region: Rect,
    reduce: u8,
    notes: &mut Vec<String>,
) -> Result<(Rect, DecodedImage), String> {
    let plan = decoder
        .native_region_decode_plan(region, reduce)
        .map_err(|error| error.to_string())?;
    let samples = &plan.samples;
    if samples.precision < 8 || (samples.signed && samples.precision == 8) {
        // The packed 8- and 16-bit layouts widen these samples across their
        // range; the component planes keep them as they are.
        return decode_planes(decoder, region, reduce, samples.color.channels());
    }
    let layout = plan.region;
    let mut output = vec![0_u8; layout.buffer_bytes];
    let outcome = decoder
        .decode_region_scaled_pow2_into(
            &mut J2kScratchPool::new(),
            &mut output,
            layout.row_bytes,
            layout.format,
            region,
            reduce,
        )
        .map_err(|error| format!("failed to decode image: {error}"))?;
    notes.extend(
        outcome
            .warnings
            .iter()
            .map(|warning| format!("warning: {warning}")),
    );
    let samples = match plan.samples.sample_type {
        NativeSampleType::U8 => output.iter().copied().map(i32::from).collect(),
        NativeSampleType::U16 => output
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&pair| i32::from(u16::from_ne_bytes(pair)))
            .collect(),
        NativeSampleType::I16 => output
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&pair| i32::from(i16::from_ne_bytes(pair)))
            .collect(),
        _ => return Err("unsupported native sample type".into()),
    };
    let covering = layout.output_region;
    Ok((
        covering,
        DecodedImage {
            width: covering.w,
            height: covering.h,
            channels: plan.samples.color.channels(),
            precision: plan.samples.precision,
            signed: plan.samples.signed,
            samples,
        },
    ))
}

/// Native component planes of samples shorter than 8 bits or signed 8-bit,
/// and the reduced rectangle they cover. The facade decodes these planes for
/// a region at full resolution and for the whole image at a reduction, so a
/// reduced region decodes the whole reduced image and the caller crops it.
fn decode_planes(
    decoder: &mut J2kDecoder<'_>,
    region: Rect,
    reduce: u8,
    channels: usize,
) -> Result<(Rect, DecodedImage), String> {
    let components = if reduce == 0 {
        decoder.decode_native_region_components(region)
    } else {
        decoder.decode_native_components_at_reduction(reduce)
    }
    .map_err(|error| format!("failed to decode image: {error}"))?;
    let (width, height) = components.dimensions();
    let covering = if reduce == 0 {
        region
    } else {
        Rect {
            x: 0,
            y: 0,
            w: width,
            h: height,
        }
    };
    let image = interleave(&components, channels)?;
    if (image.width, image.height) != (covering.w, covering.h) {
        return Err("decoded component planes disagree with the decode area".into());
    }
    Ok((covering, image))
}

/// Interleave `channels` same-shaped planes of `decoded`.
fn interleave(
    decoded: &J2kDecodedNativeComponents,
    channels: usize,
) -> Result<DecodedImage, String> {
    let planes = decoded.planes();
    let Some(first) = planes.first() else {
        return Err("the decoded image has no component planes".into());
    };
    let shape = |plane: &J2kNativeComponentPlane| {
        (
            plane.dimensions(),
            plane.bit_depth(),
            plane.signed(),
            plane.bytes_per_sample(),
        )
    };
    if planes.len() != channels || planes.iter().any(|plane| shape(plane) != shape(first)) {
        return Err("unsupported native component planes".into());
    }
    let values = planes
        .iter()
        .map(|plane| plane_samples(plane.data(), plane.bytes_per_sample(), plane.signed()))
        .collect::<Result<Vec<_>, _>>()?;
    let (width, height) = first.dimensions();
    let count = width as usize * height as usize;
    if values.iter().any(|plane| plane.len() != count) {
        return Err("decoded component planes disagree with their dimensions".into());
    }
    let samples = (0..count)
        .flat_map(|index| values.iter().map(move |plane| plane[index]))
        .collect();
    Ok(DecodedImage {
        width,
        height,
        channels,
        precision: first.bit_depth(),
        signed: first.signed(),
        samples,
    })
}

/// The little-endian samples of one plane, `bytes` bytes each.
fn plane_samples(data: &[u8], bytes: u8, signed: bool) -> Result<Vec<i32>, String> {
    Ok(match (bytes, signed) {
        (1, false) => data.iter().copied().map(i32::from).collect(),
        (1, true) => data
            .iter()
            .map(|&byte| i32::from(i8::from_le_bytes([byte])))
            .collect(),
        (2, false) => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&pair| i32::from(u16::from_le_bytes(pair)))
            .collect(),
        (2, true) => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&pair| i32::from(i16::from_le_bytes(pair)))
            .collect(),
        _ => return Err(format!("unsupported {bytes}-byte component samples")),
    })
}

/// The facade's explicit 8-bit sRGB output, and the reduced rectangle it covers.
fn decode_srgb8(
    decoder: &mut J2kDecoder<'_>,
    region: Rect,
    reduce: u8,
    notes: &mut Vec<String>,
) -> Result<(Rect, DecodedImage), String> {
    let covering = decoder
        .srgb8_region_decode_plan(region, reduce)
        .map_err(|error| error.to_string())?
        .output_region;
    let image = decoder
        .decode_region_srgb8_at_reduction(region, reduce)
        .map_err(|error| format!("failed to decode image: {error}"))?;
    notes.extend(
        image
            .warnings()
            .iter()
            .map(|warning| format!("warning: {warning}")),
    );
    let channels = match image.layout() {
        J2kSrgb8Layout::Gray => 1,
        J2kSrgb8Layout::Rgb => 3,
        J2kSrgb8Layout::Rgba => 4,
        _ => return Err("unsupported sRGB output layout".into()),
    };
    if image.dimensions() != (covering.w, covering.h) {
        return Err("sRGB output disagrees with its plan".into());
    }
    Ok((
        covering,
        DecodedImage {
            width: covering.w,
            height: covering.h,
            channels,
            precision: 8,
            signed: false,
            samples: image.data().iter().copied().map(i32::from).collect(),
        },
    ))
}

/// Keep `window` of `image`, which covers `covering`; both are reduced image
/// rectangles, and the facade's covering contains the decode area.
fn crop(image: DecodedImage, covering: Rect, window: Rect) -> Result<DecodedImage, String> {
    if window == covering {
        return Ok(image);
    }
    let left = window.x.checked_sub(covering.x);
    let top = window.y.checked_sub(covering.y);
    let (Some(left), Some(top)) = (left, top) else {
        return Err("decoded samples do not cover the decode area".into());
    };
    if left + window.w > covering.w || top + window.h > covering.h {
        return Err("decoded samples do not cover the decode area".into());
    }
    let channels = image.channels;
    let row = covering.w as usize * channels;
    let span = window.w as usize * channels;
    let first = left as usize * channels;
    let samples = image
        .samples
        .chunks_exact(row)
        .skip(top as usize)
        .take(window.h as usize)
        .flat_map(|samples| &samples[first..first + span])
        .copied()
        .collect();
    Ok(DecodedImage {
        width: window.w,
        height: window.h,
        samples,
        ..image
    })
}

#[cfg(test)]
mod tests {
    use super::{check_components, crop, plane_samples, source_region, DecodedImage};
    use crate::opj::args::DecodeArea;
    use j2k::{J2kComponentInfo, J2kDecoder, Rect};
    use j2k_test_support::OPENJPEG_ORIGIN_23X17_AT_3_5;

    fn component(bit_depth: u8, signed: bool, sampling: (u8, u8)) -> J2kComponentInfo {
        J2kComponentInfo {
            bit_depth,
            signed,
            x_rsiz: sampling.0,
            y_rsiz: sampling.1,
        }
    }

    #[test]
    fn components_must_share_sampling_precision_and_sign_up_to_16_bits() {
        let rgb = [component(12, false, (1, 1)); 3];
        assert_eq!(check_components(&rgb), Ok(()));
        let deep = [component(29, false, (1, 1))];
        let error = check_components(&deep).expect_err("29-bit");
        assert!(error.contains("precision 29 is larger than 16"), "{error}");
        for odd in [
            component(8, false, (2, 2)),
            component(12, false, (1, 1)),
            component(8, true, (1, 1)),
        ] {
            let mixed = [component(8, false, (1, 1)), odd];
            let error = check_components(&mixed).expect_err("mixed");
            assert!(error.contains("components differ"), "{error}");
        }
        assert!(check_components(&[]).is_err());
    }

    #[test]
    fn plane_samples_read_little_endian_two_s_complement() {
        assert_eq!(plane_samples(&[0, 255], 1, false), Ok(vec![0, 255]));
        assert_eq!(
            plane_samples(&[0x80, 0xFF, 0x7F], 1, true),
            Ok(vec![-128, -1, 127])
        );
        assert_eq!(plane_samples(&[0x34, 0x12], 2, false), Ok(vec![0x1234]));
        assert_eq!(plane_samples(&[0x00, 0x80], 2, true), Ok(vec![-32768]));
        assert!(plane_samples(&[0; 4], 4, false).is_err());
    }

    const fn area(x0: u32, y0: u32, x1: u32, y1: u32) -> DecodeArea {
        DecodeArea { x0, y0, x1, y1 }
    }

    #[test]
    fn decode_areas_use_reference_grid_coordinates() {
        let grid = J2kDecoder::inspect_reference_grid(OPENJPEG_ORIGIN_23X17_AT_3_5).expect("grid");
        let mut notes = Vec::new();
        assert_eq!(
            source_region(grid, None, &mut notes),
            Ok(Rect {
                x: 0,
                y: 0,
                w: 23,
                h: 17
            })
        );
        // The image area spans x 3..26 and y 5..22 of the reference grid.
        assert_eq!(
            source_region(grid, Some(area(4, 6, 10, 9)), &mut notes),
            Ok(Rect {
                x: 1,
                y: 1,
                w: 6,
                h: 3
            })
        );
        assert_eq!(notes, Vec::<String>::new());
        assert_eq!(
            source_region(grid, Some(area(0, 0, 40, 40)), &mut notes),
            Ok(Rect {
                x: 0,
                y: 0,
                w: 23,
                h: 17
            })
        );
        assert_eq!(notes.len(), 4, "{notes:?}");
    }

    #[test]
    fn decode_areas_outside_or_empty_are_errors() {
        let grid = J2kDecoder::inspect_reference_grid(OPENJPEG_ORIGIN_23X17_AT_3_5).expect("grid");
        let mut notes = Vec::new();
        for (bad, needle) in [
            (area(27, 5, 30, 9), "after the image area"),
            (area(3, 5, 2, 9), "before the image area"),
            (area(3, 5, 0, 9), "before the image area"),
            (area(8, 5, 8, 9), "is empty"),
            (area(9, 5, 8, 9), "is empty"),
        ] {
            let error = source_region(grid, Some(bad), &mut notes).expect_err(needle);
            assert!(error.contains(needle), "{error}");
        }
    }

    #[test]
    fn crop_keeps_the_window_inside_the_covering() {
        let image = DecodedImage {
            width: 3,
            height: 2,
            channels: 1,
            precision: 8,
            signed: false,
            samples: vec![1, 2, 3, 4, 5, 6],
        };
        let covering = Rect {
            x: 4,
            y: 7,
            w: 3,
            h: 2,
        };
        let window = Rect {
            x: 5,
            y: 8,
            w: 2,
            h: 1,
        };
        let cropped = crop(image.clone(), covering, window).expect("crop");
        assert_eq!((cropped.width, cropped.height), (2, 1));
        assert_eq!(cropped.samples, vec![5, 6]);
        assert_eq!(crop(image.clone(), covering, covering), Ok(image.clone()));
        let outside = Rect {
            x: 3,
            y: 7,
            w: 2,
            h: 1,
        };
        assert!(crop(image, covering, outside).is_err());
    }
}
