// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integer RGB output for JP2 files tagged with an IEC 61966-2-1 sRGB ICC
//! profile (issue #321).

use j2k::{
    wrap_j2k_codestream, J2kDecoder, J2kError, J2kFileColorSpec, J2kFileWrapOptions,
    J2kRowDecodeOptions, J2kScratchPool, PixelFormat, Rect, RowSink,
};
use j2k_core::{Colorspace, Unsupported};
use j2k_native::{encode, EncodeOptions};
use moxcms::{ColorProfile, ToneReprCurve, Xyzd};

const WIDTH: u32 = 16;
const HEIGHT: u32 = 16;

fn rgb_codestream() -> Vec<u8> {
    let pixels: Vec<u8> = (0..WIDTH * HEIGHT * 3)
        .map(|i| u8::try_from((i * 37 + i / 7) % 256).expect("byte"))
        .collect();
    encode(
        &pixels,
        WIDTH,
        HEIGHT,
        3,
        8,
        false,
        &EncodeOptions {
            reversible: true,
            ..EncodeOptions::default()
        },
    )
    .expect("encode RGB fixture")
}

fn jp2(color: J2kFileColorSpec<'_>) -> Vec<u8> {
    wrap_j2k_codestream(
        &rgb_codestream(),
        J2kFileWrapOptions::jp2().with_color(color),
    )
    .expect("wrap JP2 fixture")
}

fn enumerated_srgb_jp2() -> Vec<u8> {
    jp2(J2kFileColorSpec::Enumerated(Colorspace::SRgb))
}

fn icc_jp2(profile: &ColorProfile) -> Vec<u8> {
    let profile = profile.encode().expect("encode ICC profile");
    jp2(J2kFileColorSpec::IccProfile(&profile))
}

/// sRGB with sampled `curv` tables, as in the widely distributed
/// "sRGB IEC61966-2.1" profile, rather than moxcms's parametric curves.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "linear sRGB values lie in [0, 1], so scaled samples fit u16"
)]
fn sampled_trc_srgb_profile() -> ColorProfile {
    let table: Vec<u16> = (0..1024_u32)
        .map(|i| {
            let encoded = f64::from(i) / 1023.0;
            let linear = if encoded <= 0.040_45 {
                encoded / 12.92
            } else {
                ((encoded + 0.055) / 1.055).powf(2.4)
            };
            (linear * 65535.0).round() as u16
        })
        .collect();
    let mut profile = ColorProfile::new_srgb();
    profile.red_trc = Some(ToneReprCurve::Lut(table.clone()));
    profile.green_trc = Some(ToneReprCurve::Lut(table.clone()));
    profile.blue_trc = Some(ToneReprCurve::Lut(table));
    profile.cicp = None;
    profile
}

fn region(bytes: &[u8], fmt: PixelFormat, roi: Rect) -> Result<Vec<u8>, j2k::J2kError> {
    let row_bytes = roi.w as usize * fmt.bytes_per_pixel();
    let stride = row_bytes + 5;
    let mut out = vec![0xA5_u8; stride * roi.h as usize];
    J2kDecoder::new(bytes)?.decode_region_into(
        &mut J2kScratchPool::new(),
        &mut out,
        stride,
        fmt,
        roi,
    )?;
    let mut packed = Vec::new();
    for row in out.chunks_exact(stride) {
        packed.extend_from_slice(&row[..row_bytes]);
        assert!(row[row_bytes..].iter().all(|&byte| byte == 0xA5));
    }
    Ok(packed)
}

fn scaled_region(bytes: &[u8], fmt: PixelFormat, roi: Rect, levels: u8) -> Vec<u8> {
    let scaled_w = (roi.x + roi.w).div_ceil(1 << levels) - roi.x / (1 << levels);
    let scaled_h = (roi.y + roi.h).div_ceil(1 << levels) - roi.y / (1 << levels);
    let stride = scaled_w as usize * fmt.bytes_per_pixel();
    let mut out = vec![0_u8; stride * scaled_h as usize];
    J2kDecoder::new(bytes)
        .expect("decoder")
        .decode_region_scaled_pow2_into(
            &mut J2kScratchPool::new(),
            &mut out,
            stride,
            fmt,
            roi,
            levels,
        )
        .expect("scaled region decode");
    out
}

fn whole(bytes: &[u8], fmt: PixelFormat) -> Vec<u8> {
    let stride = WIDTH as usize * fmt.bytes_per_pixel();
    let mut out = vec![0_u8; stride * HEIGHT as usize];
    J2kDecoder::new(bytes)
        .expect("decoder")
        .decode_into(&mut out, stride, fmt)
        .expect("whole-image decode");
    out
}

const ROI: Rect = Rect {
    x: 3,
    y: 5,
    w: 9,
    h: 7,
};

#[test]
fn srgb_icc_region_rgb8_and_rgba8_match_enumerated_srgb() {
    let reference = enumerated_srgb_jp2();
    for profile in [ColorProfile::new_srgb(), sampled_trc_srgb_profile()] {
        let tagged = icc_jp2(&profile);
        for fmt in [PixelFormat::Rgb8, PixelFormat::Rgba8] {
            assert_eq!(
                region(&tagged, fmt, ROI).expect("sRGB ICC region decode"),
                region(&reference, fmt, ROI).expect("enumerated sRGB region decode"),
                "{fmt:?}"
            );
        }
    }
}

#[test]
fn srgb_icc_scaled_region_matches_enumerated_srgb() {
    let reference = enumerated_srgb_jp2();
    let tagged = icc_jp2(&ColorProfile::new_srgb());
    for fmt in [PixelFormat::Rgb8, PixelFormat::Rgba8] {
        for levels in [1, 2] {
            assert_eq!(
                scaled_region(&tagged, fmt, ROI, levels),
                scaled_region(&reference, fmt, ROI, levels),
                "{fmt:?} at {levels} levels"
            );
        }
    }
}

#[test]
fn srgb_icc_region_plan_matches_supported_output_and_keeps_native16_unsupported() {
    let tagged = icc_jp2(&sampled_trc_srgb_profile());
    let mut decoder = J2kDecoder::new(&tagged).expect("decoder");
    for levels in [0, 1, 2] {
        for fmt in [PixelFormat::Rgb8, PixelFormat::Rgba8] {
            let plan = decoder
                .region_decode_plan(fmt, ROI, levels)
                .expect("sRGB ICC plan");
            assert_eq!(plan.format, fmt);
            assert_eq!(
                plan.row_bytes,
                plan.output_region.w as usize * fmt.bytes_per_pixel()
            );
            assert_eq!(
                scaled_region(&tagged, fmt, ROI, levels).len(),
                plan.buffer_bytes
            );
        }
    }
    for fmt in [PixelFormat::Rgb16, PixelFormat::Rgba16] {
        assert!(matches!(
            decoder.region_decode_plan(fmt, ROI, 0),
            Err(J2kError::Unsupported(_))
        ));
        assert!(matches!(
            region(&tagged, fmt, ROI),
            Err(J2kError::Unsupported(_))
        ));
    }
}

#[test]
fn srgb_icc_whole_image_rgb8_matches_enumerated_srgb_and_decode_srgb8() {
    let reference = enumerated_srgb_jp2();
    let tagged = icc_jp2(&ColorProfile::new_srgb());
    let rgb = whole(&tagged, PixelFormat::Rgb8);
    assert_eq!(rgb, whole(&reference, PixelFormat::Rgb8));
    assert_eq!(
        whole(&tagged, PixelFormat::Rgba8),
        whole(&reference, PixelFormat::Rgba8)
    );
    let normalized = J2kDecoder::new(&tagged)
        .expect("decoder")
        .decode_srgb8()
        .expect("decode_srgb8");
    assert!(rgb
        .iter()
        .zip(normalized.data())
        .all(|(&raw, &converted)| raw.abs_diff(converted) <= 1));
}

#[derive(Default)]
struct CollectRows(Vec<u8>);

impl RowSink<u8> for CollectRows {
    type Error = J2kError;

    fn write_row(&mut self, _y: u32, row: &[u8]) -> Result<(), Self::Error> {
        self.0.extend_from_slice(row);
        Ok(())
    }
}

fn bounded_rows(bytes: &[u8], rows_per_stripe: u32) -> Vec<u8> {
    let mut sink = CollectRows::default();
    J2kDecoder::new(bytes)
        .expect("decoder")
        .decode_rows_u8_bounded(&mut sink, J2kRowDecodeOptions::new(rows_per_stripe))
        .expect("bounded row decode");
    sink.0
}

#[test]
fn srgb_icc_bounded_rows_match_enumerated_srgb() {
    let reference = enumerated_srgb_jp2();
    let tagged = icc_jp2(&ColorProfile::new_srgb());
    for rows_per_stripe in [1, 5, HEIGHT] {
        assert_eq!(
            bounded_rows(&tagged, rows_per_stripe),
            bounded_rows(&reference, rows_per_stripe),
            "{rows_per_stripe} rows per stripe"
        );
    }
}

#[test]
fn non_srgb_icc_region_rgb8_remains_unsupported() {
    let unsupported = j2k::J2kError::from(Unsupported {
        what: "backend color space cannot be mapped to requested 8-bit pixel format",
    });
    for profile in [
        ColorProfile::new_display_p3(),
        ColorProfile::new_adobe_rgb(),
        ColorProfile::new_pro_photo_rgb(),
    ] {
        let tagged = icc_jp2(&profile);
        assert_eq!(
            region(&tagged, PixelFormat::Rgb8, ROI).expect_err("non-sRGB ICC"),
            unsupported
        );
    }
}

/// Near-sRGB profile whose linear matrix to sRGB is
/// `[[1.01, -0.01, 0], [0, 1, 0], [0, 0, 1]]`. Its red, green, blue, and
/// neutral 8-bit ramps all convert to sRGB within one code value, because
/// the negative term clips to zero on a green ramp and cancels on neutrals,
/// yet it moves the red of (30, 255, 0) to 10.
fn hidden_crosstalk_profile() -> ColorProfile {
    let srgb = ColorProfile::new_srgb();
    let (red, green) = (srgb.red_colorant, srgb.green_colorant);
    let mut profile = ColorProfile::new_srgb();
    profile.red_colorant = Xyzd {
        x: red.x * 1.01,
        y: red.y * 1.01,
        z: red.z * 1.01,
    };
    profile.green_colorant = Xyzd {
        x: green.x - 0.01 * red.x,
        y: green.y - 0.01 * red.y,
        z: green.z - 0.01 * red.z,
    };
    profile.cicp = None;
    profile
}

#[test]
fn near_srgb_icc_with_mixed_colour_crosstalk_remains_unsupported() {
    let profile = hidden_crosstalk_profile();
    let pixels = [30_u8, 255, 0];
    let codestream = encode(
        &pixels,
        1,
        1,
        3,
        8,
        false,
        &EncodeOptions {
            reversible: true,
            ..EncodeOptions::default()
        },
    )
    .expect("encode mixed-colour fixture");
    let encoded = profile.encode().expect("encode ICC profile");
    let tagged = wrap_j2k_codestream(
        &codestream,
        J2kFileWrapOptions::jp2().with_color(J2kFileColorSpec::IccProfile(&encoded)),
    )
    .expect("wrap mixed-colour fixture");

    // The real transform shifts this mixed colour well beyond one code value,
    // so writing the samples unchanged would be wrong.
    let normalized = J2kDecoder::new(&tagged)
        .expect("decoder")
        .decode_srgb8()
        .expect("decode_srgb8");
    assert!(normalized.data()[0].abs_diff(pixels[0]) > 10);

    let mut out = [0_u8; 3];
    let error = J2kDecoder::new(&tagged)
        .expect("decoder")
        .decode_region_into(
            &mut J2kScratchPool::new(),
            &mut out,
            3,
            PixelFormat::Rgb8,
            Rect {
                x: 0,
                y: 0,
                w: 1,
                h: 1,
            },
        )
        .expect_err("crosstalk profile must not bypass the ICC transform");
    assert_eq!(
        error,
        j2k::J2kError::from(Unsupported {
            what: "backend color space cannot be mapped to requested 8-bit pixel format",
        })
    );
}
