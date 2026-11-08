// SPDX-License-Identifier: MIT OR Apache-2.0

//! ICC colour management stays within bounded memory.
//!
//! A multi-localized Unicode (`mluc`) tag's records may all name one string,
//! and a profile's six text signatures may all name one tag, so parsing such
//! text allocates in proportion to the square of the tag's size. Text never
//! affects the colour transform, so it is not parsed. Under an explicit
//! working limit, the colour-management state of a restricted profile is
//! charged before it exists, so the operation's measured heap stays within
//! the limit. Heap peaks come from the process-wide counting allocator of
//! `j2k_alloc_probe`; the tests here take one lock and decode serially.

use std::num::NonZeroUsize;
use std::sync::Mutex;

use j2k::{
    wrap_j2k_codestream, CpuDecodeParallelism, J2kDecodeControl, J2kDecoder, J2kError,
    J2kFileColorSpec, J2kFileWrapOptions,
};
use j2k_core::PixelFormat;
use j2k_native::{encode, EncodeOptions};

static SERIAL: Mutex<()> = Mutex::new(());

const TEXT_SIGNATURES: [&[u8; 4]; 6] = [b"cprt", b"desc", b"vued", b"dmnd", b"dmdd", b"targ"];

fn be32(bytes: &[u8]) -> usize {
    u32::from_be_bytes(bytes[..4].try_into().expect("four bytes")) as usize
}

fn put32(bytes: &mut [u8], value: usize) {
    bytes[..4].copy_from_slice(&u32::try_from(value).expect("ICC field").to_be_bytes());
}

/// The standard sRGB profile with every text signature naming one `mluc`
/// tag whose `records` all name the same `units` UTF-16 code units. With
/// `duplicate`, the profile's own text entries are kept as well.
fn srgb_profile_with_amplifying_text(records: usize, units: usize, duplicate: bool) -> Vec<u8> {
    let profile = moxcms::ColorProfile::new_srgb()
        .encode()
        .expect("sRGB profile");
    let records_end = 16 + 12 * records;
    let mut tag = Vec::with_capacity(records_end + 2 * units);
    tag.extend_from_slice(b"mluc\0\0\0\0");
    tag.extend_from_slice(&u32::try_from(records).expect("records").to_be_bytes());
    tag.extend_from_slice(&12_u32.to_be_bytes());
    for _ in 0..records {
        tag.extend_from_slice(b"enUS");
        tag.extend_from_slice(&u32::try_from(2 * units).expect("length").to_be_bytes());
        tag.extend_from_slice(&u32::try_from(records_end).expect("offset").to_be_bytes());
    }
    for _ in 0..units {
        // One CJK ideograph: three UTF-8 bytes per UTF-16 code unit.
        tag.extend_from_slice(&0x4e00_u16.to_be_bytes());
    }

    let count = be32(&profile[128..]);
    let table_end = 132 + 12 * count;
    let kept: Vec<&[u8; 12]> = profile[132..table_end]
        .as_chunks::<12>()
        .0
        .iter()
        .filter(|entry| duplicate || !TEXT_SIGNATURES.iter().any(|tag| entry[..4] == tag[..]))
        .collect();
    let new_count = kept.len() + TEXT_SIGNATURES.len();
    // The rebuilt table moves the original tag data by its change in size.
    let shift = 12 * new_count - 12 * count;
    let mut data = profile[table_end..].to_vec();
    while !(table_end + shift + data.len()).is_multiple_of(4) {
        data.push(0);
    }
    let tag_offset = table_end + shift + data.len();
    let mut out = profile[..132].to_vec();
    for entry in kept {
        out.extend_from_slice(&entry[..4]);
        out.extend_from_slice(
            &u32::try_from(be32(&entry[4..]) + shift)
                .expect("offset")
                .to_be_bytes(),
        );
        out.extend_from_slice(&entry[8..12]);
    }
    for signature in TEXT_SIGNATURES {
        out.extend_from_slice(signature);
        out.extend_from_slice(&u32::try_from(tag_offset).expect("offset").to_be_bytes());
        out.extend_from_slice(&u32::try_from(tag.len()).expect("size").to_be_bytes());
    }
    out.extend_from_slice(&data);
    out.extend_from_slice(&tag);
    let total = out.len();
    put32(&mut out[0..], total);
    put32(&mut out[128..], new_count);
    out
}

fn peak_during<R>(operation: impl FnOnce() -> R) -> (R, usize) {
    let before = j2k_alloc_probe::reset_peak();
    let result = operation();
    (result, j2k_alloc_probe::peak_bytes() - before)
}

fn tagged(pixels: &[u8], width: u32, height: u32, components: u16, profile: &[u8]) -> Vec<u8> {
    let codestream = encode(
        pixels,
        width,
        height,
        components,
        8,
        false,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels: 0,
            ..EncodeOptions::default()
        },
    )
    .expect("fixture codestream");
    wrap_j2k_codestream(
        &codestream,
        J2kFileWrapOptions::jp2().with_color(J2kFileColorSpec::IccProfile(profile)),
    )
    .expect("ICC JP2")
}

#[test]
fn icc_text_tags_do_not_amplify_parse_memory() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // About 29 KB of profile; parsing its text would hold well over 100 MiB.
    let profile = srgb_profile_with_amplifying_text(1024, 8192, false);
    assert!(profile.len() < 32 << 10, "{}", profile.len());
    let pixels = [200_u8, 100, 50, 10, 20, 30];
    let codestream = encode(
        &pixels,
        2,
        1,
        3,
        8,
        false,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels: 0,
            ..EncodeOptions::default()
        },
    )
    .expect("RGB fixture");
    let bytes = wrap_j2k_codestream(
        &codestream,
        J2kFileWrapOptions::jp2().with_color(J2kFileColorSpec::IccProfile(&profile)),
    )
    .expect("ICC JP2");
    let bound = 8 << 20;

    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    decoder.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);
    let mut out = [0_u8; 6];
    let (result, peak) = peak_during(|| decoder.decode_into(&mut out, 6, PixelFormat::Rgb8));
    result.expect("an sRGB-equivalent profile decodes on the ordinary route");
    assert_eq!(out, pixels);
    assert!(peak < bound, "ordinary decode held {peak} bytes");

    let (normalized, peak) = peak_during(|| decoder.decode_srgb8());
    assert_eq!(normalized.expect("normalized").data(), pixels);
    assert!(peak < bound, "normalized decode held {peak} bytes");

    // ICC forbids repeated tag signatures; such a profile is invalid, not
    // parsed twice.
    let duplicated = wrap_j2k_codestream(
        &codestream,
        J2kFileWrapOptions::jp2().with_color(J2kFileColorSpec::IccProfile(
            &srgb_profile_with_amplifying_text(1, 1, true),
        )),
    )
    .expect("duplicate-tag JP2");
    let (normalized, peak) = peak_during(|| {
        J2kDecoder::new(&duplicated)
            .expect("decoder")
            .decode_srgb8()
    });
    assert!(
        matches!(normalized, Err(j2k::J2kError::InvalidIccProfile)),
        "{normalized:?}"
    );
    assert!(peak < bound, "rejected decode held {peak} bytes");
}

/// Every 65,536-entry tone curve distinct: the largest restricted RGB state.
fn rgb_profile_with_distinct_full_curves() -> Vec<u8> {
    let mut profile = moxcms::ColorProfile::new_srgb();
    let curve = |exponent: f64| {
        (0..=u16::MAX)
            .map(|value| {
                let linear = (f64::from(value) / 65_535.0).powf(exponent) * 65_535.0;
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "a power of a unit fraction scaled by 65,535 rounds into the u16 range"
                )]
                let sample = linear.round() as u16;
                sample
            })
            .collect::<Vec<u16>>()
    };
    profile.red_trc = Some(moxcms::ToneReprCurve::Lut(curve(2.2)));
    profile.green_trc = Some(moxcms::ToneReprCurve::Lut(curve(2.0)));
    profile.blue_trc = Some(moxcms::ToneReprCurve::Lut(curve(1.8)));
    profile.encode().expect("full-curve RGB profile")
}

/// Tagged RGB and gray fixtures with the restricted profile shapes whose
/// colour-management state a working limit charges, and the RGB samples.
fn working_limit_cases(width: u32, height: u32) -> (Vec<(&'static str, Vec<u8>)>, Vec<u8>) {
    let rgb: Vec<u8> = (0..width * height * 3)
        .map(|index| u8::try_from(index * 37 % 251).expect("masked"))
        .collect();
    let gray: Vec<u8> = (0..width * height)
        .map(|index| u8::try_from(index * 53 % 251).expect("masked"))
        .collect();
    let cases = vec![
        (
            "sRGB",
            tagged(
                &rgb,
                width,
                height,
                3,
                &moxcms::ColorProfile::new_srgb().encode().expect("sRGB"),
            ),
        ),
        (
            "ProPhoto",
            tagged(
                &rgb,
                width,
                height,
                3,
                &moxcms::ColorProfile::new_pro_photo_rgb()
                    .encode()
                    .expect("ProPhoto"),
            ),
        ),
        (
            "distinct full curves",
            tagged(
                &rgb,
                width,
                height,
                3,
                &rgb_profile_with_distinct_full_curves(),
            ),
        ),
        (
            "gray",
            tagged(
                &gray,
                width,
                height,
                1,
                &moxcms::ColorProfile::new_gray_with_gamma(2.2)
                    .encode()
                    .expect("gray"),
            ),
        ),
    ];
    (cases, rgb)
}

#[test]
fn icc_conversion_under_a_working_limit_stays_within_it() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (width, height) = (24_u32, 16_u32);
    let (cases, rgb) = working_limit_cases(width, height);
    for (name, bytes) in &cases {
        let unlimited = J2kDecoder::new(bytes)
            .expect("decoder")
            .decode_srgb8()
            .unwrap_or_else(|error| panic!("{name}: {error:?}"));
        let run = |limit: usize| {
            let mut control = J2kDecodeControl::default();
            control.max_working_memory_bytes = NonZeroUsize::new(limit);
            let mut decoder = J2kDecoder::new(bytes).expect("decoder");
            decoder.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);
            peak_during(|| decoder.decode_srgb8_with_control(&control))
        };
        let (mut low, mut high) = (1, 64 << 20);
        assert!(run(high).0.is_ok(), "{name}");
        while low < high {
            let middle = low + (high - low) / 2;
            if run(middle).0.is_ok() {
                high = middle;
            } else {
                low = middle + 1;
            }
        }
        let (decoded, peak) = run(high);
        assert_eq!(decoded.expect("at the smallest limit"), unlimited, "{name}");
        assert!(
            peak <= high,
            "{name}: measured {peak} bytes above the {high}-byte working limit"
        );
        let refused = run(high - 1).0;
        assert!(
            matches!(refused, Err(J2kError::ResourceLimit { .. })),
            "{name}: {refused:?}"
        );
    }

    // A profile connecting through Lab takes an unbounded lookup path and is
    // refused under a limit, while it still converts without one.
    let mut lab = moxcms::ColorProfile::new_srgb().encode().expect("sRGB");
    lab[20..24].copy_from_slice(b"Lab ");
    let lab = tagged(&rgb, width, height, 3, &lab);
    let mut control = J2kDecodeControl::default();
    control.max_working_memory_bytes = NonZeroUsize::new(64 << 20);
    let limited = J2kDecoder::new(&lab)
        .expect("decoder")
        .decode_srgb8_with_control(&control);
    assert!(
        matches!(limited, Err(J2kError::Unsupported(_))),
        "{limited:?}"
    );
}
