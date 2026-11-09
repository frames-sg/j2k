// SPDX-License-Identifier: MIT OR Apache-2.0

//! Parameterized encoding and colour inspection through the exported C ABI.
//! The shim only translates fields, so each encode must equal the codec's own
//! output for the options the fields describe.

use j2k::{
    encode_j2k_lossless, encode_j2k_lossy, wrap_j2k_codestream, J2kBlockCodingMode,
    J2kFileColorSpec, J2kFileWrapOptions, J2kLosslessEncodeOptions, J2kLosslessSamples,
    J2kLossyEncodeOptions, J2kLossySamples, J2kMarkerSegment, J2kProgressionOrder, J2kQualityLayer,
    J2kRateTarget, ReversibleTransform,
};
use j2k_c::*;
use std::ptr;

const WIDTH: u32 = 40;
const HEIGHT: u32 = 28;

fn rgb() -> Vec<u8> {
    (0..WIDTH * HEIGHT * 3)
        .map(|i| u8::try_from((i * 37 + i / 97 * 11) % 251).unwrap())
        .collect()
}

fn rgb_options(flags: u32) -> J2kCEncodeOptions {
    J2kCEncodeOptions {
        width: WIDTH,
        height: HEIGHT,
        components: 3,
        bit_depth: 8,
        flags,
        ..J2kCEncodeOptions::default()
    }
}

fn message(error: &J2kCError) -> String {
    // SAFETY: The library NUL-terminates every diagnostic it writes.
    unsafe { std::ffi::CStr::from_ptr(error.message.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// What `j2k_encode` returned: its status, bytes, report, and diagnostic.
struct Outcome {
    status: u32,
    bytes: Vec<u8>,
    report: J2kCEncodeReport,
    message: String,
}

fn encode(
    samples: &[u8],
    options: &J2kCEncodeOptions,
    params: Option<&J2kCEncodeParams>,
) -> Outcome {
    let mut data = ptr::without_provenance_mut(1);
    let mut length = 7;
    let mut report = J2kCEncodeReport {
        quality_layers: 99,
        ..J2kCEncodeReport::default()
    };
    let mut error = J2kCError::default();
    // SAFETY: All buffers and output slots are valid and disjoint for this call.
    let status = unsafe {
        j2k_encode(
            samples.as_ptr(),
            samples.len(),
            options,
            params.map_or(ptr::null(), ptr::from_ref),
            &raw mut data,
            &raw mut length,
            &raw mut report,
            &raw mut error,
        )
    };
    if status != J2K_OK {
        assert!(
            data.is_null() && length == 0,
            "failure leaves the slots empty"
        );
        assert_eq!(
            report.quality_layers, 99,
            "failure leaves the report unwritten"
        );
        return Outcome {
            status,
            bytes: Vec::new(),
            report,
            message: message(&error),
        };
    }
    // SAFETY: A successful encode returns `length` readable bytes at `data`,
    // released here exactly once.
    let bytes = unsafe { std::slice::from_raw_parts(data, length) }.to_vec();
    // SAFETY: `data` and `length` came from `j2k_encode`.
    unsafe { j2k_buffer_free(data, length) };
    Outcome {
        status,
        bytes,
        report,
        message: String::new(),
    }
}

fn lossless(samples: &[u8], options: J2kLosslessEncodeOptions) -> Vec<u8> {
    let input = J2kLosslessSamples::new(samples, WIDTH, HEIGHT, 3, 8, false).unwrap();
    encode_j2k_lossless(input, &options.with_cpu_only_backend())
        .unwrap()
        .codestream
}

fn lossy(samples: &[u8], options: J2kLossyEncodeOptions) -> j2k::EncodedLossyJ2k {
    let input = J2kLossySamples::new(samples, WIDTH, HEIGHT, 3, 8, false).unwrap();
    encode_j2k_lossy(input, &options.with_cpu_only_backend()).unwrap()
}

/// Whether `bytes` contains the two-byte marker `code`.
fn has_marker(bytes: &[u8], code: u16) -> bool {
    bytes.windows(2).any(|pair| pair == code.to_be_bytes())
}

#[test]
fn lossless_parameters_encode_as_the_codec_does_and_round_trip() {
    let samples = rgb();
    let params = J2kCEncodeParams {
        levels: 2,
        progression: J2K_PROGRESSION_RPCL,
        tile_width: 24,
        tile_height: 16,
        markers: J2K_MARKER_SOP | J2K_MARKER_EPH | J2K_MARKER_TLM | J2K_MARKER_PLT,
        tile_part_packets: 3,
        quality_layers: 2,
        flags: J2K_PARAMS_NO_COLOR_TRANSFORM,
        ..J2kCEncodeParams::default()
    };
    for ht in [false, true] {
        let flags = if ht { J2K_ENCODE_HTJ2K } else { 0 };
        let outcome = encode(&samples, &rgb_options(flags), Some(&params));
        assert_eq!(outcome.status, J2K_OK, "ht={ht}: {}", outcome.message);
        let expected = lossless(
            &samples,
            J2kLosslessEncodeOptions::default()
                .with_block_coding_mode(if ht {
                    J2kBlockCodingMode::HighThroughput
                } else {
                    J2kBlockCodingMode::Classic
                })
                .with_max_decomposition_levels(Some(2))
                .with_progression(J2kProgressionOrder::Rpcl)
                .with_tile_size(Some((24, 16)))
                .with_marker_segments(&[
                    J2kMarkerSegment::Sop,
                    J2kMarkerSegment::Eph,
                    J2kMarkerSegment::Tlm,
                    J2kMarkerSegment::Plt,
                ])
                .with_tile_part_packet_limit(Some(3))
                .with_quality_layers(2)
                .with_reversible_transform(ReversibleTransform::None53),
        );
        assert_eq!(outcome.bytes, expected, "ht={ht}");
        assert!(has_marker(&outcome.bytes, 0xFF91) && has_marker(&outcome.bytes, 0xFF55));
        let bytes = u32::try_from(expected.len()).unwrap();
        assert_eq!(
            outcome.report,
            J2kCEncodeReport {
                codestream_bytes: u64::from(bytes),
                bits_per_pixel: f64::from(bytes) * 8.0 / f64::from(WIDTH * HEIGHT),
                quality_layers: 2,
                ..J2kCEncodeReport::default()
            }
        );
        let info = Handle::open(&outcome.bytes).info();
        assert_eq!((info.tile_width, info.tile_height), (24, 16));
        let mut decoder = j2k::J2kDecoder::new(&outcome.bytes).unwrap();
        let mut pixels = vec![0; samples.len()];
        decoder
            .decode_into(&mut pixels, WIDTH as usize * 3, j2k::PixelFormat::Rgb8)
            .unwrap();
        assert_eq!(pixels, samples, "lossless parameters stay lossless");
    }
}

#[test]
fn null_parameters_encode_exactly_as_lossless_encode() {
    let samples = rgb();
    let options = rgb_options(J2K_ENCODE_HTJ2K | J2K_ENCODE_FILE);
    let defaults = encode(&samples, &options, None);
    assert_eq!(defaults.status, J2K_OK, "{}", defaults.message);
    let initialized = encode(&samples, &options, Some(&J2kCEncodeParams::default()));
    assert_eq!(initialized.bytes, defaults.bytes);
    let mut data = ptr::null_mut();
    let mut length = 0;
    // SAFETY: All buffers and output slots are valid and disjoint for this call.
    let status = unsafe {
        j2k_encode_lossless(
            samples.as_ptr(),
            samples.len(),
            &raw const options,
            &raw mut data,
            &raw mut length,
            ptr::null_mut(),
        )
    };
    assert_eq!(status, J2K_OK);
    // SAFETY: A successful encode returns `length` readable bytes at `data`.
    let legacy = unsafe { std::slice::from_raw_parts(data, length) }.to_vec();
    // SAFETY: `data` and `length` came from `j2k_encode_lossless`.
    unsafe { j2k_buffer_free(data, length) };
    assert_eq!(legacy, defaults.bytes);
    assert_eq!(
        (defaults.report.quality_layers, defaults.report.psnr_db),
        (1, 0.0)
    );
}

#[test]
fn lossy_targets_encode_as_the_codec_does_and_report_its_measurements() {
    let samples = rgb();
    let layer_targets = [0.75, 1.5, 3.0];
    let cases: [(&str, u32, J2kCEncodeParams, &[f64], J2kLossyEncodeOptions); 5] = [
        (
            "fixed quantization",
            0,
            J2kCEncodeParams::default(),
            &[],
            J2kLossyEncodeOptions::default(),
        ),
        (
            "bits per pixel",
            0,
            J2kCEncodeParams {
                rate: J2K_RATE_BITS_PER_PIXEL,
                progression: J2K_PROGRESSION_CPRL,
                levels: 3,
                ..J2kCEncodeParams::default()
            },
            &[2.0],
            J2kLossyEncodeOptions::default()
                .with_rate_target(Some(J2kRateTarget::BitsPerPixel(2.0)))
                .with_progression(J2kProgressionOrder::Cprl)
                .with_max_decomposition_levels(Some(3)),
        ),
        (
            "bytes",
            J2K_ENCODE_HTJ2K,
            J2kCEncodeParams {
                rate: J2K_RATE_BYTES,
                markers: J2K_MARKER_TLM | J2K_MARKER_PLT,
                ..J2kCEncodeParams::default()
            },
            &[600.0],
            J2kLossyEncodeOptions::default()
                .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
                .with_rate_target(Some(J2kRateTarget::Bytes(600)))
                .with_marker_segments(vec![J2kMarkerSegment::Tlm, J2kMarkerSegment::Plt]),
        ),
        (
            "cumulative layers",
            0,
            J2kCEncodeParams {
                rate: J2K_RATE_BITS_PER_PIXEL,
                tile_width: 20,
                tile_height: 14,
                ..J2kCEncodeParams::default()
            },
            &layer_targets,
            J2kLossyEncodeOptions::default()
                .with_quality_layers(
                    layer_targets
                        .iter()
                        .map(|&bpp| J2kQualityLayer::new(J2kRateTarget::BitsPerPixel(bpp)))
                        .collect(),
                )
                .with_tile_size(Some((20, 14))),
        ),
        (
            "qfactor",
            J2K_ENCODE_HTJ2K,
            J2kCEncodeParams {
                qfactor: 80,
                ..J2kCEncodeParams::default()
            },
            &[],
            J2kLossyEncodeOptions::default()
                .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
                .with_qfactor(Some(80)),
        ),
    ];
    for (case, flags, mut params, targets, expected) in cases {
        params.mode = J2K_MODE_LOSSY;
        params.target_count = u32::try_from(targets.len()).unwrap();
        params.targets = if targets.is_empty() {
            ptr::null()
        } else {
            targets.as_ptr()
        };
        let outcome = encode(&samples, &rgb_options(flags), Some(&params));
        assert_eq!(outcome.status, J2K_OK, "{case}: {}", outcome.message);
        let expected = lossy(&samples, expected);
        assert_eq!(outcome.bytes, expected.codestream, "{case}");
        assert_eq!(
            outcome.report,
            J2kCEncodeReport {
                codestream_bytes: expected.report.actual_bytes,
                bits_per_pixel: expected.report.actual_bits_per_pixel,
                psnr_db: expected.report.psnr_db.unwrap_or(0.0),
                quality_layers: u32::from(expected.report.quality_layers),
                ..J2kCEncodeReport::default()
            },
            "{case}"
        );
    }
}

#[test]
fn psnr_targets_reach_the_requested_quality() {
    let samples = rgb();
    let target = [32.0];
    let params = J2kCEncodeParams {
        mode: J2K_MODE_LOSSY,
        rate: J2K_RATE_PSNR_DB,
        target_count: 1,
        targets: target.as_ptr(),
        ..J2kCEncodeParams::default()
    };
    let outcome = encode(&samples, &rgb_options(0), Some(&params));
    assert_eq!(outcome.status, J2K_OK, "{}", outcome.message);
    let expected = lossy(
        &samples,
        J2kLossyEncodeOptions::default().with_rate_target(Some(J2kRateTarget::PsnrDb(32.0))),
    );
    assert_eq!(outcome.bytes, expected.codestream);
    assert!(
        outcome.report.psnr_db >= 32.0 - 0.25,
        "{:?}",
        outcome.report
    );
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one table of every rejected parameter combination"
)]
fn invalid_parameters_are_rejected_and_leave_every_result_unwritten() {
    let samples = rgb();
    let one = [1.0];
    let fraction = [1.5];
    let lossy = J2kCEncodeParams {
        mode: J2K_MODE_LOSSY,
        ..J2kCEncodeParams::default()
    };
    let rated = J2kCEncodeParams {
        rate: J2K_RATE_BITS_PER_PIXEL,
        target_count: 1,
        targets: one.as_ptr(),
        ..lossy
    };
    let profile = b"test-icc-profile";
    let invalid = [
        ("mode", J2kCEncodeParams { mode: 2, ..lossy }, 0),
        ("parameter flag", J2kCEncodeParams { flags: 2, ..lossy }, 0),
        (
            "marker bit",
            J2kCEncodeParams {
                markers: 128,
                ..lossy
            },
            0,
        ),
        (
            "progression",
            J2kCEncodeParams {
                progression: 5,
                ..lossy
            },
            0,
        ),
        (
            "levels",
            J2kCEncodeParams {
                levels: 256,
                ..lossy
            },
            0,
        ),
        (
            "tile-part packets",
            J2kCEncodeParams {
                tile_part_packets: 65_536,
                ..lossy
            },
            0,
        ),
        (
            "lossless qfactor",
            J2kCEncodeParams {
                qfactor: 50,
                ..J2kCEncodeParams::default()
            },
            J2K_ENCODE_HTJ2K,
        ),
        (
            "lossless targets",
            J2kCEncodeParams {
                mode: J2K_MODE_LOSSLESS,
                ..rated
            },
            0,
        ),
        (
            "lossless layer count",
            J2kCEncodeParams {
                quality_layers: 256,
                ..J2kCEncodeParams::default()
            },
            0,
        ),
        (
            "lossy layer count",
            J2kCEncodeParams {
                quality_layers: 2,
                ..lossy
            },
            0,
        ),
        (
            "lossy without colour transform",
            J2kCEncodeParams {
                flags: J2K_PARAMS_NO_COLOR_TRANSFORM,
                ..lossy
            },
            0,
        ),
        (
            "targets without a rate",
            J2kCEncodeParams {
                rate: J2K_RATE_NONE,
                ..rated
            },
            0,
        ),
        (
            "rate without targets",
            J2kCEncodeParams {
                target_count: 0,
                ..rated
            },
            0,
        ),
        ("rate unit", J2kCEncodeParams { rate: 4, ..rated }, 0),
        (
            "fractional bytes",
            J2kCEncodeParams {
                rate: J2K_RATE_BYTES,
                targets: fraction.as_ptr(),
                ..rated
            },
            0,
        ),
        (
            "missing targets",
            J2kCEncodeParams {
                targets: ptr::null(),
                ..rated
            },
            0,
        ),
        (
            "ICC without a file",
            J2kCEncodeParams {
                icc_profile: profile.as_ptr(),
                icc_profile_len: profile.len(),
                ..J2kCEncodeParams::default()
            },
            0,
        ),
        (
            "ICC length without a profile",
            J2kCEncodeParams {
                icc_profile_len: 4,
                ..J2kCEncodeParams::default()
            },
            J2K_ENCODE_FILE,
        ),
        (
            "empty ICC profile",
            J2kCEncodeParams {
                icc_profile: profile.as_ptr(),
                ..J2kCEncodeParams::default()
            },
            J2K_ENCODE_FILE,
        ),
    ];
    for (case, params, flags) in invalid {
        let outcome = encode(&samples, &rgb_options(flags), Some(&params));
        assert_eq!(
            outcome.status, J2K_INVALID_ARGUMENT,
            "{case}: {}",
            outcome.message
        );
    }
    // The codec, not the shim, rules on coding combinations.
    for (case, params, flags) in [
        (
            "classic qfactor",
            J2kCEncodeParams {
                qfactor: 50,
                ..lossy
            },
            0,
        ),
        (
            "qfactor with a rate",
            J2kCEncodeParams {
                qfactor: 50,
                ..rated
            },
            J2K_ENCODE_HTJ2K,
        ),
    ] {
        let outcome = encode(&samples, &rgb_options(flags), Some(&params));
        assert_eq!(
            outcome.status, J2K_UNSUPPORTED,
            "{case}: {}",
            outcome.message
        );
    }
    let mut report = J2kCEncodeReport::default();
    report.header.struct_size -= 1;
    let mut data = ptr::null_mut();
    let mut length = 0;
    let options = rgb_options(0);
    // SAFETY: All buffers and output slots are valid; the report prefix is too small.
    let status = unsafe {
        j2k_encode(
            samples.as_ptr(),
            samples.len(),
            &raw const options,
            ptr::null(),
            &raw mut data,
            &raw mut length,
            &raw mut report,
            ptr::null_mut(),
        )
    };
    assert_eq!(status, J2K_ABI_MISMATCH);
    assert!(data.is_null() && length == 0);
}

struct Handle(*mut J2kCDecoder);

impl Handle {
    fn open(bytes: &[u8]) -> Self {
        let mut handle = ptr::null_mut();
        let mut error = J2kCError::default();
        // SAFETY: All buffers and output slots are valid and disjoint for this call.
        let status = unsafe {
            j2k_decoder_open(
                bytes.as_ptr(),
                bytes.len(),
                ptr::null(),
                &raw mut handle,
                &raw mut error,
            )
        };
        assert_eq!(status, J2K_OK, "{}", message(&error));
        Self(handle)
    }

    fn info(&self) -> J2kCImageInfo {
        let mut info = J2kCImageInfo::default();
        // SAFETY: Live exclusive handle and an initialized result structure.
        let status = unsafe { j2k_decoder_info(self.0, &raw mut info, ptr::null_mut()) };
        assert_eq!(status, J2K_OK);
        info
    }

    fn color(&self) -> J2kCColorInfo {
        let mut color = J2kCColorInfo::default();
        // SAFETY: Live exclusive handle and an initialized result structure.
        let status = unsafe { j2k_decoder_color_info(self.0, &raw mut color, ptr::null_mut()) };
        assert_eq!(status, J2K_OK);
        color
    }

    fn icc_profile(&self, output: &mut [u8]) -> u32 {
        // SAFETY: Live exclusive handle and a writable output of the given length.
        unsafe {
            j2k_decoder_icc_profile(self.0, output.as_mut_ptr(), output.len(), ptr::null_mut())
        }
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        assert_eq!(
            // SAFETY: This is the only owner and its slot is valid for writing.
            unsafe { j2k_decoder_close(&raw mut self.0, ptr::null_mut()) },
            J2K_OK
        );
    }
}

#[test]
fn colour_inspection_reports_the_specification_decoding_applies() {
    let samples = rgb();
    let profile = b"test-icc-profile";
    let params = J2kCEncodeParams {
        icc_profile: profile.as_ptr(),
        icc_profile_len: profile.len(),
        ..J2kCEncodeParams::default()
    };
    let tagged = encode(&samples, &rgb_options(J2K_ENCODE_FILE), Some(&params));
    assert_eq!(tagged.status, J2K_OK, "{}", tagged.message);
    let handle = Handle::open(&tagged.bytes);
    assert_eq!(
        handle.color(),
        J2kCColorInfo {
            specification: J2K_COLOR_ICC,
            specifications: 1,
            icc_profile_bytes: profile.len() as u64,
            ..J2kCColorInfo::default()
        }
    );
    let mut short = [0xA5_u8; 15];
    assert_eq!(handle.icc_profile(&mut short), J2K_BUFFER_TOO_SMALL);
    assert_eq!(short, [0xA5; 15], "a short buffer stays untouched");
    let mut copied = [0_u8; 20];
    assert_eq!(handle.icc_profile(&mut copied), J2K_OK);
    assert_eq!(&copied[..profile.len()], profile);
    assert_eq!(copied[profile.len()..], [0; 4]);

    // A later ICC box is preserved but not applied.
    let codestream = encode(&samples, &rgb_options(0), None).bytes;
    let colors = [
        J2kFileColorSpec::Infer,
        J2kFileColorSpec::IccProfile(profile),
    ];
    let file = wrap_j2k_codestream(
        &codestream,
        J2kFileWrapOptions::jp2().with_color_specs(&colors),
    )
    .unwrap();
    let handle = Handle::open(&file);
    assert_eq!(
        handle.color(),
        J2kCColorInfo {
            specification: J2K_COLOR_ENUMERATED,
            value: 16,
            specifications: 2,
            ..J2kCColorInfo::default()
        }
    );
    assert_eq!(handle.icc_profile(&mut copied), J2K_INVALID_ARGUMENT);

    // A raw codestream has no colour specification.
    assert_eq!(Handle::open(&codestream).color(), J2kCColorInfo::default());
}
