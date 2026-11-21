// SPDX-License-Identifier: MIT OR Apache-2.0

//! DICOM RLE Lossless frames through the C ABI.

use j2k_c::*;
use j2k_tilecodec::{DicomRleCodec, DicomRleLayout, DicomRlePlanarConfiguration};
use std::ptr;

const CASES_TSV: &str = include_str!("../../j2k-tilecodec/tests/fixtures/dicom_rle/cases.tsv");
const DATA: &[u8] = include_bytes!("../../j2k-tilecodec/tests/fixtures/dicom_rle/data.bin");

fn layout(rows: u32, columns: u32, samples_per_pixel: u32, bits_allocated: u32) -> J2kCRleLayout {
    J2kCRleLayout {
        rows,
        columns,
        samples_per_pixel,
        bits_allocated,
        ..J2kCRleLayout::default()
    }
}

/// Decode through the C ABI into `output`, returning the status and report.
fn decode(input: &[u8], layout: &J2kCRleLayout, output: &mut [u8]) -> (u32, J2kCRleReport) {
    let mut report = J2kCRleReport {
        overlong_segments: u32::MAX,
        nonzero_unused_offsets: u32::MAX,
        ..J2kCRleReport::default()
    };
    let mut error = J2kCError::default();
    // SAFETY: Every pointer is live, aligned and disjoint for this call.
    let status = unsafe {
        j2k_rle_decode(
            input.as_ptr(),
            input.len(),
            layout,
            output.as_mut_ptr(),
            output.len(),
            &raw mut report,
            &raw mut error,
        )
    };
    assert_eq!(status, error.code);
    if status != J2K_OK {
        assert_ne!(error.message[0], 0, "a failed decode explains itself");
    }
    (status, report)
}

/// Encode through the C ABI, returning an owned copy or the failing status.
fn encode(samples: &[u8], layout: &J2kCRleLayout) -> Result<Vec<u8>, u32> {
    let (mut data, mut length) = (ptr::dangling_mut::<u8>(), usize::MAX);
    // SAFETY: Every pointer is live, aligned and disjoint for this call.
    let status = unsafe {
        j2k_rle_encode(
            samples.as_ptr(),
            samples.len(),
            layout,
            &raw mut data,
            &raw mut length,
            ptr::null_mut(),
        )
    };
    if status != J2K_OK {
        assert!(
            data.is_null() && length == 0,
            "a failed encode returns no buffer"
        );
        return Err(status);
    }
    // SAFETY: A successful encode returned `length` readable bytes at `data`.
    let copy = unsafe { std::slice::from_raw_parts(data, length) }.to_vec();
    // SAFETY: The buffer came from j2k_rle_encode and is released once.
    unsafe { j2k_buffer_free(data, length) };
    Ok(copy)
}

fn planar(configuration: u32) -> DicomRlePlanarConfiguration {
    if configuration == 0 {
        DicomRlePlanarConfiguration::Interleaved
    } else {
        DicomRlePlanarConfiguration::Planar
    }
}

fn rust_layout(layout: &J2kCRleLayout) -> DicomRleLayout {
    let field = |value: u32| u16::try_from(value).unwrap();
    DicomRleLayout::new(
        field(layout.rows),
        field(layout.columns),
        field(layout.samples_per_pixel),
        field(layout.bits_allocated),
        planar(layout.planar_configuration),
    )
    .unwrap()
}

/// Native bytes with runs and literals.
fn frame_samples(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| {
            if i % 29 < 11 {
                7
            } else {
                u8::try_from((i * 73 + i / 5) % 256).unwrap()
            }
        })
        .collect()
}

#[test]
fn every_layout_round_trips_and_matches_the_rust_codec() {
    for bits in [8, 16, 32] {
        for samples in [1, 3] {
            for configuration in [0, 1] {
                let context = format!("{bits} bits, {samples} samples, planar {configuration}");
                let layout = J2kCRleLayout {
                    planar_configuration: configuration,
                    ..layout(7, 9, samples, bits)
                };
                let frame_len = rust_layout(&layout).frame_len();
                let source = frame_samples(frame_len);
                let encoded = encode(&source, &layout).unwrap();
                let mut expected = Vec::new();
                DicomRleCodec::encode_frame(&source, rust_layout(&layout), &mut expected).unwrap();
                assert_eq!(
                    encoded, expected,
                    "{context}: the C encode matches the crate's"
                );

                // A longer buffer is accepted; only the frame is written.
                let mut output = vec![0xA5_u8; frame_len + 8];
                let (status, report) = decode(&encoded, &layout, &mut output);
                assert_eq!(status, J2K_OK, "{context}");
                assert_eq!(&output[..frame_len], &source[..], "{context}");
                assert!(
                    output[frame_len..].iter().all(|&byte| byte == 0xA5),
                    "{context}"
                );
                assert_eq!(
                    (report.overlong_segments, report.nonzero_unused_offsets),
                    (0, 0)
                );
                let lenient = J2kCRleLayout {
                    flags: J2K_RLE_LENIENT,
                    ..layout
                };
                let (status, report) = decode(&encoded, &lenient, &mut output);
                assert_eq!(status, J2K_OK, "{context}");
                assert_eq!(
                    (report.overlong_segments, report.nonzero_unused_offsets),
                    (0, 0)
                );
            }
        }
    }
}

#[test]
fn fixture_frames_decode_and_encode_as_the_rust_codec_does() {
    let slice = |field: &str| {
        let (offset, len) = field.split_once(':').unwrap();
        let offset: usize = offset.parse().unwrap();
        &DATA[offset..offset + len.parse::<usize>().unwrap()]
    };
    let mut cases = 0;
    for line in CASES_TSV
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
    {
        let fields: Vec<&str> = line.split('\t').collect();
        let name = fields[0];
        let number = |index: usize| fields[index].parse::<u32>().unwrap();
        let layout = layout(number(2), number(3), number(4), number(5));
        let (frame, reference, pydicom) = (slice(fields[6]), slice(fields[7]), slice(fields[8]));
        let codec = rust_layout(&layout);

        let mut output = vec![0_u8; reference.len()];
        let (strict, _) = decode(frame, &layout, &mut output);
        let mut rust = vec![0_u8; reference.len()];
        match DicomRleCodec::decode_frame(frame, codec, &mut rust) {
            Ok(()) => {
                assert_eq!(strict, J2K_OK, "{name}");
                assert_eq!(output, reference, "{name}");
            }
            Err(_) => assert_eq!(strict, J2K_INVALID_INPUT, "{name}"),
        }
        let lenient = J2kCRleLayout {
            flags: J2K_RLE_LENIENT,
            ..layout
        };
        let (status, report) = decode(frame, &lenient, &mut output);
        let rust_report = DicomRleCodec::decode_frame_lenient(frame, codec, &mut rust).unwrap();
        assert_eq!(status, J2K_OK, "{name}");
        assert_eq!(output, reference, "{name}");
        assert_eq!(
            report.overlong_segments,
            u32::from(rust_report.overlong_segments()),
            "{name}"
        );
        assert_eq!(
            report.nonzero_unused_offsets,
            u32::from(rust_report.has_nonzero_unused_offsets()),
            "{name}"
        );
        assert_eq!(
            encode(reference, &layout).unwrap(),
            pydicom,
            "{name}: matches pydicom"
        );
        cases += 1;
    }
    assert!(cases >= 20, "only {cases} fixtures");
}

#[test]
fn non_conformant_frames_fail_strict_and_are_reported_lenient() {
    let layout = layout(4, 6, 1, 8);
    let source = frame_samples(24);
    let conformant = encode(&source, &layout).unwrap();
    let lenient = J2kCRleLayout {
        flags: J2K_RLE_LENIENT,
        ..layout
    };
    let mut output = vec![0_u8; 24];

    // A literal run after the segment's rows are complete.
    let mut overlong = conformant.clone();
    overlong.extend_from_slice(&[0x00, 0x55]);
    assert_eq!(decode(&overlong, &layout, &mut output).0, J2K_INVALID_INPUT);
    let (status, report) = decode(&overlong, &lenient, &mut output);
    assert_eq!(status, J2K_OK);
    assert_eq!(
        (report.overlong_segments, report.nonzero_unused_offsets),
        (1, 0)
    );
    assert_eq!(output, source);

    // A header offset after the last segment that is not zero.
    let mut offsets = conformant;
    offsets[60..64].copy_from_slice(&1234_u32.to_le_bytes());
    assert_eq!(decode(&offsets, &layout, &mut output).0, J2K_INVALID_INPUT);
    let (status, report) = decode(&offsets, &lenient, &mut output);
    assert_eq!(status, J2K_OK);
    assert_eq!(
        (report.overlong_segments, report.nonzero_unused_offsets),
        (0, 1)
    );
    assert_eq!(output, source);

    // A report is optional.
    let mut error = J2kCError::default();
    // SAFETY: Every pointer is live, aligned and disjoint; the report is null.
    let status = unsafe {
        j2k_rle_decode(
            offsets.as_ptr(),
            offsets.len(),
            &raw const lenient,
            output.as_mut_ptr(),
            output.len(),
            ptr::null_mut(),
            &raw mut error,
        )
    };
    assert_eq!((status, error.code), (J2K_OK, J2K_OK));
}

/// A 4x6 16-bit gray layout, its samples and its frame.
fn small_frame() -> (J2kCRleLayout, Vec<u8>, Vec<u8>) {
    let base = layout(4, 6, 1, 16);
    let source = frame_samples(48);
    let encoded = encode(&source, &base).unwrap();
    (base, source, encoded)
}

#[test]
fn invalid_layouts_are_rejected() {
    let (base, source, encoded) = small_frame();
    let mut output = vec![0_u8; 48];

    for (layout, expected, context) in [
        (layout(0, 6, 1, 16), J2K_INVALID_ARGUMENT, "zero rows"),
        (
            layout(65_536, 6, 1, 16),
            J2K_INVALID_ARGUMENT,
            "rows above 65535",
        ),
        (layout(4, 0, 1, 16), J2K_INVALID_ARGUMENT, "zero columns"),
        (layout(4, 6, 0, 16), J2K_INVALID_ARGUMENT, "zero samples"),
        (
            layout(4, 6, 1, 65_552),
            J2K_INVALID_ARGUMENT,
            "bits above 65535",
        ),
        (
            J2kCRleLayout {
                planar_configuration: 2,
                ..base
            },
            J2K_INVALID_ARGUMENT,
            "planar 2",
        ),
        (
            J2kCRleLayout { flags: 2, ..base },
            J2K_INVALID_ARGUMENT,
            "unknown flag",
        ),
        (layout(4, 6, 1, 12), J2K_UNSUPPORTED, "12 bits allocated"),
        (layout(4, 6, 1, 1), J2K_UNSUPPORTED, "1 bit allocated"),
        (layout(4, 6, 16, 8), J2K_UNSUPPORTED, "16 segments"),
    ] {
        assert_eq!(
            decode(&encoded, &layout, &mut output).0,
            expected,
            "decode: {context}"
        );
        assert_eq!(encode(&source, &layout), Err(expected), "encode: {context}");
    }
    let lenient = J2kCRleLayout {
        flags: J2K_RLE_LENIENT,
        ..base
    };
    assert_eq!(
        encode(&source, &lenient),
        Err(J2K_INVALID_ARGUMENT),
        "lenient encode"
    );
    assert_eq!(
        encode(&source[..47], &base),
        Err(J2K_INVALID_ARGUMENT),
        "short samples"
    );
}

#[test]
fn invalid_headers_pointers_buffers_and_frames_are_rejected() {
    let (base, source, encoded) = small_frame();
    let mut output = vec![0_u8; 48];
    let mut stale = base;
    stale.header.abi_version = 2;
    assert_eq!(decode(&encoded, &stale, &mut output).0, J2K_ABI_MISMATCH);
    assert_eq!(encode(&source, &stale), Err(J2K_ABI_MISMATCH));
    let mut report = J2kCRleReport::default();
    report.header.struct_size -= 1;
    // SAFETY: Every pointer is live, aligned and disjoint for this call.
    let status = unsafe {
        j2k_rle_decode(
            encoded.as_ptr(),
            encoded.len(),
            &raw const base,
            output.as_mut_ptr(),
            output.len(),
            &raw mut report,
            ptr::null_mut(),
        )
    };
    assert_eq!(status, J2K_ABI_MISMATCH);

    // Null arguments and short buffers.
    // SAFETY: A null layout is rejected before any other access.
    let status = unsafe {
        j2k_rle_decode(
            encoded.as_ptr(),
            encoded.len(),
            ptr::null(),
            output.as_mut_ptr(),
            output.len(),
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    assert_eq!(status, J2K_INVALID_ARGUMENT);
    // SAFETY: A null output with a nonzero length is rejected before any write.
    let status = unsafe {
        j2k_rle_decode(
            encoded.as_ptr(),
            encoded.len(),
            &raw const base,
            ptr::null_mut(),
            48,
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    assert_eq!(status, J2K_INVALID_ARGUMENT);
    assert_eq!(
        decode(&encoded, &base, &mut output[..47]).0,
        J2K_BUFFER_TOO_SMALL
    );
    let (mut data, mut length) = (ptr::dangling_mut::<u8>(), 7_usize);
    // SAFETY: A null layout is rejected; the output slots are live and cleared.
    let status = unsafe {
        j2k_rle_encode(
            source.as_ptr(),
            source.len(),
            ptr::null(),
            &raw mut data,
            &raw mut length,
            ptr::null_mut(),
        )
    };
    assert_eq!(status, J2K_INVALID_ARGUMENT);
    assert!(data.is_null() && length == 0);

    // Frames that are not RLE, or are cut off.
    assert_eq!(decode(&[0_u8; 64], &base, &mut output).0, J2K_INVALID_INPUT);
    for cut in [0, 10, 64, encoded.len() - 4] {
        assert_eq!(
            decode(&encoded[..cut.max(1)], &base, &mut output).0,
            J2K_INVALID_INPUT,
            "cut at {cut}"
        );
    }
}
