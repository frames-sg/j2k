// SPDX-License-Identifier: MIT OR Apache-2.0

//! JPEG-LS through the C ABI: inspection, decoding, encoding and rejections.

use j2k_c::*;
use std::ptr;

const CASES_TSV: &str = include_str!("../../j2k-jpegls/fixtures/jpegls_matrix/cases.tsv");
const DATA: &[u8] = include_bytes!("../../j2k-jpegls/fixtures/jpegls_matrix/data.bin");

fn inspect(input: &[u8]) -> Result<J2kCJpegLsInfo, u32> {
    let mut info = J2kCJpegLsInfo::default();
    // SAFETY: The input slice and the output structure are live and disjoint.
    let status =
        unsafe { j2k_jpegls_inspect(input.as_ptr(), input.len(), &raw mut info, ptr::null_mut()) };
    if status == J2K_OK {
        Ok(info)
    } else {
        Err(status)
    }
}

fn decode(input: &[u8], output: &mut [u8], stride: usize) -> J2kCError {
    let mut error = J2kCError::default();
    // SAFETY: The input and output slices are live and disjoint.
    let status = unsafe {
        j2k_jpegls_decode(
            input.as_ptr(),
            input.len(),
            output.as_mut_ptr(),
            output.len(),
            stride,
            &raw mut error,
        )
    };
    assert_eq!(status, error.code, "the status and the error agree");
    error
}

/// Encode through the C ABI, returning an owned copy or the failing status.
fn encode(samples: &[u8], options: &J2kCJpegLsEncodeOptions) -> Result<Vec<u8>, u32> {
    let (mut data, mut length) = (ptr::dangling_mut::<u8>(), usize::MAX);
    let mut error = J2kCError::default();
    // SAFETY: Every pointer is live, aligned and disjoint for this call.
    let status = unsafe {
        j2k_jpegls_encode(
            samples.as_ptr(),
            samples.len(),
            options,
            &raw mut data,
            &raw mut length,
            &raw mut error,
        )
    };
    assert_eq!(status, error.code);
    if status != J2K_OK {
        assert!(
            data.is_null() && length == 0,
            "a failed encode returns no buffer"
        );
        assert_ne!(error.message[0], 0, "a failed encode explains itself");
        return Err(status);
    }
    // SAFETY: A successful encode returned `length` readable bytes at `data`.
    let copy = unsafe { std::slice::from_raw_parts(data, length) }.to_vec();
    // SAFETY: The buffer came from j2k_jpegls_encode and is released once.
    unsafe { j2k_buffer_free(data, length) };
    Ok(copy)
}

fn options(width: u32, height: u32, components: u32, bit_depth: u32) -> J2kCJpegLsEncodeOptions {
    J2kCJpegLsEncodeOptions {
        width,
        height,
        components,
        bit_depth,
        ..J2kCJpegLsEncodeOptions::default()
    }
}

/// Smooth samples with some noise, as interleaved native bytes.
fn samples(width: u32, height: u32, components: u32, bit_depth: u32) -> Vec<u8> {
    let max = (1_u32 << bit_depth) - 1;
    let count = width * height * components;
    let values = (0..count).map(|i| {
        let (pixel, component) = (i / components, i % components);
        let (x, y) = (pixel % width, pixel / width);
        (x * 37 + y * 11 + component * 101 + (i.wrapping_mul(2_654_435_761) >> 28)) % (max + 1)
    });
    if bit_depth > 8 {
        values
            .flat_map(|value| u16::try_from(value).unwrap().to_le_bytes())
            .collect()
    } else {
        values.map(|value| u8::try_from(value).unwrap()).collect()
    }
}

fn values(bytes: &[u8], bit_depth: u32) -> Vec<i32> {
    if bit_depth > 8 {
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&pair| i32::from(u16::from_le_bytes(pair)))
            .collect()
    } else {
        bytes.iter().map(|&value| i32::from(value)).collect()
    }
}

#[test]
fn encodes_inspects_and_decodes_with_every_parameter() {
    let (width, height) = (23, 17);
    for (bit_depth, components, interleave, near, restart) in [
        (8, 1, J2K_JPEGLS_INTERLEAVE_NONE, 0, 0),
        (12, 1, J2K_JPEGLS_INTERLEAVE_NONE, 0, 0),
        (16, 1, J2K_JPEGLS_INTERLEAVE_NONE, 0, 4),
        (2, 1, J2K_JPEGLS_INTERLEAVE_NONE, 0, 0),
        (8, 3, J2K_JPEGLS_INTERLEAVE_NONE, 0, 0),
        (8, 3, J2K_JPEGLS_INTERLEAVE_LINE, 0, 5),
        (8, 3, J2K_JPEGLS_INTERLEAVE_SAMPLE, 2, 0),
        (12, 4, J2K_JPEGLS_INTERLEAVE_SAMPLE, 0, 3),
        (10, 3, J2K_JPEGLS_INTERLEAVE_LINE, 3, 0),
        (16, 2, J2K_JPEGLS_INTERLEAVE_NONE, 7, 0),
    ] {
        let context = format!("{bit_depth} bits, {components} components, ILV {interleave}, NEAR {near}, restart {restart}");
        let source = samples(width, height, components, bit_depth);
        let encoded = encode(
            &source,
            &J2kCJpegLsEncodeOptions {
                near,
                interleave,
                restart_interval: restart,
                ..options(width, height, components, bit_depth)
            },
        )
        .unwrap_or_else(|status| panic!("{context}: encode status {status}"));

        let info = inspect(&encoded).unwrap();
        let sample_bytes = if bit_depth > 8 { 2 } else { 1 };
        let row_bytes = u64::from(width * components * sample_bytes);
        assert_eq!(
            (
                info.width,
                info.height,
                info.components,
                info.bit_depth,
                info.near
            ),
            (width, height, components, bit_depth, near),
            "{context}"
        );
        let coded_interleave = if components == 1 {
            J2K_JPEGLS_INTERLEAVE_NONE
        } else {
            interleave
        };
        assert_eq!(info.interleave, coded_interleave, "{context}");
        assert_eq!(info.restart_interval, restart, "{context}");
        assert_eq!(info.reserved, 0);
        assert_eq!(
            (info.row_bytes, info.buffer_bytes),
            (row_bytes, row_bytes * u64::from(height))
        );

        // A padded stride leaves the gaps alone.
        let row_bytes = usize::try_from(row_bytes).unwrap();
        let stride = row_bytes + 5;
        let span = stride * (usize::try_from(height).unwrap() - 1) + row_bytes;
        let mut output = vec![0xA5_u8; span + 3];
        assert_eq!(
            decode(&encoded, &mut output, stride).code,
            J2K_OK,
            "{context}"
        );
        let mut packed = Vec::new();
        for (row, chunk) in output.chunks(stride).enumerate() {
            let (pixels, gap) = chunk.split_at(row_bytes.min(chunk.len()));
            packed.extend_from_slice(pixels);
            let allowed = if row + 1 == usize::try_from(height).unwrap() {
                3
            } else {
                5
            };
            assert!(
                gap.iter().all(|&byte| byte == 0xA5) && gap.len() <= allowed,
                "{context}"
            );
        }
        assert_eq!(packed.len(), source.len());
        let rust = j2k_jpegls::Decoder::new(&encoded).unwrap();
        let mut expected = vec![0_u8; source.len()];
        rust.decode_into(&mut expected, row_bytes).unwrap();
        assert_eq!(
            packed, expected,
            "{context}: the C decode matches the crate's"
        );
        let tolerance = i32::try_from(near).unwrap();
        for (decoded, original) in values(&packed, bit_depth)
            .iter()
            .zip(values(&source, bit_depth))
        {
            assert!((decoded - original).abs() <= tolerance, "{context}");
        }
    }
}

#[test]
fn fixture_matrix_decodes_as_the_rust_crate_does() {
    let slice = |field: &str| {
        let (offset, len) = field.split_once(':').unwrap();
        let offset: usize = offset.parse().unwrap();
        &DATA[offset..offset + len.parse::<usize>().unwrap()]
    };
    let mut decoded = 0;
    for line in CASES_TSV
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
    {
        let fields: Vec<&str> = line.split('\t').collect();
        let (name, codestream) = (fields[0], slice(fields[12]));
        let Ok(rust) = j2k_jpegls::Decoder::new(codestream) else {
            let mut output = vec![0_u8; 1 << 16];
            let status = decode(codestream, &mut output, 1 << 10).code;
            assert_ne!(status, J2K_OK, "{name}");
            continue;
        };
        let info = inspect(codestream).unwrap();
        assert_eq!(info.width, rust.info().dimensions.0, "{name}");
        assert_eq!(info.bit_depth, u32::from(rust.info().bit_depth), "{name}");
        let row_bytes = rust.info().row_bytes().unwrap();
        let mut expected = vec![0_u8; rust.info().output_len().unwrap()];
        let rust_result = rust.decode_into(&mut expected, row_bytes);
        let mut output = vec![0_u8; expected.len()];
        let status = decode(codestream, &mut output, row_bytes).code;
        if rust_result.is_ok() {
            assert_eq!(status, J2K_OK, "{name}");
            assert_eq!(output, expected, "{name}");
            decoded += 1;
        } else {
            assert_ne!(status, J2K_OK, "{name}");
        }
    }
    assert!(decoded >= 50, "only {decoded} fixtures decoded");
}

/// An 8x6 12-bit gray image: its samples and its codestream.
fn small_image() -> (Vec<u8>, Vec<u8>) {
    let source = samples(8, 6, 1, 12);
    let encoded = encode(&source, &options(8, 6, 1, 12)).unwrap();
    (source, encoded)
}

#[test]
fn inspect_and_decode_reject_invalid_pointers_headers_buffers_and_strides() {
    let (_, encoded) = small_image();
    let mut output = vec![0_u8; 96];
    let mut info = J2kCJpegLsInfo::default();
    // SAFETY: A null input with a nonzero length is rejected before any read.
    let status = unsafe { j2k_jpegls_inspect(ptr::null(), 4, &raw mut info, ptr::null_mut()) };
    assert_eq!(status, J2K_INVALID_ARGUMENT);
    // SAFETY: A null output structure is rejected before any write.
    let status = unsafe {
        j2k_jpegls_inspect(
            encoded.as_ptr(),
            encoded.len(),
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    assert_eq!(status, J2K_INVALID_ARGUMENT);
    for header in [
        J2kCHeader {
            abi_version: 2,
            ..info.header
        },
        J2kCHeader {
            struct_size: info.header.struct_size - 1,
            ..info.header
        },
    ] {
        let mut stale = J2kCJpegLsInfo {
            header,
            ..J2kCJpegLsInfo::default()
        };
        // SAFETY: The input slice and the output structure are live and disjoint.
        let status = unsafe {
            j2k_jpegls_inspect(
                encoded.as_ptr(),
                encoded.len(),
                &raw mut stale,
                ptr::null_mut(),
            )
        };
        assert_eq!(status, J2K_ABI_MISMATCH);
        assert_eq!(stale.width, 0, "a rejected call writes nothing");
    }
    assert_eq!(
        decode(&encoded, &mut output[..95], 16).code,
        J2K_BUFFER_TOO_SMALL
    );
    assert_eq!(decode(&encoded, &mut output, 15).code, J2K_INVALID_ARGUMENT);
    assert_eq!(decode(&encoded, &mut output, 16).code, J2K_OK);
    // SAFETY: A null output with a nonzero length is rejected before any write.
    let status = unsafe {
        j2k_jpegls_decode(
            encoded.as_ptr(),
            encoded.len(),
            ptr::null_mut(),
            96,
            16,
            ptr::null_mut(),
        )
    };
    assert_eq!(status, J2K_INVALID_ARGUMENT);
}

#[test]
fn inputs_that_are_not_jpegls_are_cut_off_or_unsupported_are_rejected() {
    let (_, encoded) = small_image();
    let mut output = vec![0_u8; 96];
    assert_eq!(inspect(b"not an image"), Err(J2K_INVALID_INPUT));
    for cut in [2, 20, encoded.len() - 2] {
        assert_eq!(
            decode(&encoded[..cut], &mut output, 16).code,
            J2K_INVALID_INPUT,
            "cut at {cut}"
        );
    }
    let baseline_jpeg = [
        0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x10, 0x00, 0x10, 0x01, 0x01, 0x11, 0x00,
    ];
    assert_eq!(inspect(&baseline_jpeg), Err(J2K_UNSUPPORTED));
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one table of every rejected encode option"
)]
fn encode_rejects_invalid_options_samples_and_slots() {
    let (source, _) = small_image();
    let base = options(8, 6, 1, 12);
    for (options, samples, context) in [
        (
            J2kCJpegLsEncodeOptions {
                reserved: 1,
                ..base
            },
            &source[..],
            "reserved",
        ),
        (
            J2kCJpegLsEncodeOptions {
                interleave: 3,
                ..base
            },
            &source[..],
            "interleave",
        ),
        (
            J2kCJpegLsEncodeOptions {
                restart_interval: 65_536,
                ..base
            },
            &source[..],
            "restart",
        ),
        (
            J2kCJpegLsEncodeOptions {
                bit_depth: 17,
                ..base
            },
            &source[..],
            "bit depth 17",
        ),
        (
            J2kCJpegLsEncodeOptions {
                bit_depth: 1,
                ..base
            },
            &source[..48],
            "bit depth 1",
        ),
        (
            J2kCJpegLsEncodeOptions {
                bit_depth: 256,
                ..base
            },
            &source[..],
            "bit depth 256",
        ),
        (
            J2kCJpegLsEncodeOptions {
                components: 256,
                ..base
            },
            &source[..],
            "components",
        ),
        (
            J2kCJpegLsEncodeOptions { near: 256, ..base },
            &source[..],
            "near 256",
        ),
        (
            J2kCJpegLsEncodeOptions { near: 2048, ..base },
            &source[..],
            "near above MAXVAL / 2",
        ),
        (
            J2kCJpegLsEncodeOptions { width: 0, ..base },
            &source[..],
            "width 0",
        ),
        (
            J2kCJpegLsEncodeOptions {
                width: 65_536,
                ..base
            },
            &source[..],
            "width 65536",
        ),
        (base, &source[..source.len() - 1], "odd byte count"),
        (base, &source[..source.len() - 2], "sample count"),
        (
            J2kCJpegLsEncodeOptions {
                bit_depth: 4,
                ..options(8, 6, 1, 4)
            },
            &[200_u8; 48][..],
            "sample above MAXVAL",
        ),
    ] {
        assert_eq!(
            encode(samples, &options),
            Err(J2K_INVALID_ARGUMENT),
            "{context}"
        );
    }
    let mut stale = base;
    stale.header.abi_version = 2;
    assert_eq!(encode(&source, &stale), Err(J2K_ABI_MISMATCH));
    let (mut data, mut length) = (ptr::dangling_mut::<u8>(), 7_usize);
    // SAFETY: Null options are rejected; the output slots are live and cleared.
    let status = unsafe {
        j2k_jpegls_encode(
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
    // SAFETY: Null output slots are rejected before any write.
    let status = unsafe {
        j2k_jpegls_encode(
            source.as_ptr(),
            source.len(),
            &raw const base,
            ptr::null_mut(),
            &raw mut length,
            ptr::null_mut(),
        )
    };
    assert_eq!(status, J2K_INVALID_ARGUMENT);
}
