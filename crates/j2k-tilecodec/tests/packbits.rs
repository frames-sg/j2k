// SPDX-License-Identifier: MIT OR Apache-2.0

//! TIFF `PackBits` (compression 32773) tile decompression.

use j2k_core::{BufferError, CodecError, ScratchPool, TileDecompress};
use j2k_tilecodec::{NoPool, PackBitsCodec, TileCodecError};

fn zeroed(len: usize) -> Vec<u8> {
    std::iter::repeat_n(0_u8, len).collect()
}

fn decode(input: &[u8], out_len: usize) -> Result<(usize, Vec<u8>), TileCodecError> {
    let mut out = zeroed(out_len);
    PackBitsCodec::decompress_into(&mut NoPool, input, &mut out).map(|written| (written, out))
}

/// The worked example from Apple Technical Note TN1023, which the TIFF 6.0
/// specification cites for `PackBits`.
const TN1023_PACKED: [u8; 15] = [
    0xFE, 0xAA, 0x02, 0x80, 0x00, 0x2A, 0xFD, 0xAA, 0x03, 0x80, 0x00, 0x2A, 0x22, 0xF7, 0xAA,
];
const TN1023_UNPACKED: [u8; 24] = [
    0xAA, 0xAA, 0xAA, 0x80, 0x00, 0x2A, 0xAA, 0xAA, 0xAA, 0xAA, 0x80, 0x00, 0x2A, 0x22, 0xAA, 0xAA,
    0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
];

#[test]
fn decodes_the_tn1023_example() {
    assert_eq!(
        PackBitsCodec::expected_size(&TN1023_PACKED).expect("size"),
        Some(TN1023_UNPACKED.len())
    );
    let (written, out) = decode(&TN1023_PACKED, TN1023_UNPACKED.len()).expect("decode");
    assert_eq!(written, TN1023_UNPACKED.len());
    assert_eq!(out, TN1023_UNPACKED);
}

#[test]
fn every_header_value_decodes_to_its_run() {
    for header in 0..=u8::MAX {
        let payload = (1..=128_u8).collect::<Vec<_>>();
        let input = std::iter::once(header)
            .chain(payload.iter().copied())
            .collect::<Vec<_>>();
        let (expected, consumed): (Vec<u8>, usize) = match header {
            0..=127 => {
                let len = usize::from(header) + 1;
                (Vec::from(&payload[..len]), 1 + len)
            }
            128 => (Vec::new(), 1),
            129..=255 => (
                std::iter::repeat_n(1, 257 - usize::from(header)).collect(),
                2,
            ),
        };
        let input = &input[..consumed];
        let (written, out) = decode(input, expected.len()).expect("decode");
        assert_eq!(written, expected.len(), "header {header}");
        assert_eq!(out, expected, "header {header}");
        assert_eq!(
            PackBitsCodec::expected_size(input).expect("size"),
            Some(expected.len()),
            "header {header}"
        );
    }
}

#[test]
fn empty_and_no_op_payloads_decode_to_nothing() {
    for input in [&[][..], &[0x80], &[0x80, 0x80, 0x80]] {
        assert_eq!(decode(input, 0).expect("decode").0, 0, "{input:?}");
        assert_eq!(PackBitsCodec::expected_size(input).expect("size"), Some(0));
    }
}

#[test]
fn a_larger_output_buffer_keeps_its_tail() {
    let mut out = std::iter::repeat_n(0xEE_u8, 30).collect::<Vec<_>>();
    let written =
        PackBitsCodec::decompress_into(&mut NoPool, &TN1023_PACKED, &mut out).expect("decode");
    assert_eq!(written, 24);
    assert_eq!(out[..24], TN1023_UNPACKED);
    assert!(out[24..].iter().all(|&byte| byte == 0xEE));
}

#[test]
fn output_overflow_reports_the_full_decoded_length() {
    for have in [0, 1, 3, 10, 23] {
        match decode(&TN1023_PACKED, have) {
            Err(TileCodecError::Buffer(BufferError::OutputTooSmall {
                required: 24,
                have: h,
            })) if h == have => {}
            other => panic!("{have}-byte output: {other:?}"),
        }
    }
    // A replicate run of 128 bytes into 127.
    match decode(&[0x81, 9], 127) {
        Err(error @ TileCodecError::Buffer(BufferError::OutputTooSmall { required: 128, .. })) => {
            assert!(error.is_buffer_error());
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn truncated_runs_are_truncated_input_errors() {
    for input in [
        &[0_u8][..],
        &[1, 7],
        &[127],
        &[0x81],
        &[0xFF],
        &[0xFE, 0xAA, 0x02, 0x80],
    ] {
        let error = decode(input, 64).expect_err("truncated run");
        assert!(error.is_truncated(), "{input:?}: {error:?}");
        let error = PackBitsCodec::expected_size(input).expect_err("truncated run");
        assert!(error.is_truncated(), "{input:?}: {error:?}");
    }
    // Truncation wins over overflow: the payload is broken either way.
    let error = decode(&[0x81, 9, 3, 1], 4).expect_err("overflow and truncation");
    assert!(error.is_truncated(), "{error:?}");
}

#[test]
fn every_prefix_decodes_to_a_prefix_of_the_output_or_is_truncated() {
    for cut in 0..TN1023_PACKED.len() {
        let prefix = &TN1023_PACKED[..cut];
        match decode(prefix, TN1023_UNPACKED.len()) {
            Ok((written, out)) => {
                // The cut fell between runs.
                assert_eq!(out[..written], TN1023_UNPACKED[..written], "cut {cut}");
                assert!(written < TN1023_UNPACKED.len(), "cut {cut}");
            }
            Err(error) => assert!(error.is_truncated(), "cut {cut}: {error:?}"),
        }
    }
}

#[test]
fn round_trips_runs_of_every_length() {
    // Literal and replicate runs of every length from 1 to 128 bytes, back
    // to back.
    let mut packed = Vec::new();
    let mut unpacked = Vec::new();
    for len in 1..=128_u8 {
        packed.push(len - 1);
        packed.extend((0..len).map(|index| index.wrapping_mul(37) ^ len));
        unpacked.extend((0..len).map(|index| index.wrapping_mul(37) ^ len));
        if len > 1 {
            packed.extend([0_u8.wrapping_sub(len - 1), len]);
            unpacked.extend(std::iter::repeat_n(len, usize::from(len)));
        }
    }
    let (written, out) = decode(&packed, unpacked.len()).expect("decode");
    assert_eq!(written, unpacked.len());
    assert_eq!(out, unpacked);
}

#[test]
fn uses_no_scratch_memory() {
    let mut pool = NoPool;
    let mut out = zeroed(24);
    PackBitsCodec::decompress_into(&mut pool, &TN1023_PACKED, &mut out).expect("decode");
    assert_eq!(pool.bytes_allocated(), 0);
}
