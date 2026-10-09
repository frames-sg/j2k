// SPDX-License-Identifier: MIT OR Apache-2.0

//! Signed grayscale output must match the CPU byte for byte. The CPU writes
//! signed sources of more than eight bits to Gray16 as two's complement; the
//! Metal fused store used to clamp negative samples to zero, which erased
//! every negative CT Hounsfield value on an explicit Metal Gray16 request.

use super::*;

const WIDTH: u32 = 67;
const HEIGHT: u32 = 45;

fn grayscale_fixture(bit_depth: u8, signed: bool, ht: bool) -> Vec<u8> {
    let range = 1_i32 << bit_depth;
    let offset = if signed { range / 2 } else { 0 };
    let samples = (0..WIDTH * HEIGHT).map(|index| {
        let index = i32::try_from(index).expect("fixture index fits i32");
        // Spans the full sample range, including both extremes.
        ((index * 7919) % range) - offset
    });
    let pixels = if bit_depth <= 8 {
        samples
            .map(|sample| sample.to_le_bytes()[0])
            .collect::<Vec<_>>()
    } else {
        samples
            .flat_map(|sample| {
                let bytes = sample.to_le_bytes();
                [bytes[0], bytes[1]]
            })
            .collect::<Vec<_>>()
    };
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 3,
        ..EncodeOptions::default()
    };
    if ht {
        encode_htj2k(&pixels, WIDTH, HEIGHT, 1, bit_depth, signed, &options)
            .expect("encode HTJ2K grayscale fixture")
    } else {
        encode(&pixels, WIDTH, HEIGHT, 1, bit_depth, signed, &options)
            .expect("encode classic grayscale fixture")
    }
}

#[test]
fn explicit_metal_grayscale_output_matches_cpu_for_signed_and_unsigned_sources() {
    if !should_run_metal_runtime() {
        return;
    }

    for (bit_depth, signed, fmt) in [
        (8, true, PixelFormat::Gray8),
        (8, true, PixelFormat::Gray16),
        (8, false, PixelFormat::Gray16),
        (12, true, PixelFormat::Gray16),
        (12, false, PixelFormat::Gray16),
        (16, true, PixelFormat::Gray16),
        (16, false, PixelFormat::Gray16),
    ] {
        let twos_complement = signed && bit_depth > 8;
        for ht in [true, false] {
            let label = format!("{bit_depth}-bit signed={signed} ht={ht} {fmt:?}");
            let bytes = grayscale_fixture(bit_depth, signed, ht);

            let mut host = j2k::J2kDecoder::new(&bytes)
                .unwrap_or_else(|error| panic!("{label}: host decoder: {error}"));
            let stride = WIDTH as usize * fmt.bytes_per_pixel();
            let mut expected = vec![0_u8; stride * HEIGHT as usize];
            host.decode_into(&mut expected, stride, fmt)
                .unwrap_or_else(|error| panic!("{label}: host decode: {error}"));
            if twos_complement {
                assert!(
                    expected.iter().skip(1).step_by(2).any(|&byte| byte >= 0x80),
                    "{label}: output must carry negative samples"
                );
            }

            let mut metal = J2kDecoder::new(&bytes)
                .unwrap_or_else(|error| panic!("{label}: Metal decoder: {error}"));
            let surface = metal
                .decode_request_to_device(MetalDecodeRequest::full(fmt, BackendRequest::Metal))
                .unwrap_or_else(|error| panic!("{label}: Metal decode: {error}"));
            let actual = surface
                .as_bytes()
                .unwrap_or_else(|error| panic!("{label}: Metal readback: {error}"));
            assert_eq!(actual.as_ref(), expected.as_slice(), "{label}");
        }
    }
}
