#![no_main]

//! TIFF PackBits and DICOM RLE decoding of arbitrary bytes. Neither may
//! panic. PackBits must decode exactly the length `expected_size` reports.
//! The first four bytes choose a DICOM RLE layout and the rest is the frame:
//! a frame the strict decoder accepts must decode the same leniently with a
//! conformant report, and must survive an encode and decode round trip.

use j2k_tilecodec::{
    DicomRleCodec, DicomRleLayout, DicomRlePlanarConfiguration, NoPool, PackBitsCodec,
    TileDecompress,
};
use libfuzzer_sys::fuzz_target;

const SAMPLES: [u16; 6] = [1, 3, 2, 4, 5, 15];
const BITS: [u16; 4] = [8, 16, 32, 12];

fn packbits(data: &[u8]) {
    let mut out = vec![0_u8; 4096];
    let decoded = PackBitsCodec::decompress_into(&mut NoPool, data, &mut out);
    match PackBitsCodec::expected_size(data) {
        Ok(Some(len)) if len <= out.len() => {
            assert_eq!(
                decoded.ok(),
                Some(len),
                "decoded length differs from expected_size"
            );
        }
        Ok(_) => assert!(decoded.is_err(), "decoded past the output buffer"),
        Err(_) => assert!(decoded.is_err(), "decoded a payload expected_size rejects"),
    }
}

fn dicom_rle(data: &[u8]) {
    let Some((head, frame)) = data.split_first_chunk::<4>() else {
        return;
    };
    let rows = 1 + u16::from(head[0] % 32);
    let columns = 1 + u16::from(head[1] % 64);
    let samples = SAMPLES[usize::from(head[2] % 6)];
    let bits = BITS[usize::from(head[3] % 4)];
    let planar = if head[3] & 0x80 == 0 {
        DicomRlePlanarConfiguration::Interleaved
    } else {
        DicomRlePlanarConfiguration::Planar
    };
    let Ok(layout) = DicomRleLayout::new(rows, columns, samples, bits, planar) else {
        return;
    };

    let mut strict = vec![0_u8; layout.frame_len()];
    let strict_result = DicomRleCodec::decode_frame(frame, layout, &mut strict);
    let mut lenient = vec![0_u8; layout.frame_len()];
    let lenient_result = DicomRleCodec::decode_frame_lenient(frame, layout, &mut lenient);
    if strict_result.is_err() {
        return;
    }
    let report = lenient_result.expect("lenient decoding rejected a strict frame");
    assert!(
        report.is_conformant(),
        "strict frame reported as non-conformant"
    );
    assert_eq!(strict, lenient, "strict and lenient decodes differ");

    let mut encoded = Vec::new();
    DicomRleCodec::encode_frame(&strict, layout, &mut encoded).expect("encode");
    let mut round_trip = vec![0_u8; layout.frame_len()];
    DicomRleCodec::decode_frame(&encoded, layout, &mut round_trip).expect("decode encoded");
    assert_eq!(strict, round_trip, "round trip changed the samples");
}

fuzz_target!(|data: &[u8]| {
    packbits(data);
    dicom_rle(data);
});
