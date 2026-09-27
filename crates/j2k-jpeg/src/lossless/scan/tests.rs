// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec;
use alloc::vec::Vec;

use super::*;
use crate::entropy::huffman::HuffmanTable;
use crate::parse::tables::{HuffmanTableRole, HuffmanValues, RawHuffmanTable};

/// DC table giving every category 0..=16 a 5-bit code equal to the category.
fn category_table() -> HuffmanTable {
    let mut bits = [0u8; 16];
    bits[4] = 17;
    let values: Vec<u8> = (0..=16).collect();
    HuffmanTable::from_raw(
        &RawHuffmanTable {
            bits,
            values: HuffmanValues::from_slice(&values),
        },
        HuffmanTableRole::Dc,
    )
    .expect("17 five-bit codes fit")
}

#[derive(Default)]
struct BitWriter {
    bytes: Vec<u8>,
    acc: u32,
    bits: u8,
}

impl BitWriter {
    fn put(&mut self, value: u32, count: u8) {
        for bit in (0..count).rev() {
            self.acc = (self.acc << 1) | ((value >> bit) & 1);
            self.bits += 1;
            if self.bits == 8 {
                self.push_byte();
            }
        }
    }

    fn push_byte(&mut self) {
        let byte = u8::try_from(self.acc & 0xff).expect("masked byte");
        self.bytes.push(byte);
        if byte == 0xff {
            self.bytes.push(0);
        }
        self.acc = 0;
        self.bits = 0;
    }

    /// Pad to a byte boundary with one bits, as T.81 F.1.2.3 requires.
    fn align(&mut self) {
        while self.bits != 0 {
            self.put(1, 1);
        }
    }

    /// Encode one difference, interpreted modulo 2^16 like T.81 H.1.2.2.
    fn diff(&mut self, diff: i32) {
        let wrapped = diff.rem_euclid(1 << 16);
        let signed = if wrapped > 0x8000 {
            wrapped - 0x1_0000
        } else {
            wrapped
        };
        if signed == 0x8000 {
            self.put(16, 5);
            return;
        }
        let category = u8::try_from(32 - signed.unsigned_abs().leading_zeros()).expect("category");
        self.put(u32::from(category), 5);
        if category != 0 {
            let magnitude = if signed < 0 { signed - 1 } else { signed };
            let mask = (1u32 << category) - 1;
            self.put(u32::from_ne_bytes(magnitude.to_ne_bytes()) & mask, category);
        }
    }

    fn finish(mut self) -> Vec<u8> {
        self.align();
        self.bytes.extend_from_slice(&[0xff, 0xd9]);
        self.bytes
    }
}

/// Reference T.81 H.1.2.1 prediction with libjpeg-turbo's modulo-2^16
/// arithmetic, evaluated independently of the row kernels.
fn reference_predict(predictor: u8, ra: u16, rb: u16, rc: u16) -> u16 {
    let (ra, rb, rc) = (i32::from(ra), i32::from(rb), i32::from(rc));
    let prediction = match predictor {
        1 => ra,
        2 => rb,
        3 => rc,
        4 => ra + rb - rc,
        5 => ra + ((rb - rc) >> 1),
        6 => rb + ((ra - rc) >> 1),
        _ => (ra + rb) >> 1,
    };
    u16::try_from(prediction.rem_euclid(1 << 16)).expect("reduced prediction")
}

fn pseudo_random_row(seed: u32, len: usize) -> Vec<u16> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            u16::try_from(state >> 16).expect("16-bit sample")
        })
        .collect()
}

#[test]
fn first_row_accumulates_from_the_initial_predictor_modulo_2_16() {
    let mut out = [0u16; 4];
    undifference_first_row(&[5, 0xffff, 3, 0x8000], &mut out, 128);
    assert_eq!(out, [133, 132, 135, 0x8087]);
}

#[test]
fn later_rows_start_from_rb_then_apply_each_predictor() {
    let above = pseudo_random_row(7, 33);
    let diffs = pseudo_random_row(11, 33);
    for predictor in 1..=7 {
        let mut out = vec![0u16; 33];
        undifference_row(predictor, &diffs, &above, &mut out);
        let mut expected = vec![0u16; 33];
        expected[0] = above[0].wrapping_add(diffs[0]);
        for x in 1..33 {
            let prediction = reference_predict(predictor, expected[x - 1], above[x], above[x - 1]);
            expected[x] = prediction.wrapping_add(diffs[x]);
        }
        assert_eq!(out, expected, "predictor {predictor}");
    }
}

#[test]
fn predictors_read_left_above_and_diagonal_neighbours() {
    // Ra = 30, Rb = 20, Rc = 10 for the second sample of the lower row.
    let above = [10u16, 20];
    let expected = [
        (1, 30),
        (2, 20),
        (3, 10),
        (4, 40),
        (5, 35),
        (6, 30),
        (7, 25),
    ];
    for (predictor, prediction) in expected {
        let mut out = [0u16; 2];
        undifference_row(predictor, &[20, 0], &above, &mut out);
        assert_eq!(out, [30, prediction], "predictor {predictor}");
    }
}

#[test]
fn predictions_wrap_modulo_2_16() {
    // Rb = 0 plus 65535 gives Ra = 65535; predictor 4 then gives
    // 65535 + 65535 - 0, which wraps to 65534, and +3 wraps to 1.
    let mut out = [0u16; 2];
    undifference_row(4, &[0xffff, 3], &[0, 0xffff], &mut out);
    assert_eq!(out, [0xffff, 1]);
}

#[test]
fn category_16_decodes_as_32768_without_extra_bits() {
    let table = category_table();
    let table = table.dc().expect("DC table");
    let mut writer = BitWriter::default();
    writer.diff(0x8000);
    writer.diff(-1);
    let bytes = writer.finish();
    let mut br = BitReader::new(&bytes);
    assert_eq!(table.decode_lossless_diff(&mut br), Ok(0x8000));
    assert_eq!(table.decode_lossless_diff(&mut br), Ok(0xffff));
}

fn scan_samples(
    table: &HuffmanTable,
    bytes: &[u8],
    spec: LosslessScanSpec,
) -> Result<Vec<u16>, JpegError> {
    let dc = table.dc()?;
    let mut scan = LosslessScan::new(
        bytes,
        spec,
        &[LosslessComponentSpec {
            h: 1,
            v: 1,
            table: dc,
        }],
    )?;
    let mut samples = Vec::new();
    while scan.decode_mcu_row()? {
        samples.extend_from_slice(scan.row(0, 0));
    }
    scan.finish()?;
    Ok(samples)
}

/// Encode a grayscale image with T.81 prediction, restarting every
/// `restart_rows` rows (0 disables restarts).
fn encode_gray(
    samples: &[u16],
    width: usize,
    predictor: u8,
    bias: u16,
    restart_rows: usize,
) -> Vec<u8> {
    let mut writer = BitWriter::default();
    let rows: Vec<&[u16]> = samples.chunks(width).collect();
    let mut rst = 0u8;
    for (y, row) in rows.iter().enumerate() {
        let interval_row = if restart_rows == 0 {
            y
        } else {
            y % restart_rows
        };
        if restart_rows != 0 && y != 0 && interval_row == 0 {
            writer.align();
            writer.bytes.extend_from_slice(&[0xff, 0xd0 | rst]);
            rst = (rst + 1) & 7;
        }
        for (x, &sample) in row.iter().enumerate() {
            let prediction = match (interval_row, x) {
                (0, 0) => bias,
                (0, _) => row[x - 1],
                (_, 0) => rows[y - 1][0],
                _ => reference_predict(predictor, row[x - 1], rows[y - 1][x], rows[y - 1][x - 1]),
            };
            writer.diff(i32::from(sample) - i32::from(prediction));
        }
    }
    writer.finish()
}

fn gray_spec(width: u32, height: u32, predictor: u8, precision: u8) -> LosslessScanSpec {
    LosslessScanSpec {
        dimensions: (width, height),
        predictor,
        precision,
        point_transform: 0,
        restart_interval: None,
    }
}

#[test]
fn sixteen_bit_scan_round_trips_wrapping_differences_for_every_predictor() {
    let table = category_table();
    let (width, height) = (9usize, 5usize);
    let mut samples = pseudo_random_row(3, width * height);
    samples[0] = 0;
    samples[1] = 0xffff;
    samples[width] = 0x8000;
    for predictor in 1..=7 {
        let bytes = encode_gray(&samples, width, predictor, 0x8000, 0);
        let decoded = scan_samples(&table, &bytes, gray_spec(9, 5, predictor, 16));
        assert_eq!(decoded, Ok(samples.clone()), "predictor {predictor}");
    }
}

#[test]
fn restart_intervals_restart_the_one_dimensional_first_row() {
    let table = category_table();
    let (width, height) = (6usize, 7usize);
    let samples: Vec<u16> = pseudo_random_row(5, width * height)
        .into_iter()
        .map(|sample| sample >> 8)
        .collect();
    for predictor in 1..=7 {
        for restart_rows in [1usize, 2, 3] {
            let bytes = encode_gray(&samples, width, predictor, 128, restart_rows);
            let spec = LosslessScanSpec {
                restart_interval: Some(u16::try_from(restart_rows * width).expect("interval")),
                ..gray_spec(6, 7, predictor, 8)
            };
            let decoded = scan_samples(&table, &bytes, spec);
            assert_eq!(
                decoded,
                Ok(samples.clone()),
                "predictor {predictor}, restart every {restart_rows} rows"
            );
        }
    }
}

#[test]
fn restart_intervals_must_cover_whole_rows() {
    let table = category_table();
    let spec = LosslessScanSpec {
        restart_interval: Some(4),
        ..gray_spec(6, 2, 1, 8)
    };
    let dc = table.dc().expect("DC table");
    let result = LosslessScan::new(
        &[0xff, 0xd9],
        spec,
        &[LosslessComponentSpec {
            h: 1,
            v: 1,
            table: dc,
        }],
    );
    assert!(matches!(
        result,
        Err(JpegError::NotImplemented {
            sof: SofKind::Lossless
        })
    ));
}

#[test]
fn samples_wider_than_the_precision_are_rejected() {
    let table = category_table();
    // 128 + 200 = 328 does not fit an 8-bit sample.
    let mut writer = BitWriter::default();
    writer.diff(200);
    let bytes = writer.finish();
    let decoded = scan_samples(&table, &bytes, gray_spec(1, 1, 1, 8));
    assert_eq!(
        decoded,
        Err(JpegError::HuffmanDecode {
            mcu: 0,
            reason: HuffmanFailure::InvalidSymbol,
        })
    );
}

#[test]
fn point_transform_scales_the_initial_predictor() {
    let table = category_table();
    // P = 12, Pt = 4: the first sample predicts 2^(12-4-1) = 128.
    let mut writer = BitWriter::default();
    writer.diff(1);
    let bytes = writer.finish();
    let spec = LosslessScanSpec {
        point_transform: 4,
        ..gray_spec(1, 1, 1, 12)
    };
    let dc = table.dc().expect("DC table");
    let mut scan = LosslessScan::new(
        &bytes,
        spec,
        &[LosslessComponentSpec {
            h: 1,
            v: 1,
            table: dc,
        }],
    )
    .expect("scan");
    assert_eq!(scan.decode_mcu_row(), Ok(true));
    assert_eq!(scan.row(0, 0), &[129]);
    assert_eq!(scan.point_transform(), 4);
}

#[test]
fn allocation_bytes_cover_padded_mcu_rows() {
    let interleaved =
        lossless_scan_allocation_bytes((5, 3), &[(2, 2), (1, 1), (1, 1)]).expect("layout");
    // Luma: 2 rows of 3 MCUs x 2 diffs, 3 rows of 5 samples; chroma: 1 row of
    // 3 diffs, 2 rows of 3 samples each.
    let samples = (12 + 15) + 2 * (3 + 6);
    assert_eq!(
        interleaved,
        3 * core::mem::size_of::<ComponentRows<'_>>() + samples * 2
    );
    let gray = lossless_scan_allocation_bytes((5, 3), &[(2, 2)]).expect("layout");
    assert_eq!(
        gray,
        core::mem::size_of::<ComponentRows<'_>>() + (5 + 10) * 2
    );
}

#[test]
fn stored_samples_are_little_endian_and_shifted_by_the_point_transform() {
    use crate::lossless::LosslessSample;

    let mut gray16 = [0u8; 4];
    <u16 as LosslessSample>::store_row(&[0x1234, 0x0abc], 4, &mut gray16);
    assert_eq!(gray16, [0x40, 0x23, 0xc0, 0xab]);

    let mut rgb8 = [0u8; 6];
    <u8 as LosslessSample>::store_interleaved([&[1, 4], &[2, 5], &[3, 6]], 1, &mut rgb8);
    assert_eq!(rgb8, [2, 4, 6, 8, 10, 12]);

    let mut rgb16 = [0u8; 6];
    <u16 as LosslessSample>::store_interleaved([&[0x0102], &[0x0304], &[0x0506]], 0, &mut rgb16);
    assert_eq!(rgb16, [0x02, 0x01, 0x04, 0x03, 0x06, 0x05]);
}
