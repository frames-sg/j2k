// SPDX-License-Identifier: MIT OR Apache-2.0

//! Golden model for the HT `SigProp` forward reader's byte-stuffing rule.

const HT_CLEANUP_SOURCE: &str = include_str!("../ht_cleanup.metal");

#[derive(Default)]
struct ForwardReaderModel {
    reservoir: u64,
    bits: u32,
    unstuff: bool,
}

#[derive(Default)]
struct ReverseReaderModel {
    reservoir: u64,
    bits: u32,
    unstuff: bool,
}

impl ReverseReaderModel {
    fn push(&mut self, raw_byte: u8) {
        let stuffed = self.unstuff && (raw_byte & 0x7f) == 0x7f;
        let value = if stuffed { raw_byte & 0x7f } else { raw_byte };
        self.reservoir |= u64::from(value) << self.bits;
        self.bits += 8 - u32::from(stuffed);
        self.unstuff = raw_byte > 0x8f;
    }
}

impl ForwardReaderModel {
    fn push(&mut self, raw_byte: u8) {
        let valid_bits = 8 - u32::from(self.unstuff);
        let value = if self.unstuff {
            raw_byte & 0x7f
        } else {
            raw_byte
        };
        self.reservoir |= u64::from(value) << self.bits;
        self.bits += valid_bits;
        self.unstuff = raw_byte == 0xff;
    }
}

#[test]
fn sigprop_forward_reader_discards_stuffed_msb_from_overlap_byte() {
    let mut reader = ForwardReaderModel::default();
    for byte in [0xff, 0x80, 0x00, 0x00, 0x00] {
        reader.push(byte);
    }

    assert_eq!(reader.reservoir, 0x0000_00ff_u64);
    assert_eq!(reader.bits, 39);
    assert!(!reader.unstuff);
}

#[test]
fn metal_sigprop_reader_masks_value_but_uses_raw_byte_for_next_state() {
    assert!(
        HT_CLEANUP_SOURCE
            .contains("const uchar value = reader.unstuff ? raw_byte & uchar(0x7F) : raw_byte;"),
        "Metal forward reader must discard the stuffed MSB after 0xFF"
    );
    assert!(
        HT_CLEANUP_SOURCE.contains("reader.unstuff = raw_byte == uchar(0xFF);"),
        "next-byte unstuff state must be derived from the unmasked byte"
    );
}

#[test]
fn magref_reverse_reader_discards_stuffed_msb_from_shared_byte() {
    let mut reader = ReverseReaderModel {
        unstuff: true,
        ..ReverseReaderModel::default()
    };
    for byte in [0xff, 0x00, 0x00, 0x00, 0x00] {
        reader.push(byte);
    }

    assert_eq!(reader.reservoir, 0x0000_007f_u64);
    assert_eq!(reader.bits, 39);
    assert!(!reader.unstuff);
}

#[test]
fn metal_magref_reader_masks_stuffed_value_but_keeps_raw_next_state() {
    assert!(
        HT_CLEANUP_SOURCE
            .contains("const uchar value = stuffed ? raw_byte & uchar(0x7F) : raw_byte;"),
        "Metal reverse reader must discard its stuffed MSB"
    );
    assert!(
        HT_CLEANUP_SOURCE.contains("reader.unstuff = raw_byte > uchar(0x8F);"),
        "reverse next-byte unstuff state must be derived from the raw byte"
    );
}
