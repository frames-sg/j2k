#![no_main]

use j2k::J2kDecoder;
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 64 * 1024;
const MAX_CODESTREAM_BYTES: usize = 32 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }

    let mut jp2h = Vec::new();
    push_box(&mut jp2h, *b"ihdr", &ihdr_payload(data), false);
    push_box(&mut jp2h, *b"colr", &colr_payload(data), data_flag(data, 0));
    push_box(&mut jp2h, *b"pclr", bounded_tail(data, 3, 96), data_flag(data, 1));
    push_box(&mut jp2h, *b"cmap", bounded_tail(data, 17, 96), data_flag(data, 2));
    push_box(&mut jp2h, *b"cdef", bounded_tail(data, 31, 96), data_flag(data, 3));

    let mut wrapped = Vec::with_capacity(data.len().saturating_add(128));
    push_jp2_prefix(&mut wrapped, brand_from(data.first().copied().unwrap_or_default()));
    push_box(&mut wrapped, *b"jp2h", &jp2h, data_flag(data, 4));
    push_box(
        &mut wrapped,
        *b"jp2c",
        bounded_tail(data, 47, MAX_CODESTREAM_BYTES),
        data_flag(data, 5),
    );

    let _ = J2kDecoder::inspect(&wrapped);
});

fn brand_from(value: u8) -> [u8; 4] {
    if value & 1 == 0 {
        *b"jp2 "
    } else {
        *b"jph "
    }
}

fn ihdr_payload(data: &[u8]) -> [u8; 14] {
    let width = 1 + u32::from(byte(data, 1));
    let height = 1 + u32::from(byte(data, 2));
    let components = 1 + u16::from(byte(data, 3) % 4);
    let bit_depth = byte(data, 4);
    let compression = byte(data, 5);
    let unknown_colorspace = byte(data, 6);
    let intellectual_property = byte(data, 7);

    let mut payload = [0_u8; 14];
    payload[0..4].copy_from_slice(&height.to_be_bytes());
    payload[4..8].copy_from_slice(&width.to_be_bytes());
    payload[8..10].copy_from_slice(&components.to_be_bytes());
    payload[10] = bit_depth;
    payload[11] = compression;
    payload[12] = unknown_colorspace;
    payload[13] = intellectual_property;
    payload
}

fn colr_payload(data: &[u8]) -> [u8; 7] {
    let colorspace = match byte(data, 8) % 3 {
        0 => 16_u32,
        1 => 17_u32,
        _ => 18_u32,
    };
    let mut payload = [0_u8; 7];
    payload[0] = 1;
    payload[1] = byte(data, 9);
    payload[2] = byte(data, 10);
    payload[3..7].copy_from_slice(&colorspace.to_be_bytes());
    payload
}

fn bounded_tail(data: &[u8], start: usize, max_len: usize) -> &[u8] {
    if start >= data.len() {
        return &[];
    }
    let len = usize::from(data[start]).min(max_len);
    let payload_start = start.saturating_add(1);
    let payload_end = payload_start.saturating_add(len).min(data.len());
    &data[payload_start..payload_end]
}

fn data_flag(data: &[u8], offset: usize) -> bool {
    byte(data, offset) & 0x80 != 0
}

fn byte(data: &[u8], offset: usize) -> u8 {
    data.get(offset).copied().unwrap_or_default()
}

fn push_jp2_prefix(out: &mut Vec<u8>, brand: [u8; 4]) {
    push_u32(out, 12);
    out.extend_from_slice(b"jP  ");
    out.extend_from_slice(&[0x0d, 0x0a, 0x87, 0x0a]);

    push_u32(out, 20);
    out.extend_from_slice(b"ftyp");
    out.extend_from_slice(&brand);
    push_u32(out, 0);
    out.extend_from_slice(&brand);
}

fn push_box(out: &mut Vec<u8>, ty: [u8; 4], payload: &[u8], extended: bool) {
    if extended {
        push_u32(out, 1);
        out.extend_from_slice(&ty);
        push_u64(out, payload.len().saturating_add(16) as u64);
    } else {
        push_u32(out, payload.len().saturating_add(8) as u32);
        out.extend_from_slice(&ty);
    }
    out.extend_from_slice(payload);
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_be_bytes());
}
