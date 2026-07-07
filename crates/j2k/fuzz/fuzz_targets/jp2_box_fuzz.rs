#![no_main]

use j2k::J2kDecoder;
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 64 * 1024;
const MAX_WRAPPED_BYTES: usize = 128 * 1024;
const BOX_TYPES: &[[u8; 4]] = &[
    *b"jp2h", *b"colr", *b"cdef", *b"pclr", *b"cmap", *b"res ", *b"uuid", *b"jp2c",
];

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }

    let _ = J2kDecoder::inspect(data);

    let mut wrapped = Vec::with_capacity(data.len().saturating_add(64).min(MAX_WRAPPED_BYTES));
    push_jp2_prefix(&mut wrapped, brand_from(data.first().copied().unwrap_or_default()));

    let mut offset = 1;
    while offset < data.len() && wrapped.len() < MAX_WRAPPED_BYTES {
        let selector = data[offset];
        offset += 1;
        let ty = BOX_TYPES[usize::from(selector) % BOX_TYPES.len()];
        let extended = selector & 0x80 != 0;
        let payload_len = data
            .get(offset)
            .copied()
            .map_or(0, |len| usize::from(len).min(data.len().saturating_sub(offset + 1)));
        offset = offset.saturating_add(1);
        let payload_end = offset.saturating_add(payload_len).min(data.len());
        push_box(&mut wrapped, ty, &data[offset..payload_end], extended);
        offset = payload_end;
    }

    let _ = J2kDecoder::inspect(&wrapped);
});

fn brand_from(value: u8) -> [u8; 4] {
    if value & 1 == 0 {
        *b"jp2 "
    } else {
        *b"jph "
    }
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
