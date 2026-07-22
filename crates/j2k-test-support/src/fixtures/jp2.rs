// SPDX-License-Identifier: MIT OR Apache-2.0

pub fn minimal_j2k_codestream() -> Vec<u8> {
    let mut bytes = vec![0xff, 0x4f];
    let mut siz = Vec::new();
    push_u16(&mut siz, 0);
    push_u32(&mut siz, 128);
    push_u32(&mut siz, 64);
    push_u32(&mut siz, 0);
    push_u32(&mut siz, 0);
    push_u32(&mut siz, 64);
    push_u32(&mut siz, 64);
    push_u32(&mut siz, 0);
    push_u32(&mut siz, 0);
    push_u16(&mut siz, 3);
    for _ in 0..3 {
        siz.extend_from_slice(&[0x07, 0x01, 0x01]);
    }
    bytes.extend_from_slice(&[0xff, 0x51]);
    push_u16(&mut bytes, segment_length_u16(siz.len()));
    bytes.extend_from_slice(&siz);

    let cod = [0x00, 0x00, 0x00, 0x01, 0x01, 0x05, 0x04, 0x04, 0x00, 0x01];
    bytes.extend_from_slice(&[0xff, 0x52]);
    push_u16(&mut bytes, segment_length_u16(cod.len()));
    bytes.extend_from_slice(&cod);
    bytes.extend_from_slice(&[0xff, 0x90, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
    bytes
}

/// Rewrites one component's SIZ sampling factors in a raw codestream fixture.
///
/// # Panics
///
/// Panics if `codestream` does not contain a SIZ marker or the requested
/// component descriptor is not present.
pub fn rewrite_j2k_component_sampling(
    codestream: &mut [u8],
    component: usize,
    x_rsiz: u8,
    y_rsiz: u8,
) {
    let siz = codestream
        .windows(2)
        .position(|marker| marker == [0xFF, 0x51])
        .expect("SIZ marker");
    let component_offset = siz + 40 + component * 3;
    codestream[component_offset + 1] = x_rsiz;
    codestream[component_offset + 2] = y_rsiz;
}

/// Minimal JP2 wrapper around [`minimal_j2k_codestream`].
pub fn minimal_jp2() -> Vec<u8> {
    wrap_jp2_codestream(&minimal_j2k_codestream(), 128, 64, 3, 8, 16)
}

/// Wraps a codestream in a JP2 container with an enumerated colorspace.
pub fn wrap_jp2_codestream(
    codestream: &[u8],
    width: u32,
    height: u32,
    components: u16,
    bit_depth: u8,
    colorspace_enum: u32,
) -> Vec<u8> {
    let mut bytes = jp2_prefix();
    let bpc = bit_depth.saturating_sub(1);
    bytes.extend_from_slice(&[
        0, 0, 0, 45, b'j', b'p', b'2', b'h', 0, 0, 0, 22, b'i', b'h', b'd', b'r',
    ]);
    bytes.extend_from_slice(&height.to_be_bytes());
    bytes.extend_from_slice(&width.to_be_bytes());
    bytes.extend_from_slice(&components.to_be_bytes());
    bytes.extend_from_slice(&[bpc, 7, 0, 0]);
    bytes.extend_from_slice(&[0, 0, 0, 15, b'c', b'o', b'l', b'r', 1, 0, 0]);
    bytes.extend_from_slice(&colorspace_enum.to_be_bytes());
    append_jp2c(&mut bytes, codestream);
    bytes
}

/// Wraps a four-component codestream in a JP2 container with an alpha channel.
pub fn wrap_jp2_rgba_codestream(
    codestream: &[u8],
    width: u32,
    height: u32,
    bit_depth: u8,
) -> Vec<u8> {
    let mut bytes = jp2_prefix();
    let bpc = bit_depth.saturating_sub(1);
    let jp2h_len = 8_u32 + 22 + 15 + 34;
    bytes.extend_from_slice(&jp2h_len.to_be_bytes());
    bytes.extend_from_slice(b"jp2h");
    bytes.extend_from_slice(&[0, 0, 0, 22, b'i', b'h', b'd', b'r']);
    bytes.extend_from_slice(&height.to_be_bytes());
    bytes.extend_from_slice(&width.to_be_bytes());
    bytes.extend_from_slice(&4_u16.to_be_bytes());
    bytes.extend_from_slice(&[bpc, 7, 0, 0]);
    bytes.extend_from_slice(&[0, 0, 0, 15, b'c', b'o', b'l', b'r', 1, 0, 0]);
    bytes.extend_from_slice(&16_u32.to_be_bytes());
    bytes.extend_from_slice(&[0, 0, 0, 34, b'c', b'd', b'e', b'f']);
    bytes.extend_from_slice(&4_u16.to_be_bytes());
    for (channel, channel_type, association) in [
        (0_u16, 0_u16, 1_u16),
        (1_u16, 0_u16, 2_u16),
        (2_u16, 0_u16, 3_u16),
        (3_u16, 1_u16, 0_u16),
    ] {
        bytes.extend_from_slice(&channel.to_be_bytes());
        bytes.extend_from_slice(&channel_type.to_be_bytes());
        bytes.extend_from_slice(&association.to_be_bytes());
    }
    append_jp2c(&mut bytes, codestream);
    bytes
}

fn jp2_prefix() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0, 0, 0, 12, b'j', b'P', b' ', b' ', 0x0d, 0x0a, 0x87, 0x0a]);
    bytes.extend_from_slice(&[
        0, 0, 0, 20, b'f', b't', b'y', b'p', b'j', b'p', b'2', b' ', 0, 0, 0, 0, b'j', b'p', b'2',
        b' ',
    ]);
    bytes
}

fn append_jp2c(bytes: &mut Vec<u8>, codestream: &[u8]) {
    let len = u32::try_from(
        codestream
            .len()
            .checked_add(8)
            .expect("JP2 codestream box length must not overflow usize"),
    )
    .expect("JP2 codestream box length must fit u32");
    bytes.extend_from_slice(&len.to_be_bytes());
    bytes.extend_from_slice(b"jp2c");
    bytes.extend_from_slice(codestream);
}

fn segment_length_u16(payload_len: usize) -> u16 {
    u16::try_from(
        payload_len
            .checked_add(2)
            .expect("marker segment length must not overflow usize"),
    )
    .expect("marker segment length must fit u16")
}

fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}
