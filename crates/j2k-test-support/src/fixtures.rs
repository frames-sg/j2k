// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared byte fixtures and container builders for integration tests.

pub const JPEG_BASELINE_420_16X16: &[u8] =
    include_bytes!("../fixtures/conformance/baseline_420_16x16.jpg");
pub const JPEG_BASELINE_420_16X16_RGB: &[u8] =
    include_bytes!("../fixtures/conformance/baseline_420_16x16.rgb");
pub const JPEG_GRAYSCALE_8X8: &[u8] = include_bytes!("../fixtures/conformance/grayscale_8x8.jpg");
pub const JPEG_GRAYSCALE_8X8_GRAY: &[u8] =
    include_bytes!("../fixtures/conformance/grayscale_8x8.gray");
pub const JPEG_BASELINE_444_8X8: &[u8] =
    include_bytes!("../fixtures/conformance/baseline_444_8x8.jpg");
pub const JPEG_BASELINE_444_8X8_RGB: &[u8] =
    include_bytes!("../fixtures/conformance/baseline_444_8x8.rgb");
pub const JPEG_BASELINE_422_16X8: &[u8] =
    include_bytes!("../fixtures/conformance/baseline_422_16x8.jpg");
pub const JPEG_BASELINE_422_16X8_RGB: &[u8] =
    include_bytes!("../fixtures/conformance/baseline_422_16x8.rgb");
pub const JPEG_BASELINE_420_RESTART_32X16: &[u8] =
    include_bytes!("../fixtures/conformance/baseline_420_restart_32x16.jpg");
pub const JPEG_BASELINE_420_RESTART_32X16_RGB: &[u8] =
    include_bytes!("../fixtures/conformance/baseline_420_restart_32x16.rgb");

/// Minimal grayscale JPEG with caller-provided dimensions and one entropy byte.
///
/// This is intentionally not a complete image for large dimensions; tests use
/// it to exercise header validation and row-streaming paths without allocating
/// full-image entropy payloads.
pub fn minimal_grayscale_jpeg_with_dimensions(width: u16, height: u16) -> Vec<u8> {
    let mut bytes = grayscale_jpeg_header(width, height);
    bytes.extend_from_slice(&[0x00, 0xff, 0xd9]);
    bytes
}

/// Baseline grayscale JPEG with one zero-DC entropy byte per MCU.
pub fn baseline_grayscale_jpeg(width: u16, height: u16) -> Vec<u8> {
    let mut bytes = grayscale_jpeg_header(width, height);
    let mcu_cols = u32::from(width).div_ceil(8);
    let mcu_rows = u32::from(height).div_ceil(8);
    let mcu_count = (mcu_cols * mcu_rows) as usize;
    bytes.extend(core::iter::repeat_n(0x00, mcu_count));
    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

/// Minimal 16x16 baseline JPEG with 4:2:0 sampling.
pub fn minimal_baseline_jpeg() -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&[0xff, 0xd8]);
    out.extend_from_slice(&[0xff, 0xdb, 0x00, 67, 0x00]);
    out.extend(core::iter::repeat_n(1u8, 64));
    out.extend_from_slice(&[
        0xff, 0xc0, 0x00, 17, 8, 0, 16, 0, 16, 3, 1, 0x22, 0, 2, 0x11, 0, 3, 0x11, 0,
    ]);
    out.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xaa,
    ]);
    out.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xbb,
    ]);
    out.extend_from_slice(&[0xff, 0xda, 0x00, 12, 3, 1, 0x00, 2, 0x00, 3, 0x00, 0, 63, 0]);
    out.extend_from_slice(&[0x00, 0xff, 0xd9]);
    out
}

/// Minimal baseline JPEG with a DRI marker inserted before SOS.
///
/// # Panics
///
/// Panics if the generated minimal fixture no longer contains an SOS marker.
pub fn minimal_baseline_jpeg_with_restart_interval(interval: u16) -> Vec<u8> {
    let mut bytes = minimal_baseline_jpeg();
    let sos_pos = bytes
        .windows(2)
        .position(|window| window == [0xff, 0xda])
        .expect("minimal fixture includes SOS marker");
    let [interval_high, interval_low] = interval.to_be_bytes();
    bytes.splice(
        sos_pos..sos_pos,
        [0xff, 0xdd, 0x00, 0x04, interval_high, interval_low],
    );
    bytes
}

/// Restart-coded grayscale JPEG with one zero-DC block per MCU.
pub fn restart_coded_grayscale_jpeg(width: u16, height: u16) -> Vec<u8> {
    let mut bytes = grayscale_jpeg_prefix(width, height);
    bytes.extend_from_slice(&[0xff, 0xdd, 0x00, 0x04, 0x00, 0x01]);
    append_grayscale_huffman_and_scan_header(&mut bytes);

    let mcu_cols = u32::from(width).div_ceil(8);
    let mcu_rows = u32::from(height).div_ceil(8);
    let mcu_count = (mcu_cols * mcu_rows) as usize;
    for mcu in 0..mcu_count {
        bytes.push(0x00);
        if mcu + 1 != mcu_count {
            bytes.extend_from_slice(&[0xff, 0xd0 | restart_index(mcu)]);
        }
    }

    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

fn grayscale_jpeg_header(width: u16, height: u16) -> Vec<u8> {
    let mut bytes = grayscale_jpeg_prefix(width, height);
    append_grayscale_huffman_and_scan_header(&mut bytes);
    bytes
}

fn restart_index(mcu: usize) -> u8 {
    u8::try_from(mcu & 0x07).expect("restart index is three bits")
}

fn grayscale_jpeg_prefix(width: u16, height: u16) -> Vec<u8> {
    let mut bytes = Vec::new();
    let [height_high, height_low] = height.to_be_bytes();
    let [width_high, width_low] = width.to_be_bytes();
    bytes.extend_from_slice(&[0xff, 0xd8]);
    bytes.extend_from_slice(&[0xff, 0xdb, 0x00, 67, 0x00]);
    bytes.extend(core::iter::repeat_n(16u8, 64));
    bytes.extend_from_slice(&[
        0xff,
        0xc0,
        0x00,
        11,
        8,
        height_high,
        height_low,
        width_high,
        width_low,
        1,
        1,
        0x11,
        0,
    ]);
    bytes
}

fn append_grayscale_huffman_and_scan_header(bytes: &mut Vec<u8>) {
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x00, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    bytes.extend_from_slice(&[0xff, 0xda, 0x00, 0x08, 1, 1, 0x00, 0, 63, 0]);
}

/// Minimal raw J2K codestream containing SIZ, COD, and SOT markers.
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

#[cfg(feature = "j2k-native-fixtures")]
fn htj2k_options(reversible: bool) -> j2k_native::EncodeOptions {
    j2k_native::EncodeOptions {
        reversible,
        num_decomposition_levels: 1,
        ..j2k_native::EncodeOptions::default()
    }
}

#[cfg(feature = "j2k-native-fixtures")]
fn encode_htj2k_fixture(
    pixels: &[u8],
    width: u32,
    height: u32,
    components: u8,
    reversible: bool,
) -> Vec<u8> {
    j2k_native::encode_htj2k(
        pixels,
        width,
        height,
        u16::from(components),
        8,
        false,
        &htj2k_options(reversible),
    )
    .expect("encode HTJ2K fixture")
}

#[cfg(feature = "j2k-native-fixtures")]
/// Deterministic reversible HTJ2K grayscale fixture.
pub fn htj2k_gray8_fixture(width: u32, height: u32) -> Vec<u8> {
    let pixels = (0..width * height)
        .map(|idx| (idx & 0xff) as u8)
        .collect::<Vec<_>>();
    encode_htj2k_fixture(&pixels, width, height, 1, true)
}

#[cfg(feature = "j2k-native-fixtures")]
/// Deterministic irreversible 9/7 HTJ2K grayscale fixture.
pub fn htj2k_gray8_97_fixture(width: u32, height: u32) -> Vec<u8> {
    let pixels = (0..width * height)
        .map(|idx| ((idx * 11) & 0xff) as u8)
        .collect::<Vec<_>>();
    encode_htj2k_fixture(&pixels, width, height, 1, false)
}

#[cfg(feature = "j2k-native-fixtures")]
/// Larger reversible grayscale HTJ2K fixture with 64x64 code blocks.
///
/// # Panics
///
/// Panics if the native encoder rejects the deterministic fixture input.
pub fn htj2k_gray8_large_fixture(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 3 + y * 5) & 0xff) as u8);
        }
    }
    let options = j2k_native::EncodeOptions {
        reversible: true,
        num_decomposition_levels: 3,
        code_block_width_exp: 0,
        code_block_height_exp: 0,
        ..j2k_native::EncodeOptions::default()
    };
    j2k_native::encode_htj2k(&pixels, width, height, 1, 8, false, &options)
        .expect("encode large HTJ2K grayscale fixture")
}

#[cfg(feature = "j2k-native-fixtures")]
/// Deterministic reversible HTJ2K RGB fixture.
pub fn htj2k_rgb8_fixture(width: u32, height: u32) -> Vec<u8> {
    htj2k_rgb8_fixture_with_pixels(width, height).0
}

#[cfg(feature = "j2k-native-fixtures")]
/// Deterministic reversible HTJ2K RGB fixture and its source pixels.
pub fn htj2k_rgb8_fixture_with_pixels(width: u32, height: u32) -> (Vec<u8>, Vec<u8>) {
    let pixels = (0u32..width * height * 3)
        .map(|idx| ((idx * 13 + idx / 3) & 0xff) as u8)
        .collect::<Vec<_>>();
    let codestream = encode_htj2k_fixture(&pixels, width, height, 3, true);
    (codestream, pixels)
}

#[cfg(feature = "j2k-native-fixtures")]
/// Seeded reversible HTJ2K RGB fixture.
pub fn htj2k_rgb8_pattern_fixture(width: u32, height: u32, seed: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
    for idx in 0..width * height {
        pixels.push(((idx * seed + idx / 3) & 0xff) as u8);
        pixels.push(((idx * (seed + 11) + 7) & 0xff) as u8);
        pixels.push(((idx * (seed + 23) + 19) & 0xff) as u8);
    }
    encode_htj2k_fixture(&pixels, width, height, 3, true)
}

#[cfg(feature = "j2k-native-fixtures")]
/// Deterministic irreversible 9/7 HTJ2K RGB fixture.
pub fn htj2k_rgb8_97_fixture(width: u32, height: u32) -> Vec<u8> {
    let pixels = (0u32..width * height * 3)
        .map(|idx| ((idx * 17 + idx / 5) & 0xff) as u8)
        .collect::<Vec<_>>();
    encode_htj2k_fixture(&pixels, width, height, 3, false)
}

#[cfg(feature = "j2k-native-fixtures")]
/// Deterministic classic J2K grayscale fixture.
///
/// # Panics
///
/// Panics if the native encoder rejects the deterministic fixture input.
pub fn classic_j2k_gray8_fixture(width: u32, height: u32) -> Vec<u8> {
    let pixels = (0..width * height)
        .map(|idx| (idx & 0xff) as u8)
        .collect::<Vec<_>>();
    let options = j2k_native::EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..j2k_native::EncodeOptions::default()
    };
    j2k_native::encode(&pixels, width, height, 1, 8, false, &options)
        .expect("encode classic J2K grayscale fixture")
}

/// `OpenHTJ2K` refinement fixture with a compact output plane.
pub fn openhtj2k_refinement_fixture() -> &'static [u8] {
    include_bytes!("../fixtures/htj2k/openhtj2k_ds0_ht_12_b11.j2k")
}

/// Expected grayscale pixels for [`openhtj2k_refinement_fixture`].
pub fn openhtj2k_refinement_pixels() -> &'static [u8] {
    include_bytes!("../fixtures/htj2k/openhtj2k_ds0_ht_12_b11.gray")
}

/// `OpenHTJ2K` odd refinement fixture used by CUDA plan tests.
pub fn openhtj2k_refinement_odd_fixture() -> &'static [u8] {
    include_bytes!("../fixtures/htj2k/openhtj2k_ds0_ht_09_b11.j2k")
}

/// Expected grayscale pixels for [`openhtj2k_refinement_odd_fixture`].
pub fn openhtj2k_refinement_odd_pixels() -> &'static [u8] {
    include_bytes!("../fixtures/htj2k/openhtj2k_ds0_ht_09_b11.gray")
}
