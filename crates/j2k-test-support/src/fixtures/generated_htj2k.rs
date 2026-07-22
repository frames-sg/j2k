// SPDX-License-Identifier: MIT OR Apache-2.0

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

/// Native sample profile for the shared generated RGBA HTJ2K fixture.
#[cfg(feature = "j2k-native-fixtures")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Htj2kRgbaSampleProfile {
    /// Unsigned eight-bit samples with reversible RCT enabled for RGB.
    U8Rct,
    /// Unsigned twelve-bit samples without MCT.
    U12,
    /// Signed sixteen-bit samples without MCT.
    I16,
}

/// Alpha interpretation callers must encode in the surrounding JPH/JP2 `cdef` box.
#[cfg(feature = "j2k-native-fixtures")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Htj2kRgbaAlpha {
    /// Unassociated (straight) opacity.
    Straight,
    /// Premultiplied opacity metadata. Source samples remain unchanged.
    Premultiplied,
}

/// Exact interleaved source samples for a generated RGBA HTJ2K fixture.
#[cfg(feature = "j2k-native-fixtures")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Htj2kRgbaSamples {
    /// Unsigned eight-bit RGBA samples.
    U8(Vec<u8>),
    /// Unsigned twelve-bit RGBA samples stored in `u16` values.
    U16(Vec<u16>),
    /// Signed sixteen-bit RGBA samples.
    I16(Vec<i16>),
}

/// Shared deterministic single-tile RGBA HTJ2K fixture and exact source oracle.
#[cfg(feature = "j2k-native-fixtures")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Htj2kRgbaFixture {
    /// Raw Part 15 codestream. Callers add identity RGBA `cdef` metadata.
    pub encoded: Vec<u8>,
    /// Exact interleaved source samples in R, G, B, A order.
    pub samples: Htj2kRgbaSamples,
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// Declared component precision.
    pub bit_depth: u8,
    /// Whether every component is signed.
    pub signed: bool,
    /// Whether reversible RCT is applied to RGB. Alpha is never transformed.
    pub use_mct: bool,
    /// Alpha interpretation callers must preserve in container metadata.
    pub alpha: Htj2kRgbaAlpha,
}

/// Generate the shared exact RGBA HTJ2K fixture used by CPU and GPU batch tests.
///
/// The raw codestream cannot carry alpha semantics. Callers must wrap `encoded`
/// in JPH/JP2 with identity R, G, B channel definitions followed by the requested
/// whole-image alpha interpretation.
///
/// # Panics
///
/// Panics if the native encoder rejects the deterministic reversible fixture.
#[cfg(feature = "j2k-native-fixtures")]
pub fn generated_htj2k_rgba_fixture(
    profile: Htj2kRgbaSampleProfile,
    alpha: Htj2kRgbaAlpha,
) -> Htj2kRgbaFixture {
    const WIDTH: u32 = 8;
    const HEIGHT: u32 = 8;
    const COMPONENTS: u16 = 4;
    let sample_count = WIDTH as usize * HEIGHT as usize * usize::from(COMPONENTS);
    let (samples, pixels, bit_depth, signed, use_mct) = match profile {
        Htj2kRgbaSampleProfile::U8Rct => {
            let values = (0..sample_count)
                .map(|index| {
                    u8::try_from((index * 37 + index / 3) & 0xff)
                        .expect("masked RGBA fixture sample must fit u8")
                })
                .collect::<Vec<_>>();
            (Htj2kRgbaSamples::U8(values.clone()), values, 8, false, true)
        }
        Htj2kRgbaSampleProfile::U12 => {
            let values = (0..sample_count)
                .map(|index| {
                    let index = u32::try_from(index).expect("RGBA fixture index must fit u32");
                    u16::try_from((index * 977 + 31) & 0x0fff)
                        .expect("masked RGBA fixture sample must fit u16")
                })
                .collect::<Vec<_>>();
            let pixels = values
                .iter()
                .flat_map(|sample| sample.to_le_bytes())
                .collect();
            (Htj2kRgbaSamples::U16(values), pixels, 12, false, false)
        }
        Htj2kRgbaSampleProfile::I16 => {
            let values = (0..sample_count)
                .map(|index| {
                    let index = i32::try_from(index).expect("RGBA fixture index must fit i32");
                    i16::try_from((index * 113 + 19) % 20_001 - 10_000)
                        .expect("bounded RGBA fixture sample must fit i16")
                })
                .collect::<Vec<_>>();
            let pixels = values
                .iter()
                .flat_map(|sample| sample.to_le_bytes())
                .collect();
            (Htj2kRgbaSamples::I16(values), pixels, 16, true, false)
        }
    };
    let options = j2k_native::EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        use_mct,
        ..j2k_native::EncodeOptions::default()
    };
    let encoded = j2k_native::encode_htj2k(
        &pixels, WIDTH, HEIGHT, COMPONENTS, bit_depth, signed, &options,
    )
    .expect("encode shared HTJ2K RGBA fixture");
    Htj2kRgbaFixture {
        encoded,
        samples,
        width: WIDTH,
        height: HEIGHT,
        bit_depth,
        signed,
        use_mct,
        alpha,
    }
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
