// SPDX-License-Identifier: MIT OR Apache-2.0

#[derive(Clone, Copy, Debug)]
pub struct OpenJphBatchFixture {
    /// Stable fixture label used in assertion diagnostics.
    pub name: &'static str,
    /// Raw Part 15 codestream or boxed JPH file bytes.
    pub encoded: &'static [u8],
    /// OpenJPH-decoded interleaved native sample bytes.
    pub oracle: &'static [u8],
    /// Decoded image width.
    pub width: u32,
    /// Decoded image height.
    pub height: u32,
    /// One for grayscale or three for RGB.
    pub components: usize,
    /// Uniform component precision.
    pub precision: u8,
    /// Whether every component uses a signed sample domain.
    pub signed: bool,
    /// Whether the codestream uses the reversible 5/3 transform.
    pub reversible: bool,
    /// Whether [`Self::encoded`] is a boxed JPH file.
    pub jph: bool,
}

macro_rules! openjph_fixture {
    ($name:literal, $encoded:literal, $oracle:literal, $components:literal, $precision:literal, $signed:literal, $reversible:literal, $jph:literal) => {
        OpenJphBatchFixture {
            name: $name,
            encoded: include_bytes!(concat!("../../fixtures/htj2k/openjph_batch/", $encoded)),
            oracle: include_bytes!(concat!("../../fixtures/htj2k/openjph_batch/", $oracle)),
            width: 19,
            height: 13,
            components: $components,
            precision: $precision,
            signed: $signed,
            reversible: $reversible,
            jph: $jph,
        }
    };
}

static OPENJPH_BATCH_FIXTURES: &[OpenJphBatchFixture] = &[
    openjph_fixture!(
        "openjph-gray-u8-53-raw",
        "gray_u8_53.j2c",
        "gray_u8_53.oracle.raw",
        1,
        8,
        false,
        true,
        false
    ),
    openjph_fixture!(
        "openjph-gray-u12-53-raw",
        "gray_u12_53.j2c",
        "gray_u12_53.oracle.raw",
        1,
        12,
        false,
        true,
        false
    ),
    openjph_fixture!(
        "openjph-gray-u12-53-jph",
        "gray_u12_53.jph",
        "gray_u12_53.oracle.raw",
        1,
        12,
        false,
        true,
        true
    ),
    openjph_fixture!(
        "openjph-gray-u16-53-raw",
        "gray_u16_53.j2c",
        "gray_u16_53.oracle.raw",
        1,
        16,
        false,
        true,
        false
    ),
    openjph_fixture!(
        "openjph-gray-s8-53-raw",
        "gray_s8_53.j2c",
        "gray_s8_53.oracle.raw",
        1,
        8,
        true,
        true,
        false
    ),
    openjph_fixture!(
        "openjph-gray-s12-53-raw",
        "gray_s12_53.j2c",
        "gray_s12_53.oracle.raw",
        1,
        12,
        true,
        true,
        false
    ),
    openjph_fixture!(
        "openjph-gray-s16-53-raw",
        "gray_s16_53.j2c",
        "gray_s16_53.oracle.raw",
        1,
        16,
        true,
        true,
        false
    ),
    openjph_fixture!(
        "openjph-rgb-u8-53-raw",
        "rgb_u8_53.j2c",
        "rgb_u8_53.oracle.raw",
        3,
        8,
        false,
        true,
        false
    ),
    openjph_fixture!(
        "openjph-rgb-u12-53-raw",
        "rgb_u12_53.j2c",
        "rgb_u12_53.oracle.raw",
        3,
        12,
        false,
        true,
        false
    ),
    openjph_fixture!(
        "openjph-rgb-u16-53-raw",
        "rgb_u16_53.j2c",
        "rgb_u16_53.oracle.raw",
        3,
        16,
        false,
        true,
        false
    ),
    openjph_fixture!(
        "openjph-rgb-s8-53-raw",
        "rgb_s8_53.j2c",
        "rgb_s8_53.oracle.raw",
        3,
        8,
        true,
        true,
        false
    ),
    openjph_fixture!(
        "openjph-rgb-s12-53-raw",
        "rgb_s12_53.j2c",
        "rgb_s12_53.oracle.raw",
        3,
        12,
        true,
        true,
        false
    ),
    openjph_fixture!(
        "openjph-rgb-s16-53-raw",
        "rgb_s16_53.j2c",
        "rgb_s16_53.oracle.raw",
        3,
        16,
        true,
        true,
        false
    ),
    openjph_fixture!(
        "openjph-rgb-s8-53-single-raw",
        "rgb_s8_53_single.j2c",
        "rgb_s8_53.oracle.raw",
        3,
        8,
        true,
        true,
        false
    ),
    openjph_fixture!(
        "openjph-rgb-s12-53-single-raw",
        "rgb_s12_53_single.j2c",
        "rgb_s12_53.oracle.raw",
        3,
        12,
        true,
        true,
        false
    ),
    openjph_fixture!(
        "openjph-rgb-s16-53-single-raw",
        "rgb_s16_53_single.j2c",
        "rgb_s16_53.oracle.raw",
        3,
        16,
        true,
        true,
        false
    ),
    openjph_fixture!(
        "openjph-gray-u12-97-raw",
        "gray_u12_97.j2c",
        "gray_u12_97.oracle.raw",
        1,
        12,
        false,
        false,
        false
    ),
    openjph_fixture!(
        "openjph-gray-u16-97-raw",
        "gray_u16_97.j2c",
        "gray_u16_97.oracle.raw",
        1,
        16,
        false,
        false,
        false
    ),
    openjph_fixture!(
        "openjph-gray-s12-97-raw",
        "gray_s12_97.j2c",
        "gray_s12_97.oracle.raw",
        1,
        12,
        true,
        false,
        false
    ),
    openjph_fixture!(
        "openjph-gray-s16-97-raw",
        "gray_s16_97.j2c",
        "gray_s16_97.oracle.raw",
        1,
        16,
        true,
        false,
        false
    ),
    openjph_fixture!(
        "openjph-rgb-u8-97-raw",
        "rgb_u8_97.j2c",
        "rgb_u8_97.oracle.raw",
        3,
        8,
        false,
        false,
        false
    ),
    openjph_fixture!(
        "openjph-rgb-u12-97-raw",
        "rgb_u12_97.j2c",
        "rgb_u12_97.oracle.raw",
        3,
        12,
        false,
        false,
        false
    ),
    openjph_fixture!(
        "openjph-rgb-s12-97-raw",
        "rgb_s12_97.j2c",
        "rgb_s12_97.oracle.raw",
        3,
        12,
        true,
        false,
        false
    ),
];

/// Return the independently encoded `OpenJPH` HTJ2K batch fixture matrix.
pub fn openjph_batch_fixtures() -> &'static [OpenJphBatchFixture] {
    OPENJPH_BATCH_FIXTURES
}

/// `OpenHTJ2K` refinement fixture with a compact output plane.
pub fn openhtj2k_refinement_fixture() -> &'static [u8] {
    include_bytes!("../../fixtures/htj2k/openhtj2k_ds0_ht_12_b11.j2k")
}

/// Expected grayscale pixels for [`openhtj2k_refinement_fixture`].
pub fn openhtj2k_refinement_pixels() -> &'static [u8] {
    include_bytes!("../../fixtures/htj2k/openhtj2k_ds0_ht_12_b11.gray")
}

/// `OpenHTJ2K` odd refinement fixture used by CUDA plan tests.
pub fn openhtj2k_refinement_odd_fixture() -> &'static [u8] {
    include_bytes!("../../fixtures/htj2k/openhtj2k_ds0_ht_09_b11.j2k")
}

/// Expected grayscale pixels for [`openhtj2k_refinement_odd_fixture`].
pub fn openhtj2k_refinement_odd_pixels() -> &'static [u8] {
    include_bytes!("../../fixtures/htj2k/openhtj2k_ds0_ht_09_b11.gray")
}

/// Independent `OpenHTJ2K` RGB12 conformance codestream containing exactly-two-pass jobs.
pub fn openhtj2k_sigprop_fixture() -> &'static [u8] {
    include_bytes!("../../fixtures/htj2k/openhtj2k_hifi_ht1_02.j2k")
}

/// OpenJPH-decoded little-endian RGB12 oracle for [`openhtj2k_sigprop_fixture`].
pub fn openhtj2k_sigprop_pixels_le() -> &'static [u8] {
    include_bytes!("../../fixtures/htj2k/openhtj2k_hifi_ht1_02.oracle.raw")
}

/// Independent `OpenHTJ2K` stream exercising overlapping SigProp/MagRef refinement bits.
pub fn openhtj2k_sigprop_overlap_fixture() -> &'static [u8] {
    include_bytes!("../../fixtures/htj2k/openhtj2k_sigprop_refinement_overlap.j2k")
}

/// OpenHTJ2K-decoded RGB8 oracle for [`openhtj2k_sigprop_overlap_fixture`].
pub fn openhtj2k_sigprop_overlap_pixels() -> &'static [u8] {
    const PPM_HEADER_LEN: usize = b"P6 512 64 255\n".len();
    &include_bytes!("../../fixtures/htj2k/openhtj2k_sigprop_refinement_overlap.openht.oracle.ppm")
        [PPM_HEADER_LEN..]
}
