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

/// `OpenJPEG` 2.5.4 irreversible 8x8 RGB codestream used for adapter parity tests.
///
/// The source pixels are the deterministic `patterned_rgb8` formula. `OpenJPEG`
/// encoded them with `opj_compress -I -r 8 -n 4`.
pub const OPENJPEG_IRREVERSIBLE_RGB8_8X8: &[u8] = &[
    0xff, 0x4f, 0xff, 0x51, 0x00, 0x2f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x08,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x08,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x07, 0x01, 0x01, 0x07, 0x01, 0x01,
    0x07, 0x01, 0x01, 0xff, 0x52, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x01, 0x01, 0x03, 0x04, 0x04, 0x00,
    0x00, 0xff, 0x5c, 0x00, 0x17, 0x42, 0x67, 0x38, 0x67, 0x50, 0x67, 0x50, 0x67, 0x68, 0x50, 0x05,
    0x50, 0x05, 0x50, 0x47, 0x57, 0xd3, 0x57, 0xd3, 0x57, 0x62, 0xff, 0x64, 0x00, 0x25, 0x00, 0x01,
    0x43, 0x72, 0x65, 0x61, 0x74, 0x65, 0x64, 0x20, 0x62, 0x79, 0x20, 0x4f, 0x70, 0x65, 0x6e, 0x4a,
    0x50, 0x45, 0x47, 0x20, 0x76, 0x65, 0x72, 0x73, 0x69, 0x6f, 0x6e, 0x20, 0x32, 0x2e, 0x35, 0x2e,
    0x34, 0xff, 0x90, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x28, 0x00, 0x01, 0xff, 0x93, 0xc7,
    0xea, 0x04, 0x06, 0xbf, 0x80, 0x80, 0xa0, 0xfb, 0xc0, 0x80, 0x01, 0x9f, 0xc1, 0xf7, 0x81, 0x00,
    0x04, 0x8f, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0xff, 0xd9,
];

#[cfg(feature = "j2k-native-fixtures")]
mod generated_htj2k;
mod jp2;
mod jpeg;
mod openjph;

#[cfg(feature = "j2k-native-fixtures")]
pub use generated_htj2k::{
    classic_j2k_gray8_fixture, generated_htj2k_rgba_fixture, htj2k_gray8_97_fixture,
    htj2k_gray8_fixture, htj2k_gray8_large_fixture, htj2k_rgb8_97_fixture, htj2k_rgb8_fixture,
    htj2k_rgb8_fixture_with_pixels, htj2k_rgb8_pattern_fixture, Htj2kRgbaAlpha, Htj2kRgbaFixture,
    Htj2kRgbaSampleProfile, Htj2kRgbaSamples,
};
pub use jp2::{
    minimal_j2k_codestream, minimal_jp2, rewrite_j2k_component_sampling, wrap_jp2_codestream,
    wrap_jp2_rgba_codestream,
};
pub use jpeg::{
    baseline_grayscale_jpeg, minimal_baseline_jpeg, minimal_baseline_jpeg_with_restart_interval,
    minimal_grayscale_jpeg_with_dimensions, restart_coded_grayscale_jpeg,
};
pub use openjph::{
    openhtj2k_refinement_fixture, openhtj2k_refinement_odd_fixture,
    openhtj2k_refinement_odd_pixels, openhtj2k_refinement_pixels, openhtj2k_sigprop_fixture,
    openhtj2k_sigprop_overlap_fixture, openhtj2k_sigprop_overlap_pixels,
    openhtj2k_sigprop_pixels_le, openjph_batch_fixtures, OpenJphBatchFixture,
};
