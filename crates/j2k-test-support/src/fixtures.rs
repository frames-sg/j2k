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
