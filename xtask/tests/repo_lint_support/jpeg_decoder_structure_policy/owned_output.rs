// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::super::{assert_pattern_checks, repo_root, PatternCheck};

#[test]
fn jpeg_decoder_owned_outputs_use_decode_request() {
    let root = repo_root();
    let decoder = fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder.rs"))
        .expect("read j2k-jpeg decoder");
    let routing = fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/routing.rs"))
        .expect("read j2k-jpeg decoder routing");
    let owned_output =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/routing/owned_output.rs"))
            .expect("read j2k-jpeg owned-output routing");
    let decoder_api = format!("{decoder}\n{routing}\n{owned_output}");
    let lib =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/lib.rs")).expect("read j2k-jpeg lib");

    assert_pattern_checks(&[
        PatternCheck::new("j2k-jpeg owned-output request API", &decoder_api).required(&[
            "pub struct DecodeRequest",
            "pub const fn full(fmt: PixelFormat) -> Self",
            "pub const fn scaled(fmt: PixelFormat, scale: Downscale) -> Self",
            "pub const fn region(fmt: PixelFormat, region: Rect) -> Self",
            "pub const fn region_scaled(fmt: PixelFormat, region: Rect, scale: Downscale) -> Self",
            "pub fn decode_request(",
            "pub fn decode_request_with_external_live(",
            "pub fn decode_request_with_scratch_and_external_live(",
        ]),
        PatternCheck::new("j2k-jpeg owned-output wrapper removal", &decoder_api).forbidden(&[
            "pub fn decode(&self, fmt: PixelFormat)",
            "pub fn decode_scaled(",
            "pub fn decode_with_scratch(",
            "pub fn decode_scaled_with_scratch(",
            "pub fn decode_region(",
            "pub fn decode_region_scaled(",
            "pub fn decode_region_with_scratch(",
            "pub fn decode_region_scaled_with_scratch(",
        ]),
        PatternCheck::new("j2k-jpeg DecodeRequest re-export", &lib).required(&[
            "DecodeOutcome, DecodeRequest",
            "DecodedTile, Decoder, JpegView",
        ]),
    ]);
}
