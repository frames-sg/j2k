// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::{assert_pattern_checks, read_source_files, repo_root, PatternCheck};

#[test]
fn jpeg_metal_viewport_plane_rows_use_shared_target() {
    let root = repo_root();
    let viewport_cache =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/compute/viewport_cache.rs"))
            .expect("read j2k-jpeg-metal viewport cache");

    assert_pattern_checks(&[PatternCheck::new(
        "j2k-jpeg-metal viewport row writers",
        &viewport_cache,
    )
    .required(&[
        "struct PlaneRowTarget<'a>",
        "impl ComponentRowWriter for PlaneStage",
        "impl ComponentRowWriter for ViewportPlaneWriter<'_>",
        "impl PlaneRowTarget<'_>",
        "fn write_plane_row(&self, buffer: &Buffer, y: u32, src: &[u8]) -> Result<(), Error>",
        "fn checked_write_row_u8_at(",
        "checked_copy_bytes_to_buffer_at(",
    ])
    .normalized_required(&[
        "self.row_target() .write_gray_row(y, gray_row) .map_err(jpeg_plane_write_error)",
        "self.row_target() .write_ycbcr_row(y, y_row, chroma_blue_row, chroma_red_row) .map_err(jpeg_plane_write_error)",
        "self.row_target() .write_rgb_row(y, r_row, g_row, b_row) .map_err(jpeg_plane_write_error)",
    ])
    .forbidden(&["fn write_row_u8(", ".contents()"])]);
    assert!(
        viewport_cache
            .matches("fn row_target(&self) -> PlaneRowTarget<'_>")
            .count()
            == 2,
        "PlaneStage and ViewportPlaneWriter must both delegate through PlaneRowTarget"
    );
}

#[test]
fn jpeg_metal_single_decode_uses_request_api() {
    let root = repo_root();
    let lib = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/lib.rs"))
        .expect("read j2k-jpeg-metal lib");
    let codec_batch = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/codec_batch.rs"))
        .expect("read j2k-jpeg-metal codec batch module");
    let decode_request =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/decode_request.rs"))
            .expect("read j2k-jpeg-metal decode request module");
    let decoder = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/decoder.rs"))
        .expect("read j2k-jpeg-metal decoder module");
    let tile_batch = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/tile_batch.rs"))
        .expect("read j2k-jpeg-metal tile batch module");
    let source = format!("{lib}\n{codec_batch}\n{decode_request}\n{decoder}\n{tile_batch}");

    assert_pattern_checks(&[
        PatternCheck::new("j2k-jpeg-metal request API routing", &source)
            .required(&[
                "pub enum MetalDecodeOp",
                "pub struct MetalDecodeRequest",
                "pub fn decode_request_to_device(",
                "pub fn push_tile_request(",
                "pub fn push_shared_tile_request(",
                "pub fn submit_tile_request_to_device(",
                "Self::submit_tile_request_to_device(",
                "MetalDecodeRequest::region_scaled(fmt, roi, scale, backend)",
            ])
            .forbidden(&[
                "pub fn decode_region_scaled_to_device(",
                "pub fn push_tile(",
                "pub fn push_shared_tile(",
                "pub fn push_tile_region(",
                "pub fn push_shared_tile_region(",
                "pub fn push_tile_scaled(",
                "pub fn push_shared_tile_scaled(",
                "pub fn push_tile_region_scaled(",
                "pub fn push_shared_tile_region_scaled(",
            ]),
    ]);
    assert_pattern_checks(&[
        PatternCheck::new("j2k-jpeg-metal tile batch module shell", &lib)
            .required(&["mod tile_batch;", "pub use tile_batch::JpegTileBatch;"])
            .forbidden(&["pub struct JpegTileBatch"]),
        PatternCheck::new("j2k-jpeg-metal tile batch ownership", &tile_batch)
            .required(&["pub struct JpegTileBatch", "impl JpegTileBatch"]),
        PatternCheck::new("j2k-jpeg-metal decoder module shell", &lib)
            .required(&["mod decoder;", "pub use decoder::Decoder;"])
            .forbidden(&["pub struct Decoder<'a>"]),
        PatternCheck::new("j2k-jpeg-metal decoder ownership", &decoder)
            .required(&["pub struct Decoder<'a>", "impl<'a> Decoder<'a>"]),
        PatternCheck::new("j2k-jpeg-metal codec batch module shell", &lib)
            .required(&["mod codec_batch;", "pub use codec_batch::{"])
            .forbidden(&["impl Codec {", "pub enum Rgb8MetalBatchOp"]),
        PatternCheck::new("j2k-jpeg-metal codec batch ownership", &codec_batch)
            .required(&[
                "impl Codec",
                "pub enum Rgb8MetalBatchOp",
                "pub fn submit_tile_request_to_device(",
                "pub fn decode_rgb8_batch_into_buffer_with_session(",
            ])
            .forbidden(&["pub fn submit_tile_region_scaled_to_device("]),
        PatternCheck::new("j2k-jpeg-metal decode request module shell", &lib)
            .required(&[
                "mod decode_request;",
                "pub use decode_request::{MetalDecodeOp, MetalDecodeRequest};",
            ])
            .forbidden(&["pub enum MetalDecodeOp", "pub struct MetalDecodeRequest"]),
        PatternCheck::new("j2k-jpeg-metal decode request ownership", &decode_request)
            .required(&["pub enum MetalDecodeOp", "pub struct MetalDecodeRequest"]),
    ]);
}

#[test]
fn j2k_metal_decode_and_tile_batch_use_request_api() {
    let root = repo_root();
    let lib =
        fs::read_to_string(root.join("crates/j2k-metal/src/lib.rs")).expect("read j2k-metal lib");
    let decoder = read_source_files(
        root,
        &[
            "crates/j2k-metal/src/decoder.rs",
            "crates/j2k-metal/src/decoder/adapters.rs",
            "crates/j2k-metal/src/decoder/core.rs",
            "crates/j2k-metal/src/decoder/request.rs",
        ],
    );
    let tile_batch = fs::read_to_string(root.join("crates/j2k-metal/src/tile_batch.rs"))
        .expect("read j2k-metal tile batch");

    assert_pattern_checks(&[
        PatternCheck::new("j2k-metal decoder request API routing", &decoder)
            .required(&[
                "pub enum MetalDecodeOp",
                "pub struct MetalDecodeRequest",
                "pub fn decode_request_to_device(",
                "pub fn decode_request_to_device_with_report(",
                "pub fn decode_request_to_device_with_session(",
                "pub fn decode_request_to_host_surface(",
                "pub fn decode_request_to_cpu_staged_metal_surface_with_session(",
                "let request = MetalDecodeRequest::region_scaled(fmt, roi, scale, backend);",
                "request.op.batch_op()",
            ])
            .forbidden(&[
                "pub fn decode_to_device_with_report(",
                "pub fn decode_region_to_device_with_report(",
                "pub fn decode_scaled_to_device_with_report(",
                "pub fn decode_region_scaled_to_device_with_report(",
                "pub fn decode_to_device_with_session(",
                "pub fn decode_region_to_device_with_session(",
                "pub fn decode_scaled_to_device_with_session(",
                "pub fn decode_region_scaled_to_device_with_session(",
                "pub fn decode_to_host_surface(",
                "pub fn decode_region_to_host_surface(",
                "pub fn decode_scaled_to_host_surface(",
                "pub fn decode_region_scaled_to_host_surface(",
                "pub fn decode_to_cpu_staged_metal_surface_with_session(",
                "pub fn decode_region_to_cpu_staged_metal_surface_with_session(",
                "pub fn decode_scaled_to_cpu_staged_metal_surface_with_session(",
                "pub fn decode_region_scaled_to_cpu_staged_metal_surface_with_session(",
            ]),
        PatternCheck::new("j2k-metal decode request type re-export", &lib)
            .required(&["MetalDecodeOp", "MetalDecodeRequest"]),
    ]);
    assert_pattern_checks(&[PatternCheck::new(
        "j2k-metal tile batch request API routing",
        &tile_batch,
    )
    .required(&[
        "pub fn push_tile_request(",
        "pub fn push_shared_tile_request(",
        "self.push_shared_tile_request(",
    ])
    .forbidden(&[
        "pub fn push_tile(",
        "pub fn push_shared_tile(",
        "pub fn push_tile_region(",
        "pub fn push_shared_tile_region(",
        "pub fn push_tile_scaled(",
        "pub fn push_shared_tile_scaled(",
        "pub fn push_tile_region_scaled(",
        "pub fn push_shared_tile_region_scaled(",
    ])]);
}
