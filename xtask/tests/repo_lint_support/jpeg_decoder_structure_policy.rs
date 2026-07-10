// SPDX-License-Identifier: MIT OR Apache-2.0

//! Structural ownership and size ratchets for the public JPEG decoder facade.

use std::fs;

use super::{assert_pattern_checks, read_source_files, repo_root, PatternCheck};

#[test]
fn jpeg_decoder_view_and_output_format_live_in_focused_modules() {
    let root = repo_root();
    let decoder = fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder.rs"))
        .expect("read JPEG decoder module");
    let view = fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/view.rs"))
        .expect("read JPEG decoder view module");
    let output_format =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/output_format.rs"))
            .expect("read JPEG decoder output-format module");
    let extended12_root =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/extended12.rs"))
            .expect("read JPEG decoder extended12 module");
    let extended12 = read_source_files(
        root,
        &[
            "crates/j2k-jpeg/src/decoder/extended12.rs",
            "crates/j2k-jpeg/src/decoder/extended12/planes.rs",
            "crates/j2k-jpeg/src/decoder/extended12/progressive.rs",
            "crates/j2k-jpeg/src/decoder/extended12/progressive/color444.rs",
            "crates/j2k-jpeg/src/decoder/extended12/progressive/four_component.rs",
            "crates/j2k-jpeg/src/decoder/extended12/progressive/subsampled.rs",
            "crates/j2k-jpeg/src/decoder/extended12/rgba.rs",
            "crates/j2k-jpeg/src/decoder/extended12/sampling.rs",
            "crates/j2k-jpeg/src/decoder/extended12/sequential.rs",
            "crates/j2k-jpeg/src/decoder/extended12/sequential/color444.rs",
            "crates/j2k-jpeg/src/decoder/extended12/sequential/four_component.rs",
            "crates/j2k-jpeg/src/decoder/extended12/sequential/subsampled.rs",
            "crates/j2k-jpeg/src/decoder/extended12/state.rs",
            "crates/j2k-jpeg/src/decoder/extended12/upsample.rs",
            "crates/j2k-jpeg/src/decoder/extended12/writers.rs",
        ],
    );
    let lossless = fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/lossless_helpers.rs"))
        .expect("read JPEG decoder lossless helper module");
    let lossless_region =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/lossless_region.rs"))
            .expect("read JPEG decoder lossless region module");
    let color_convert =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/color_convert.rs"))
            .expect("read JPEG decoder color-convert module");
    let core_traits = fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/core_traits.rs"))
        .expect("read JPEG decoder core-traits module");
    let scratch = fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/scratch.rs"))
        .expect("read JPEG decoder scratch module");
    let sink_writer = fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/sink_writer.rs"))
        .expect("read JPEG decoder sink-writer module");
    let plan = fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/plan.rs"))
        .expect("read JPEG decoder plan module");
    let routing = fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/routing.rs"))
        .expect("read JPEG decoder routing module");
    let rows = fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/rows.rs"))
        .expect("read JPEG decoder rows module");
    let tile = fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/tile.rs"))
        .expect("read JPEG decoder tile module");
    let sequential = fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/sequential.rs"))
        .expect("read JPEG decoder sequential module");
    let lossless_render =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/lossless_render.rs"))
            .expect("read JPEG decoder lossless-render module");
    let bench_support = fs::read_to_string(root.join("crates/j2k-jpeg/src/bench_support.rs"))
        .expect("read JPEG bench support module");

    for (path, source) in [
        ("decoder.rs", decoder.as_str()),
        ("decoder/plan.rs", plan.as_str()),
        ("decoder/routing.rs", routing.as_str()),
        ("decoder/rows.rs", rows.as_str()),
        ("decoder/tile.rs", tile.as_str()),
        ("decoder/sequential.rs", sequential.as_str()),
        ("decoder/lossless_render.rs", lossless_render.as_str()),
    ] {
        assert!(
            source.lines().count() < 800,
            "crates/j2k-jpeg/src/{path} must stay below the focused-module line-count ratchet"
        );
        assert!(
            !source.contains("use super::*"),
            "crates/j2k-jpeg/src/{path} must keep explicit module imports"
        );
        if path != "decoder.rs" {
            assert!(
                !source.contains("include!("),
                "crates/j2k-jpeg/src/{path} must be a real Rust module"
            );
        }
    }
    assert!(
        extended12_root.lines().count() < 40,
        "decoder/extended12.rs must remain a focused module shell"
    );
    assert!(
        routing.contains("fn decode_lossless_output_format_region_scaled(")
            && routing.contains("self.lossless_plan.as_ref()?;")
            && rows.matches("if self.lossless_plan.is_some()").count() == 1,
        "JPEG routing must use one shared lossless output-format dispatch helper"
    );
    assert!(
        lossless_render.contains("fn decode_lossless_color8_output_into(")
            && lossless_render.contains("fn decode_lossless_color16_output_into(")
            && lossless_render
                .matches("match lossless_color_sampling(&self.info)")
                .count()
                == 2,
        "lossless rendering must keep RGB/YCbCr sampling dispatch shared by bit depth"
    );
    assert!(
        lossless_region.contains("pub(super) enum LosslessRgbRegionFallback")
            && lossless_region.contains("YCbCr8")
            && lossless_region.contains("Rgb8")
            && lossless_region.contains("YCbCr16")
            && lossless_region.contains("Rgb16")
            && lossless_region.contains("decode_rgb_region_scaled_into(")
            && lossless_region.contains("decode_rgba_region_scaled_into(")
            && !decoder.contains("enum LosslessRgbRegionFallback")
            && !decoder.contains("fn decode_lossless_rgb_region_scaled_into(")
            && !decoder.contains("fn decode_lossless_rgba8_region_into(")
            && !decoder.contains("fn decode_lossless_rgba16_region_scaled_into("),
        "decoder.rs must keep lossless region fallback routing on the focused helper module"
    );

    assert_pattern_checks(&[
        PatternCheck::new("decoder.rs view module shell", &decoder)
            .required(&["mod view;", "pub use self::view::JpegView;"])
            .forbidden(&[
                "pub struct JpegView",
                "impl<'a> JpegView<'a>",
                "parse_header(input)?",
            ]),
        PatternCheck::new("decoder/view.rs parsed-view API", &view).required(&[
            "pub struct JpegView",
            "impl<'a> JpegView<'a>",
            "pub fn parse(",
            "pub fn parse_with_options(",
            "pub fn passthrough_candidate(",
            "pub fn restart_index(",
        ]),
        PatternCheck::new("decoder.rs output-format module shell", &decoder).required(&[
            "mod output_format;",
            "output_format_from_parts",
            "checked_output_geometry",
        ]),
    ]);
    let output_format_patterns = [
        "pub(super) fn output_format_profile_name",
        "pub(super) fn downscale_profile_name",
        "pub(super) fn jpeg_downscale",
        "pub(super) fn output_format_from_parts",
        "pub(super) fn allocate_output_buffer",
        "pub(super) fn scaled_dimensions",
        "pub(super) fn scaled_rect_covering",
        "pub(super) fn checked_output_geometry",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("decoder.rs output-format helper exclusion", &decoder)
            .forbidden(&output_format_patterns),
        PatternCheck::new("decoder/output_format.rs helpers", &output_format)
            .required(&output_format_patterns),
        PatternCheck::new("decoder.rs focused helper module wiring", &decoder).required(&[
            "mod extended12;",
            "mod lossless_helpers;",
            "mod lossless_region;",
            "mod color_convert;",
            "mod core_traits;",
            "mod scratch;",
            "mod sink_writer;",
            "mod plan;",
            "mod routing;",
            "mod rows;",
            "mod sequential;",
            "mod tile;",
            "mod lossless_render;",
        ]),
        PatternCheck::new("decoder.rs focused ownership exclusions", &decoder).forbidden(&[
            "fn build_prepared_plan(",
            "pub fn decode_into(",
            "pub fn decode_rows<",
            "pub fn decode_tile_into(",
            "fn decode_with_writer<",
            "fn decode_lossless_gray8_into(",
        ]),
        PatternCheck::new("decoder/plan.rs plan ownership", &plan).required(&[
            "pub(super) fn build_prepared_plan(",
            "pub(super) fn build_lossless_plan(",
            "pub(super) fn build_progressive_plan(",
            "fn build_decode_plan(",
        ]),
        PatternCheck::new("decoder/routing.rs output ownership", &routing).required(&[
            "pub fn decode_into(",
            "pub fn decode_request(",
            "pub(super) fn decode_region_into_output_format_with_scratch(",
            "emit_jpeg_profile_row(",
        ]),
        PatternCheck::new("decoder/rows.rs row ownership", &rows).required(&[
            "pub fn decode_rows<",
            "pub fn decode_component_rows_with_scratch<",
            "pub fn decode_region_component_rows_with_scratch<",
        ]),
        PatternCheck::new("decoder/tile.rs tile facade ownership", &tile).required(&[
            "pub fn decode_tile_into(",
            "pub fn decode_prepared_jpeg_tiles_rgb8(",
            "pub fn decode_tiles_region_scaled_into(",
        ]),
        PatternCheck::new("decoder/sequential.rs writer ownership", &sequential).required(&[
            "pub(super) fn decode_scratch_bytes(",
            "pub(super) fn decode_with_writer<",
            "pub(super) fn decode_rgb_with_writer<",
        ]),
        PatternCheck::new(
            "decoder/lossless_render.rs render ownership",
            &lossless_render,
        )
        .required(&[
            "pub(super) fn decode_lossless_gray8_into(",
            "pub(super) fn decode_lossless_color_sampled_into<",
            "pub(super) fn decode_lossless_gray16_into(",
        ]),
    ]);
    let extended12_patterns = [
        "pub(super) enum Extended12Output",
        "pub(super) struct Extended12WriteRegion",
        "pub(super) fn decode_extended12_color_planes",
        "pub(super) fn render_progressive12_color_planes",
        "trait UpsampleSample",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("decoder/extended12.rs helpers", &extended12)
            .required(&extended12_patterns),
        PatternCheck::new("decoder.rs extended12 helper exclusion", &decoder)
            .forbidden(&extended12_patterns),
    ]);
    let extended12_renderer_patterns = [
        "fn decode_12bit_rgba16_region_scaled_into",
        "fn decode_progressive12_gray16_region_scaled_into",
        "fn decode_progressive12_rgb16_region_scaled_into",
        "fn decode_progressive12_region_into",
        "fn decode_progressive12_color444_region_into",
        "fn decode_progressive12_color_subsampled_region_into",
        "fn decode_progressive12_four_component_region_into",
        "fn decode_extended12_gray16_into",
        "fn decode_extended12_rgb16_region_scaled_into",
        "fn decode_extended12_region_into",
        "fn decode_extended12_color444_region_into",
        "fn decode_extended12_four_component_subsampled_region_into",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("decoder/extended12.rs 12-bit renderers", &extended12)
            .required(&extended12_renderer_patterns),
        PatternCheck::new("decoder.rs 12-bit renderer exclusion", &decoder)
            .forbidden(&extended12_renderer_patterns),
    ]);
    let lossless_patterns = [
        "pub(super) fn restart_index_for_stream",
        "pub(super) fn consume_lossless_restart",
        "pub(super) struct LosslessRestartTracker",
        "pub(super) struct Extended12RestartTracker",
        "pub(super) fn validate_lossless_color_plan",
        "pub(super) fn decode_lossless_plane_sample",
        "pub(super) fn decode_lossless_color_sample<P, T>",
        "pub(super) struct LosslessColorIntoSample",
        "pub(super) struct LosslessColorRowSample",
        "pub(super) fn decode_lossless_sampled_color_mcu<P>",
        "pub(super) struct LosslessSampledColorPlanesMut",
        "pub(super) struct LosslessSampledMcu",
        "pub(super) fn write_lossless_color16_sampled_output",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("decoder/lossless_helpers.rs helpers", &lossless)
            .required(&lossless_patterns),
        PatternCheck::new("decoder.rs lossless helper exclusion", &decoder)
            .forbidden(&lossless_patterns),
    ]);
    assert!(
        !decoder.contains("Extended12RestartTracker::new(")
            && extended12
                .contains("Extended12RestartTracker::new(self.plan.restart_interval, total_mcus)")
            && extended12
                .contains("Extended12RestartTracker::new(plan.restart_interval, total_mcus)")
            && !decoder.contains("consume_extended12_restart(")
            && !extended12.contains("consume_extended12_restart("),
        "extended-12 restart cadence must be centralized through Extended12RestartTracker"
    );
    assert!(
        lossless_render
            .matches("validate_lossless_color_plan::<P>")
            .count()
            == 3,
        "lossless render paths must share validation through decoder/lossless_helpers.rs"
    );
    let color_convert_patterns = [
        "pub(super) fn merged_warnings",
        "pub(super) fn convert_ycbcr8_to_rgb8_in_place",
        "pub(super) fn copy_rgb16_scaled_rect",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("decoder/color_convert.rs helpers", &color_convert)
            .required(&color_convert_patterns),
        PatternCheck::new("decoder.rs color-convert helper exclusion", &decoder)
            .forbidden(&color_convert_patterns),
    ]);
    let scratch_patterns = [
        "pub(super) fn compute_decode_scratch_bytes",
        "pub(super) fn compute_lossless_scratch_bytes",
        "pub(super) fn compute_extended12_planes_scratch_bytes",
        "pub(super) fn checked_scratch_len",
        "pub(super) fn checked_usize_product",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("decoder/scratch.rs helpers", &scratch).required(&scratch_patterns),
        PatternCheck::new("decoder.rs scratch helper exclusion", &decoder)
            .forbidden(&scratch_patterns),
    ]);
    let core_trait_patterns = [
        "impl ImageCodec for Decoder<'_>",
        "impl TileBatchDecode for JpegCodec",
        "pub(super) struct CroppedWriter",
        "impl<W: ComponentRowWriter + ?Sized> OutputWriter for &mut W",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("decoder/core_traits.rs trait adapters", &core_traits)
            .required(&core_trait_patterns)
            .forbidden(&["ComponentWriterAdapter"]),
        PatternCheck::new("decoder.rs core trait adapter exclusion", &decoder)
            .forbidden(&core_trait_patterns),
        PatternCheck::new("decoder.rs component writer adapter exclusion", &decoder)
            .forbidden(&["ComponentWriterAdapter"]),
        PatternCheck::new("decoder.rs sink writer re-export", &decoder)
            .required(&["pub(crate) use self::sink_writer::SinkWriter;"]),
    ]);
    let sink_writer_patterns = [
        "pub(crate) struct SinkWriter",
        "pub(crate) fn into_rows",
        "impl<S> InterleavedRgbWriter for SinkWriter<'_, S>",
        "impl<S> OutputWriter for SinkWriter<'_, S>",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("decoder/sink_writer.rs helpers", &sink_writer)
            .required(&sink_writer_patterns),
        PatternCheck::new("decoder.rs sink writer helper exclusion", &decoder)
            .forbidden(&sink_writer_patterns),
        PatternCheck::new("bench profile shared sink writer reuse", &bench_support)
            .required(&[
                "struct BlackBoxRowSink",
                "impl RowSink<u8> for BlackBoxRowSink",
                "SinkWriter::new(&mut sink, rows, dec.backend)",
            ])
            .forbidden(&[
                "struct BenchProfileSinkWriter",
                "impl InterleavedRgbWriter for BenchProfileSinkWriter",
                "impl OutputWriter for BenchProfileSinkWriter",
            ]),
    ]);
}

#[test]
fn jpeg_decoder_owned_outputs_use_decode_request() {
    let root = repo_root();
    let decoder = fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder.rs"))
        .expect("read j2k-jpeg decoder");
    let routing = fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/routing.rs"))
        .expect("read j2k-jpeg decoder routing");
    let decoder_api = format!("{decoder}\n{routing}");
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
            "fn decode_request_with_scratch(",
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
