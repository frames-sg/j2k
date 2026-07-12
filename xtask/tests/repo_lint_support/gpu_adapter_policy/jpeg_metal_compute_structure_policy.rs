// SPDX-License-Identifier: MIT OR Apache-2.0

//! Structural inventory for the JPEG Metal compute and pack-dispatch module family.

use std::fs;

use super::{assert_pattern_checks, repo_root, rust_sources, PatternCheck};

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "this is one table-driven ownership audit of the complete JPEG Metal module family"
)]
fn jpeg_metal_compute_uses_real_focused_modules() {
    let root = repo_root();
    let source_root = root.join("crates/j2k-jpeg-metal/src");
    let compute =
        fs::read_to_string(source_root.join("compute.rs")).expect("read j2k-jpeg-metal compute");

    assert_pattern_checks(&[
        PatternCheck::new("j2k-jpeg-metal compute module shell", &compute)
            .required(&[
                "mod batch_entry;",
                "mod batch_full;",
                "mod batch_region;",
                "mod encode;",
                "mod fast_packets;",
                "mod pack_dispatch;",
                "mod single_decode;",
            ])
            .forbidden(&["include!(", "fn try_decode_fast_subsampled_"]),
    ]);
    assert!(
        compute.lines().count() < 800,
        "j2k-jpeg-metal compute.rs must stay below its real-module line-count ratchet"
    );

    for (relative, required_modules, forbidden_item) in [
        (
            "compute/fast_packets.rs",
            &["mod descriptors;", "mod params;", "mod pipelines;"][..],
            "trait FastSubsampledMetal",
        ),
        (
            "compute/pack_dispatch.rs",
            &[
                "mod conversion;",
                "mod fast444;",
                "mod grouped_output;",
                "mod requests;",
                "mod subsampled;",
                "mod surface;",
                "mod texture;",
                "mod texture_dispatch;",
                "mod split_coeff_idct;",
            ][..],
            "fn encode_fast444_batch_item(",
        ),
        (
            "compute/single_decode.rs",
            &["mod fast444;", "mod routing;", "mod subsampled;"][..],
            "pub(crate) fn decode_to_surface(",
        ),
        (
            "compute/batch_full.rs",
            &[
                "mod fast444;",
                "mod rgb;",
                "mod texture;",
                "mod texture_grouped;",
            ][..],
            "fn finish_fast_subsampled_full_rgb_batch(",
        ),
        (
            "compute/batch_region.rs",
            &["mod common;", "mod repeated;", "mod rgb;", "mod texture;"][..],
            "fn try_decode_repeated_region_scaled_batch_to_surfaces(",
        ),
    ] {
        let path = source_root.join(relative);
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        assert!(
            source.lines().count() < 50,
            "{relative} must remain a focused real-module shell"
        );
        let base_forbidden = [forbidden_item, "include!("];
        let pack_dispatch_forbidden = [forbidden_item, "include!(", "mod common;"];
        let forbidden = if relative == "compute/pack_dispatch.rs" {
            pack_dispatch_forbidden.as_slice()
        } else {
            base_forbidden.as_slice()
        };
        assert_pattern_checks(&[PatternCheck::new(relative, &source)
            .required(required_modules)
            .forbidden(forbidden)]);
    }

    for obsolete in [
        "compute/fast_packets_impl.rs",
        "compute/pack_dispatch/common.rs",
        "compute/pack_dispatch_impl.rs",
        "compute/single_decode_impl.rs",
        "compute/batch_decode_full.rs",
        "compute/batch_decode_region.rs",
        "compute/batch_decode_entry.rs",
        "compute/batch_decode_impl.rs",
    ] {
        assert!(
            !source_root.join(obsolete).exists(),
            "obsolete JPEG Metal production include fragment must not exist: {obsolete}"
        );
    }

    for (relative, max_lines, required_symbol) in [
        (
            "compute/encode.rs",
            330,
            "pub(crate) fn encode_jpeg_baseline_entropy_with_session",
        ),
        (
            "compute/batch_entry.rs",
            450,
            "pub(crate) fn decode_full_batch_to_surfaces",
        ),
        (
            "compute/fast_packets/descriptors.rs",
            400,
            "trait FastSubsampledMetal",
        ),
        (
            "compute/fast_packets/params.rs",
            500,
            "fn restart_work_for_mcu_range",
        ),
        (
            "compute/fast_packets/pipelines.rs",
            700,
            "impl FastSubsampledMetal for JpegFast420PacketV1",
        ),
        ("compute/pack_dispatch/conversion.rs", 25, "fn checked_u32"),
        (
            "compute/pack_dispatch/requests.rs",
            75,
            "struct FastSubsampledOpBatchItemRequest",
        ),
        (
            "compute/pack_dispatch/grouped_output.rs",
            175,
            "fn copy_grouped_surfaces_to_output",
        ),
        (
            "compute/pack_dispatch/surface.rs",
            150,
            "fn encode_jpeg_pack_to_surface_in_command_buffer",
        ),
        (
            "compute/pack_dispatch/texture.rs",
            250,
            "fn validate_rgba_texture_batch_output",
        ),
        (
            "compute/pack_dispatch/texture_dispatch.rs",
            100,
            "fn dispatch_rgba_texture_pack",
        ),
        (
            "compute/pack_dispatch/split_coeff_idct.rs",
            150,
            "struct SplitCoeffIdctPasses",
        ),
        (
            "compute/pack_dispatch/fast444.rs",
            500,
            "fn encode_fast444_batch_item",
        ),
        (
            "compute/pack_dispatch/subsampled.rs",
            650,
            "fn encode_fast_subsampled_op_batch_item",
        ),
        (
            "compute/single_decode/fast444.rs",
            550,
            "fn try_decode_fast444_to_surface",
        ),
        (
            "compute/single_decode/routing.rs",
            350,
            "pub(crate) fn decode_to_surface",
        ),
        (
            "compute/single_decode/subsampled.rs",
            550,
            "fn try_decode_fast420_to_surface",
        ),
        (
            "compute/batch_full/rgb.rs",
            700,
            "fn finish_fast_subsampled_full_rgb_batch",
        ),
        (
            "compute/batch_full/rgb_grouped.rs",
            175,
            "fn merge_group_results",
        ),
        (
            "compute/batch_full/texture.rs",
            750,
            "fn decode_fast_subsampled_full_rgba_fused_texture_batch",
        ),
        (
            "compute/batch_full/texture/staged.rs",
            250,
            "fn decode_fast_subsampled_full_rgba_staged_texture_batch",
        ),
        (
            "compute/batch_full/texture_grouped.rs",
            100,
            "fn try_decode_grouped_fast_subsampled_full_rgba_batch_to_textures",
        ),
        (
            "compute/batch_region/common.rs",
            450,
            "fn subsampled_region_rgb_batch_shape",
        ),
        (
            "compute/batch_region/repeated.rs",
            100,
            "fn try_decode_repeated_region_scaled_batch_to_surfaces",
        ),
        (
            "compute/batch_region/rgb.rs",
            600,
            "fn try_decode_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output",
        ),
        (
            "compute/batch_region/texture/fast444.rs",
            550,
            "fn try_decode_fast444_region_scaled_rgba_batch_to_textures",
        ),
        (
            "compute/batch_region/texture/subsampled.rs",
            450,
            "fn try_decode_fast_subsampled_region_scaled_rgba_batch_to_textures",
        ),
    ] {
        let path = source_root.join(relative);
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        assert!(
            source.lines().count() < max_lines,
            "{relative} must stay below its focused-module line-count ratchet"
        );
        assert_pattern_checks(&[PatternCheck::new(relative, &source)
            .required(&[required_symbol])
            .forbidden(&["include!(", "use super::*;", concat!("#!", "[allow(")])]);
    }

    for path in rust_sources(&source_root.join("compute")) {
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        assert!(
            !source.contains("include!("),
            "production include fragments are forbidden in {}",
            path.display()
        );
    }
}
