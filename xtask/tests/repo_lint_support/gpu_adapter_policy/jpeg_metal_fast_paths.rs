// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::{assert_pattern_checks, read_source_files, repo_root, PatternCheck};

#[test]
fn fast444_region_scaled_batches_use_shared_region_scaled_metal_path() {
    let root = repo_root();
    let fast_packets = [
        "crates/j2k-jpeg-metal/src/compute/fast_packets/descriptors.rs",
        "crates/j2k-jpeg-metal/src/compute/fast_packets/pipelines.rs",
    ]
    .map(|path| fs::read_to_string(root.join(path)).expect("read JPEG Metal fast packet module"))
    .join("\n");
    let region_plan =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/compute/region_scaled_plan.rs"))
            .expect("read JPEG Metal region scaled plan");
    let batch_decode =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/compute/batch_region/rgb.rs"))
            .expect("read JPEG Metal region RGB batch decoder");

    assert_pattern_checks(&[
        PatternCheck::new("fast packet region-scaled Metal trait", &fast_packets).required(&[
            "trait FastRegionScaledMetal",
            "impl FastRegionScaledMetal for JpegFast444PacketV1",
            "fn chroma_width(width: u32) -> u32",
        ]),
        PatternCheck::new("region-scaled packet-family planning", &region_plan).required(&[
            "mode: PlaneMode",
            "plane_mode_to_u32(mode)",
            "P::chroma_width(source_window.w)",
        ]),
        PatternCheck::new("fast444 RGB region-scaled batch path", &batch_decode)
            .required(&[
                "try_decode_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output::<JpegFast444PacketV1>",
            ])
            .forbidden(&[
                "fn try_decode_fast444_region_scaled_rgb_batch_to_surfaces_with_output(",
                "fn try_decode_fast444_restart_region_scaled_rgb_batch_to_surfaces_with_output(",
                "fn try_decode_grouped_fast444_region_scaled_rgb_batch_to_surfaces_with_output(",
            ]),
    ]);
}

#[test]
fn fast444_full_batches_use_shared_fastsubsampled_metal_path() {
    let root = repo_root();
    let fast_packets = fs::read_to_string(
        root.join("crates/j2k-jpeg-metal/src/compute/fast_packets/pipelines.rs"),
    )
    .expect("read JPEG Metal fast packet pipelines");
    let batch_decode =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/compute/batch_full/fast444.rs"))
            .expect("read JPEG Metal fast444 full batch decoder");

    assert_pattern_checks(&[
        PatternCheck::new("fast444 shared FastSubsampledMetal impl", &fast_packets)
            .required(&["impl FastSubsampledMetal for JpegFast444PacketV1"]),
        PatternCheck::new(
            "fast444 full shared region-scaled batch path",
            &batch_decode,
        )
        .required(&[
            "fn fast444_full_region_scaled_requests(",
            "scale: j2k_core::Downscale::None",
            "try_decode_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output::<",
            "JpegFast444PacketV1",
            "try_decode_fast444_region_scaled_rgba_batch_to_textures(",
        ])
        .forbidden(&[
            "struct Fast444FullRgbSurfaceShape",
            "struct Fast444FullRgbaTextureShape",
            "fn fast444_full_packets(",
            "fn try_decode_grouped_fast444_full_rgb_batch_to_surfaces_with_output(",
            "fn try_decode_grouped_fast444_full_rgba_batch_to_textures(",
            "fn encode_fast444_full_rgba_texture_decode(",
            "fast444_full_entropy",
        ]),
    ]);
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the complete fast-420 ownership and single-scan-loop ledger is one cohesive policy audit"
)]
fn jpeg_fast420_profiled_decode_uses_shared_scan_loop() {
    let root = repo_root();
    let sequential = fs::read_to_string(root.join("crates/j2k-jpeg/src/entropy/sequential.rs"))
        .expect("read JPEG entropy sequential decoder");
    let fast420 = read_source_files(
        root,
        &[
            "crates/j2k-jpeg/src/entropy/sequential/fast420/mod.rs",
            "crates/j2k-jpeg/src/entropy/sequential/fast420/rows.rs",
        ],
    );
    let profile =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/entropy/sequential/profile.rs"))
            .expect("read JPEG entropy sequential profile module");
    let layout = fs::read_to_string(root.join("crates/j2k-jpeg/src/entropy/sequential/layout.rs"))
        .expect("read JPEG entropy sequential layout module");
    let restart =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/entropy/sequential/restart.rs"))
            .expect("read JPEG entropy sequential restart module");
    let deposit =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/entropy/sequential/deposit.rs"))
            .expect("read JPEG entropy sequential deposit module");
    let emit = read_source_files(
        root,
        &[
            "crates/j2k-jpeg/src/entropy/sequential/emit.rs",
            "crates/j2k-jpeg/src/entropy/sequential/emit/region420.rs",
            "crates/j2k-jpeg/src/entropy/sequential/emit/rgb.rs",
            "crates/j2k-jpeg/src/entropy/sequential/emit/upsample.rs",
        ],
    );
    let tests = fs::read_to_string(root.join("crates/j2k-jpeg/src/entropy/sequential/tests.rs"))
        .expect("read JPEG entropy sequential tests module");

    assert!(
        sequential.lines().count() < 2_500,
        "entropy/sequential.rs must stay below the post-helper-split line-count ratchet"
    );

    assert_pattern_checks(&[
        PatternCheck::new("JPEG fast420 profile module ownership", &profile)
            .required(&[
                "trait Fast420ScanProfiler",
                "struct NoopFast420ScanProfile",
                "impl Fast420ScanProfiler for BenchFast420Profile",
            ])
            .forbidden(&["let mcu_start = Instant::now();"]),
        PatternCheck::new("JPEG fast420 shared scan loop ownership", &sequential)
            .required(&[
                "mod profile;",
                "mod layout;",
                "mod restart;",
                "mod deposit;",
                "mod emit;",
                "mod fast420;",
            ])
            .forbidden(&[
                "fn decode_scan_fast_tile_rgb_impl",
                "struct NoopFast420ScanProfile",
                "struct Fast420RegionLayout",
                "struct McuSkipState",
                "pub(super) fn deposit_block",
                "struct StripeEmit",
            ]),
        PatternCheck::new("JPEG fast420 shared scan loop implementation", &fast420)
            .required(&[
                "fn decode_scan_fast_tile_rgb_impl",
                "decode_scan_fast_tile_rgb_impl(plan, backend, scan_bytes, pool, writer, &mut profile)",
                "decode_scan_fast_tile_rgb_impl(plan, backend, scan_bytes, pool, writer, profile)",
            ])
            .forbidden(&[
                "struct NoopFast420ScanProfile",
                "struct Fast420RegionLayout",
                "struct McuSkipState",
                "pub(super) fn deposit_block",
                "struct StripeEmit",
            ]),
        PatternCheck::new("JPEG fast420 layout helper ownership", &layout).required(&[
            "pub(crate) fn stripe_region_layout(",
            "pub(crate) fn fast_tile_region_first_decode_mcu(",
            "struct Fast420RegionLayout",
            "fn expanded_output_rect(",
        ]),
        PatternCheck::new("JPEG fast420 restart helper ownership", &restart).required(&[
            "fn reader_from_checkpoint",
            "fn restart_seek_for_mcu",
            "struct McuSkipState",
            "fn skip_to_mcu",
        ]),
        PatternCheck::new("JPEG fast420 deposit helper ownership", &deposit).required(&[
            "fn assert_stripe_deposit_capacity",
            "fn deposit_block(",
            "fn deposit_dc_block(",
            "fn idct_deposit_fast_tile_block",
        ]),
        PatternCheck::new("JPEG fast420 emit helper ownership", &emit).required(&[
            "fn emit_stripe_rgb_420_region",
            "fn emit_stripe_rgb",
            "fn component_row_triplet",
            "fn upsample_component_row_stripe",
        ]),
        PatternCheck::new("JPEG fast420 sequential helper regression tests", &tests)
            .required(&["fast_tile_profiled_rgb_matches_unprofiled_decode"]),
    ]);
    assert_eq!(
        fast420.matches("finish_scan(&mut br, true)").count(),
        1,
        "JPEG fast420 profiled/unprofiled scan paths must not duplicate the scan loop"
    );
}
