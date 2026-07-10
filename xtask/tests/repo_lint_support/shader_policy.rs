// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::*;

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the J2K Metal shader subsystem split is one fail-closed source policy"
)]
fn metal_encode_bitstream_shader_is_split_by_subsystem() {
    let root = repo_root();
    let shader_source =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/shader_source.rs"))
            .expect("read j2k-metal shader source");
    let monolith = fs::read_to_string(root.join("crates/j2k-metal/src/encode_bitstream.metal"))
        .expect("read j2k-metal encode bitstream placeholder");
    let classic_placeholder =
        fs::read_to_string(root.join("crates/j2k-metal/src/encode_bitstream_classic_tier1.metal"))
            .expect("read j2k-metal classic tier1 placeholder");

    assert_pattern_checks(&[
        PatternCheck::new("j2k-metal split shader source", &shader_source).forbidden(&[
            "include_str!(\"../encode_bitstream.metal\")",
            "include_str!(\"../encode_bitstream_classic_tier1.metal\")",
        ]),
    ]);

    let chunks = [
        (
            "encode_bitstream_shared.metal",
            "J2K_ENCODE_STATUS_OK",
            "shared encode status constants",
        ),
        (
            "encode_bitstream_classic_core.metal",
            "j2k_encode_classic_code_block_impl",
            "classic tier-1 core implementation",
        ),
        (
            "encode_bitstream_classic_tokens.metal",
            "j2k_pack_classic_tier1_tokens_bypass_u16_32_impl",
            "classic tier-1 token packing",
        ),
        (
            "encode_bitstream_classic_symbol_plan.metal",
            "j2k_plan_classic_tier1_symbols_bypass_u16_32_impl",
            "classic tier-1 symbol planning",
        ),
        (
            "encode_bitstream_classic_kernels.metal",
            "kernel void j2k_encode_classic_code_blocks",
            "classic tier-1 kernel wrappers",
        ),
        (
            "encode_bitstream_ht.metal",
            "kernel void j2k_encode_ht_code_blocks",
            "HT tier-1 kernels",
        ),
        (
            "encode_bitstream_packetize.metal",
            "kernel void j2k_encode_packetization",
            "packetization kernels",
        ),
    ];

    let mut previous_idx = 0;
    for (file, required_symbol, description) in chunks {
        let include = format!("include_str!(\"../{file}\")");
        let idx = shader_source
            .find(&include)
            .unwrap_or_else(|| panic!("j2k-metal shader source must include `{file}`"));
        assert!(
            idx >= previous_idx,
            "j2k-metal shader chunk `{file}` must be included in source order"
        );
        previous_idx = idx;

        let source = fs::read_to_string(root.join("crates/j2k-metal/src").join(file))
            .unwrap_or_else(|_| panic!("read j2k-metal shader chunk `{file}`"));
        assert!(
            source.lines().count() < 3000,
            "j2k-metal shader chunk `{file}` must stay below 3000 lines"
        );
        let check_name = format!("j2k-metal shader chunk {description}");
        assert_pattern_checks(&[
            PatternCheck::new(&check_name, &source).required(&[required_symbol])
        ]);
    }

    let classic_kernels = fs::read_to_string(
        root.join("crates/j2k-metal/src/encode_bitstream_classic_kernels.metal"),
    )
    .expect("read j2k-metal classic tier1 kernel chunk");
    assert_pattern_checks(&[PatternCheck::new(
        "classic tier-1 shared dispatch body",
        &classic_kernels,
    )
    .required(&["inline void j2k_encode_classic_code_blocks_dispatch"])]);
    assert_eq!(
        classic_kernels
            .matches("j2k_encode_classic_code_blocks_dispatch(")
            .count(),
        7,
        "classic tier-1 batch kernels must share one dispatch body plus six stable entry-point calls"
    );

    for (file, source) in [
        ("encode_bitstream.metal", &monolith),
        ("encode_bitstream_classic_tier1.metal", &classic_placeholder),
    ] {
        assert!(
            source.lines().count() < 50,
            "j2k-metal `{file}` must remain a small split-file pointer"
        );
        assert_pattern_checks(&[
            PatternCheck::new("j2k-metal split-file shader pointer", source)
                .forbidden(&["kernel void"]),
        ]);
    }
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the JPEG Metal shader subsystem split is one fail-closed source policy"
)]
fn jpeg_metal_shader_is_split_by_subsystem() {
    let root = repo_root();
    let compute = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/compute.rs"))
        .expect("read j2k-jpeg-metal compute");
    let monolith = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/shaders.metal"))
        .expect("read j2k-jpeg-metal shader placeholder");

    assert_pattern_checks(&[
        PatternCheck::new("j2k-jpeg-metal split shader source", &compute)
            .required(&["const SHADER_SOURCE: &str = concat!("])
            .forbidden(&["const SHADER_SOURCE: &str = include_str!(\"shaders.metal\")"]),
    ]);

    let chunks = [
        (
            "shaders_shared.metal",
            "jpeg_encode_baseline_entropy_one",
            "shared JPEG helpers",
            800usize,
        ),
        (
            "shaders_encode.metal",
            "kernel void jpeg_encode_baseline_entropy",
            "baseline entropy encode kernels",
            1_812usize,
        ),
        (
            "shaders_decode_helpers.metal",
            "inline bool jpeg_decode_idct_deposit_region_block_or_skip(",
            "shared decode/deposit helpers",
            174usize,
        ),
        (
            "shaders_pack_444.metal",
            "kernel void jpeg_pack_444_rgb_batch",
            "4:4:4 pack kernels",
            130usize,
        ),
        (
            "shaders_decode_fast420.metal",
            "kernel void jpeg_decode_fast420",
            "fast 4:2:0 decode kernels",
            966usize,
        ),
        (
            "shaders_decode_fast422_regions.metal",
            "kernel void jpeg_decode_fast422",
            "fast 4:2:2 and region-scaled decode kernels",
            935usize,
        ),
        (
            "shaders_decode_fast444.metal",
            "kernel void jpeg_decode_fast444",
            "fast 4:4:4 decode kernels",
            405usize,
        ),
        (
            "shaders_pack_subsampled.metal",
            "kernel void jpeg_pack_420",
            "subsampled pack kernels",
            815usize,
        ),
    ];

    let mut previous_idx = 0;
    for (file, required_symbol, description, max_lines) in chunks {
        let include = format!("include_str!(\"{file}\")");
        let idx = compute
            .find(&include)
            .unwrap_or_else(|| panic!("j2k-jpeg-metal compute.rs must include `{file}`"));
        assert!(
            idx >= previous_idx,
            "j2k-jpeg-metal shader chunk `{file}` must be included in source order"
        );
        previous_idx = idx;

        let source = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src").join(file))
            .unwrap_or_else(|_| panic!("read j2k-jpeg-metal shader chunk `{file}`"));
        assert!(
            source.lines().count() < max_lines,
            "j2k-jpeg-metal shader chunk `{file}` must stay below {max_lines} lines"
        );
        let check_name = format!("j2k-jpeg-metal shader chunk {description}");
        assert_pattern_checks(&[
            PatternCheck::new(&check_name, &source).required(&[required_symbol])
        ]);
    }

    assert!(
        monolith.lines().count() < 50,
        "j2k-jpeg-metal shaders.metal must remain a small split-file pointer"
    );
    assert_pattern_checks(&[PatternCheck::new(
        "j2k-jpeg-metal split-file shader pointer",
        &monolith,
    )
    .forbidden(&["kernel void"])]);

    let shared = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/shaders_shared.metal"))
        .expect("read j2k-jpeg-metal shared shader helpers");
    let encode = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/shaders_encode.metal"))
        .expect("read j2k-jpeg-metal encode shader helpers");
    let decode_helpers =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/shaders_decode_helpers.metal"))
            .expect("read j2k-jpeg-metal decode shader helpers");
    assert_pattern_checks(&[
        PatternCheck::new("j2k-jpeg-metal shared decode status helper", &shared).required(&[
            "inline void init_decode_status(device JpegDecodeStatus *status)",
            "status->code = FAST420_STATUS_OK;",
        ]),
        PatternCheck::new("j2k-jpeg-metal shared IDCT branch helper", &encode).required(&[
            "inline void idct_block(",
            "idct_islow_dc_only(coeffs[0], pixels);",
            "idct_islow(coeffs, pixels);",
            "inline bool decode_idct_deposit_block(",
            "deposit_block(plane, stride, width, height, x, y, pixels);",
        ]),
        PatternCheck::new("j2k-jpeg-metal shared entropy setup macros", &encode).required(&[
            "#define JPEG_ENTROPY_THREAD_VARS()",
            "#define JPEG_CONFIGURE_ENTROPY_THREAD(",
            "#define JPEG_BATCH_ENTROPY_THREAD_VARS()",
            "#define JPEG_CONFIGURE_BATCH_ENTROPY_THREAD(",
        ]),
        PatternCheck::new("j2k-jpeg-metal shared subsampled interpolation helpers", &encode)
            .required(&[
                "inline uint h2v2_weighted_sample_sum(uchar primary, uchar adjacent)",
                "inline uchar h2v2_boundary_left_from_sums(uint left_sum, uint right_sum)",
                "inline uchar h2v2_boundary_right_from_sums(uint left_sum, uint right_sum)",
            ]),
        PatternCheck::new("j2k-jpeg-metal shared decode/deposit helpers", &decode_helpers)
            .required(&[
                "inline bool jpeg_decode_idct_deposit_region_block_or_skip(",
                "inline bool jpeg_decode_deposit_scaled_region_block_or_skip(",
                "inline bool jpeg_decode_deposit_scaled_block(",
                "inline void jpeg_decode_clear_meta_quad(",
                "inline uint jpeg_clamped_extent(uint origin, uint span, uint limit)",
                "inline uchar h2v1_boundary_left_from_samples(uchar left, uchar right)",
                "inline uchar h2v1_boundary_right_from_samples(uchar left, uchar right)",
                "inline void jpeg_write_ycbcr_rgba(",
                "inline void jpeg_write_h2v2_boundary_pair(",
                "inline bool jpeg_skip_h2v2_boundary_repair_row(",
                "deposit_block_region(plane, stride, width, height, origin_x, origin_y, block_x, block_y, pixels);",
            ]),
    ]);
    assert_eq!(
        encode
            .matches("return uchar((3u * curr + prev + 1u) >> 2);")
            .count(),
        3,
        "all scalar/thread-local 4:2:2 even samples must retain libjpeg ordered +1 rounding"
    );
    assert_eq!(
        encode
            .matches("return uchar((3u * curr + next + 2u) >> 2);")
            .count(),
        3,
        "all scalar/thread-local 4:2:2 odd samples must retain libjpeg ordered +2 rounding"
    );
    assert_pattern_checks(&[
        PatternCheck::new("j2k-jpeg-metal paired 4:2:2 ordered rounding", &encode)
            .required(&[
                "left = uchar((3u * curr + prev + 1u) >> 2);",
                "right = uchar((3u * curr + next + 2u) >> 2);",
            ])
            .forbidden(&["3u * curr + prev + 2u) >> 2"]),
        PatternCheck::new(
            "j2k-jpeg-metal boundary 4:2:2 ordered rounding",
            &decode_helpers,
        )
        .required(&[
            "return uchar((3u * uint(left) + uint(right) + 2u) >> 2);",
            "return uchar((3u * uint(right) + uint(left) + 1u) >> 2);",
        ])
        .forbidden(&["3u * uint(right) + uint(left) + 2u) >> 2"]),
    ]);
    for file in [
        "shaders_decode_fast420.metal",
        "shaders_decode_fast422_regions.metal",
        "shaders_decode_fast444.metal",
    ] {
        let source = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src").join(file))
            .unwrap_or_else(|_| panic!("read j2k-jpeg-metal shader chunk `{file}`"));
        assert_pattern_checks(&[PatternCheck::new(file, &source)
            .required(&[
                "init_decode_status(thread_status);",
                "idct_block(coeffs, dc_only,",
                "decode_idct_deposit_block(",
                "JPEG_ENTROPY_THREAD_VARS();",
            ])
            .forbidden(&[
                "thread_status->code = FAST420_STATUS_OK;",
                "idct_islow_dc_only(coeffs[0],",
                "if (!configure_entropy_thread(",
                "rgba_float_ycbcr(",
            ])]);
    }
    for file in [
        "shaders_decode_fast420.metal",
        "shaders_decode_fast422_regions.metal",
    ] {
        let source = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src").join(file))
            .unwrap_or_else(|_| panic!("read j2k-jpeg-metal shader chunk `{file}`"));
        assert_pattern_checks(&[PatternCheck::new(file, &source)
            .required(&["JPEG_CONFIGURE_BATCH_ENTROPY_THREAD("])
            .forbidden(&[
                "const uint checkpoint_base = tile_index * params.segment_count;",
                "min(16u, params.width - min(",
                "min(16u, params.height - min(",
                "min(8u, params.height - min(",
                "min(8u, params.chroma_width - min(",
                "min(8u, params.chroma_height - min(",
            ])]);
    }
    for file in [
        "shaders_decode_fast420.metal",
        "shaders_decode_fast422_regions.metal",
        "shaders_decode_fast444.metal",
    ] {
        let source = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src").join(file))
            .unwrap_or_else(|_| panic!("read j2k-jpeg-metal shader chunk `{file}`"));
        assert_pattern_checks(&[PatternCheck::new(file, &source)
            .required(&[
                "if (!configure_batch_entropy_thread(",
                "status_index,",
                "params.tile_index + 1u,",
            ])
            .forbidden(&[
                "const uint checkpoint_base = params.tile_index * params.segment_count;",
                "const JpegEntropyCheckpoint checkpoint = entropy_checkpoints[",
                "const uint entropy_base = entropy_offsets[params.tile_index];",
                "boundary_meta[boundary_meta_base] = 0u;",
                "vertical_meta[vertical_meta_base] = 0u;",
            ])]);
    }
    let fast444 =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/shaders_decode_fast444.metal"))
            .expect("read j2k-jpeg-metal fast444 shader chunk");
    assert_pattern_checks(&[PatternCheck::new(
        "j2k-jpeg-metal fast444 region decode helpers",
        &fast444,
    )
    .required(&[
        "jpeg_decode_idct_deposit_region_block_or_skip(",
        "jpeg_decode_deposit_scaled_region_block_or_skip(",
        "jpeg_decode_deposit_scaled_block(",
    ])
    .forbidden(&[
        "inline bool fast444_decode_idct_deposit_region_block_or_skip(",
        "inline bool fast444_decode_deposit_scaled_region_block_or_skip(",
        "const bool intersects = block_intersects_rect(",
    ])]);
    let fast422 = fs::read_to_string(
        root.join("crates/j2k-jpeg-metal/src/shaders_decode_fast422_regions.metal"),
    )
    .expect("read j2k-jpeg-metal fast422 shader chunk");
    assert_pattern_checks(&[PatternCheck::new(
        "j2k-jpeg-metal fast422 region decode helpers",
        &fast422,
    )
    .required(&[
        "jpeg_decode_idct_deposit_region_block_or_skip(",
        "jpeg_decode_deposit_scaled_region_block_or_skip(",
        "jpeg_decode_deposit_scaled_block(",
        "h2v1_boundary_left_from_samples(",
        "h2v1_boundary_right_from_samples(",
    ])
    .forbidden(&[
        "inline bool subsampled_decode_idct_deposit_region_block_or_skip(",
        "inline bool subsampled_decode_deposit_scaled_region_block_or_skip(",
        "inline bool subsampled_decode_deposit_scaled_block(",
        "const bool mcu_intersects = block_intersects_rect(",
        "const bool y0_intersects = block_intersects_rect(",
        "3u * uint(left_cb) + uint(right_cb) + 2u",
        "3u * uint(right_cb) + uint(left_cb) + 2u",
        "3u * uint(prev_cb_row[7]) + uint(cb_row[0]) + 2u",
    ])]);
    let fast420 =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/shaders_decode_fast420.metal"))
            .expect("read j2k-jpeg-metal fast420 shader chunk");
    assert_pattern_checks(&[PatternCheck::new(
        "j2k-jpeg-metal fast420 h2v2 weighted-sum helpers",
        &fast420,
    )
    .required(&[
        "h2v2_weighted_sample_sum(",
        "jpeg_skip_h2v2_boundary_repair_row(",
    ])
    .forbidden(&[
        "h2v2_boundary_left_from_sums(",
        "h2v2_boundary_right_from_sums(",
        "3u * uint(prev_cb_pixels[local_chroma_y",
        "3u * uint(prev_cr_pixels[local_chroma_y",
        "3u * uint(cb_pixels[local_chroma_y",
        "3u * uint(cr_pixels[local_chroma_y",
        "3u * uint(boundary_samples[previous_sample_base",
        "3u * uint(boundary_samples[sample_base",
        "params.mcu_rows > 1u && has_top_mcu && by == 0u",
        "params.mcu_rows > 1u && has_bottom_mcu && by + 1u == copy_height",
        "params.mcu_rows > 1u && has_top_row && by == 0u",
        "params.mcu_rows > 1u && has_bottom_row && by + 1u == copy_height",
    ])]);
}
