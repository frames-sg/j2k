// SPDX-License-Identifier: MIT OR Apache-2.0

//! Byte pins for the irreversible 9/7 JPEG-to-HTJ2K transcode at both
//! [`TranscodePrecision`] settings.
//!
//! `F64` keeps the binary64 arithmetic that was the CPU default before 0.14.
//! `F32` is the canonical binary32 path that the Metal and CUDA backends
//! reproduce; its digests change only when the canonical arithmetic in
//! `j2k_codec_math::transcode97` changes.
//!
//! The synthetic fixtures are encoded at test time by j2k-jpeg's baseline
//! encoder, whose output is the same on every architecture. [`FIXTURES`] pins
//! those bytes, so an encoder change fails here as an unpinned fixture rather
//! than as a transcode change. The `F64` pins of the six conformance fixtures
//! and the two grayscale synthetic fixtures were checked against the binary64
//! output of `c25a226b`. The integer forward DCT re-encoded the other seven
//! synthetic fixtures; their `F64` pins come from the same binary64 path on
//! the new bytes.

use j2k_transcode::{
    jpeg_to_htj2k, jpeg_to_htj2k_batch, JpegTileBatchInput, JpegToHtj2kError, TranscodePrecision,
};
use j2k_transcode_test_support::transcode97_matrix::{
    fnv1a64, matrix_jpegs, matrix_options, MATRIX_VARIANTS,
};

/// `(length, FNV-1a 64)` of one matrix codestream at each precision.
struct Pin {
    jpeg: &'static str,
    variant: &'static str,
    f64: (usize, u64),
    f32: (usize, u64),
}

const fn pin(
    jpeg: &'static str,
    variant: &'static str,
    f64: (usize, u64),
    f32: (usize, u64),
) -> Pin {
    Pin {
        jpeg,
        variant,
        f64,
        f32,
    }
}

/// FNV-1a 64 of every matrix JPEG; [`PINS`] belong to these bytes.
#[rustfmt::skip]
const FIXTURES: [(&str, u64); 15] = [
    ("conformance_grayscale_8x8", 0x2235_331d_9ead_15bd),
    ("conformance_444_8x8", 0x82c9_1d8b_2d53_2da4),
    ("conformance_422_16x8", 0x8adb_49e2_73aa_81c6),
    ("conformance_420_16x16", 0x6e50_f26a_3f75_03b5),
    ("conformance_420_restart_32x16", 0xe073_d7f4_d4d6_176f),
    ("conformance_444_restart_67x45", 0xe865_c9b5_9984_10cd),
    ("synthetic_444_224_restart", 0x1b52_302c_b46d_036a),
    ("synthetic_420_224_restart", 0xc95a_ea83_7f4b_8afb),
    ("synthetic_420_512_restart", 0x2ddb_98db_f23d_4064),
    ("synthetic_gray_100x37_restart", 0x55df_ce53_e0ed_4921),
    ("synthetic_422_131x77", 0xdd75_38ab_b34a_6514),
    ("synthetic_420_97x61_restart", 0xfbf2_994a_edbe_c3ff),
    ("synthetic_444_45x67_restart", 0xc0f5_dcfc_8870_1909),
    ("synthetic_gray_1031x9", 0x7d18_8f72_47dc_5a56),
    ("synthetic_420_1100x1030_restart", 0xfd0f_dd2f_706f_c810),
];

/// `(length, FNV-1a 64)` of every matrix codestream at `F64` and at `F32`.
#[rustfmt::skip]
const PINS: [Pin; 60] = [
    pin("conformance_grayscale_8x8", "default", (143, 0xc6d6_a1d5_ce6f_650f), (143, 0xc6d6_a1d5_ce6f_650f)),
    pin("conformance_grayscale_8x8", "levels3_cb32x16", (153, 0x3b3d_99b5_2c69_1b49), (153, 0x3b3d_99b5_2c69_1b49)),
    pin("conformance_grayscale_8x8", "levels5_scale1", (157, 0x5cd2_7a5e_4483_0984), (157, 0x5cd2_7a5e_4483_0984)),
    pin("conformance_grayscale_8x8", "levels2_layers2", (149, 0xb491_78a6_ad8c_35a8), (149, 0xb491_78a6_ad8c_35a8)),
    pin("conformance_444_8x8", "default", (254, 0x3e14_98f5_3e6a_277b), (254, 0x3e14_98f5_3e6a_277b)),
    pin("conformance_444_8x8", "levels3_cb32x16", (316, 0xa9d8_bc23_3f41_ec8d), (316, 0xa9d8_bc23_3f41_ec8d)),
    pin("conformance_444_8x8", "levels5_scale1", (344, 0x4b86_dda3_b93b_3e45), (344, 0x4b86_dda3_b93b_3e45)),
    pin("conformance_444_8x8", "levels2_layers2", (283, 0xa42d_03c1_a500_c1df), (283, 0xa42d_03c1_a500_c1df)),
    pin("conformance_422_16x8", "default", (283, 0x6550_9d3a_d9ae_bfdf), (283, 0x6550_9d3a_d9ae_bfdf)),
    pin("conformance_422_16x8", "levels3_cb32x16", (337, 0x2232_7423_0b6d_40e1), (337, 0x2232_7423_0b6d_40e1)),
    pin("conformance_422_16x8", "levels5_scale1", (371, 0xf184_6665_f2ae_9afb), (371, 0xf184_6665_f2ae_9afb)),
    pin("conformance_422_16x8", "levels2_layers2", (295, 0x7d59_a180_4374_a22b), (295, 0x7d59_a180_4374_a22b)),
    pin("conformance_420_16x16", "default", (315, 0x9194_c8a3_063e_0ccf), (315, 0x9194_c8a3_063e_0ccf)),
    pin("conformance_420_16x16", "levels3_cb32x16", (340, 0x160b_be94_be59_970c), (340, 0x160b_be94_be59_970c)),
    pin("conformance_420_16x16", "levels5_scale1", (391, 0x2ddc_3bc7_5f2b_2e79), (391, 0x2ddc_3bc7_5f2b_2e79)),
    pin("conformance_420_16x16", "levels2_layers2", (317, 0xb6be_e06e_4484_caad), (317, 0xb6be_e06e_4484_caad)),
    pin("conformance_420_restart_32x16", "default", (449, 0x007e_dcc5_917a_69df), (449, 0x007e_dcc5_917a_69df)),
    pin("conformance_420_restart_32x16", "levels3_cb32x16", (412, 0xa6fb_7b99_3d09_9718), (412, 0xa6fb_7b99_3d09_9718)),
    pin("conformance_420_restart_32x16", "levels5_scale1", (506, 0x500e_05ad_5955_badb), (506, 0x500e_05ad_5955_badb)),
    pin("conformance_420_restart_32x16", "levels2_layers2", (375, 0xed39_3ac2_d1e3_589f), (375, 0xed39_3ac2_d1e3_589f)),
    pin("conformance_444_restart_67x45", "default", (5399, 0xd3a8_0072_a4e2_c06b), (5399, 0xd3a8_0072_a4e2_c06b)),
    pin("conformance_444_restart_67x45", "levels3_cb32x16", (5144, 0x1ae4_ef2c_ac2b_2059), (5144, 0x1ae4_ef2c_ac2b_2059)),
    pin("conformance_444_restart_67x45", "levels5_scale1", (6266, 0xa175_b356_c54f_19b6), (6266, 0xa175_b356_c54f_19b6)),
    pin("conformance_444_restart_67x45", "levels2_layers2", (4406, 0x0286_97a2_b24c_ac69), (4406, 0x0286_97a2_b24c_ac69)),
    pin("synthetic_444_224_restart", "default", (132_342, 0x3577_5504_435c_829e), (132_342, 0xba71_79c6_c7d9_92de)),
    pin("synthetic_444_224_restart", "levels3_cb32x16", (131_291, 0x9448_ac87_03ca_e597), (131_291, 0x2685_8255_5459_387f)),
    pin("synthetic_444_224_restart", "levels5_scale1", (148_903, 0x102b_8cb5_4929_a0cd), (148_903, 0xcbd0_6d7c_e262_81b6)),
    pin("synthetic_444_224_restart", "levels2_layers2", (112_887, 0x0099_ed74_9cf9_46e9), (112_887, 0x8a70_7f26_2352_8f09)),
    pin("synthetic_420_224_restart", "default", (69_327, 0x4410_c2a0_6843_d665), (69_327, 0x630a_df19_7099_b9c6)),
    pin("synthetic_420_224_restart", "levels3_cb32x16", (69_022, 0xb423_bc45_37a6_2ec4), (69_022, 0xf4da_9f6e_f9f0_d09c)),
    pin("synthetic_420_224_restart", "levels5_scale1", (77_711, 0xec4d_61f0_0fd5_f67f), (77_711, 0x0ba7_ec37_6683_253c)),
    pin("synthetic_420_224_restart", "levels2_layers2", (58_977, 0x48d3_9c4a_f1a8_9349), (58_977, 0x48d3_9c4a_f1a8_9349)),
    pin("synthetic_420_512_restart", "default", (362_309, 0x024d_9e35_8955_4945), (362_309, 0x443e_743e_3480_5ce9)),
    pin("synthetic_420_512_restart", "levels3_cb32x16", (359_569, 0xc639_03bf_a731_7528), (359_569, 0x895f_8dd4_c9e3_ae83)),
    pin("synthetic_420_512_restart", "levels5_scale1", (404_255, 0x9dc2_e22d_4be4_c411), (404_255, 0x2590_c6b0_6b85_272e)),
    pin("synthetic_420_512_restart", "levels2_layers2", (307_874, 0x0cda_41cc_8ac8_ce1b), (307_876, 0x728f_e880_867b_d4e0)),
    pin("synthetic_gray_100x37_restart", "default", (3545, 0x630f_d7d7_6f9e_4167), (3545, 0x630f_d7d7_6f9e_4167)),
    pin("synthetic_gray_100x37_restart", "levels3_cb32x16", (3547, 0xda92_b724_505e_c6c8), (3547, 0xda92_b724_505e_c6c8)),
    pin("synthetic_gray_100x37_restart", "levels5_scale1", (4029, 0x4229_d12a_6f1c_0095), (4029, 0x4229_d12a_6f1c_0095)),
    pin("synthetic_gray_100x37_restart", "levels2_layers2", (3044, 0x13ad_2619_6330_453f), (3044, 0x13ad_2619_6330_453f)),
    pin("synthetic_422_131x77", "default", (18_560, 0x80c2_9f56_dd65_d992), (18_560, 0x80c2_9f56_dd65_d992)),
    pin("synthetic_422_131x77", "levels3_cb32x16", (18_650, 0xd87f_7cdf_fc07_e982), (18_650, 0xd87f_7cdf_fc07_e982)),
    pin("synthetic_422_131x77", "levels5_scale1", (21_051, 0xa3cc_6f32_40a8_3b02), (21_051, 0xa3cc_6f32_40a8_3b02)),
    pin("synthetic_422_131x77", "levels2_layers2", (15_899, 0xe4aa_3db8_50b5_3ac3), (15_899, 0xe4aa_3db8_50b5_3ac3)),
    pin("synthetic_420_97x61_restart", "default", (8323, 0x75f5_c0e4_fdcc_4dc8), (8323, 0x75f5_c0e4_fdcc_4dc8)),
    pin("synthetic_420_97x61_restart", "levels3_cb32x16", (8361, 0x09ee_24b6_9acd_b7ae), (8361, 0x09ee_24b6_9acd_b7ae)),
    pin("synthetic_420_97x61_restart", "levels5_scale1", (9537, 0xf357_2fe6_b330_0a77), (9537, 0x86b1_18e7_3ede_01fc)),
    pin("synthetic_420_97x61_restart", "levels2_layers2", (7201, 0xd9bf_e3d2_0a53_ad15), (7201, 0xd9bf_e3d2_0a53_ad15)),
    pin("synthetic_444_45x67_restart", "default", (7788, 0x13ac_8be1_00b0_b586), (7788, 0x8114_c579_ec49_97ed)),
    pin("synthetic_444_45x67_restart", "levels3_cb32x16", (7886, 0x8a93_cd12_6c08_8f30), (7886, 0x8a93_cd12_6c08_8f30)),
    pin("synthetic_444_45x67_restart", "levels5_scale1", (9077, 0x3b3a_3ca1_a4ad_045c), (9077, 0x4a12_446e_452f_4bd9)),
    pin("synthetic_444_45x67_restart", "levels2_layers2", (6720, 0x0fff_22a2_226a_0acd), (6720, 0x0fff_22a2_226a_0acd)),
    pin("synthetic_gray_1031x9", "default", (9165, 0xa423_17ad_e862_df91), (9165, 0xa423_17ad_e862_df91)),
    pin("synthetic_gray_1031x9", "levels3_cb32x16", (9241, 0xc660_39f6_ef56_7c44), (9241, 0xc660_39f6_ef56_7c44)),
    pin("synthetic_gray_1031x9", "levels5_scale1", (10_309, 0x96ed_f56e_3cbb_fe6e), (10_309, 0xbf8d_87b1_6585_0cc0)),
    pin("synthetic_gray_1031x9", "levels2_layers2", (7963, 0x5b3e_6aec_aa43_7d45), (7963, 0x5b3e_6aec_aa43_7d45)),
    pin("synthetic_420_1100x1030_restart", "default", (1_567_442, 0x6323_ba88_b184_1ad3), (1_567_442, 0xdb0d_3944_7aa6_300c)),
    pin("synthetic_420_1100x1030_restart", "levels3_cb32x16", (1_557_516, 0x1298_cffa_85e4_e48c), (1_557_516, 0x8186_265c_7c6d_c75e)),
    pin("synthetic_420_1100x1030_restart", "levels5_scale1", (1_748_554, 0xe43c_e138_fb88_c2bf), (1_748_554, 0xf9d8_d1a7_5177_a4f8)),
    pin("synthetic_420_1100x1030_restart", "levels2_layers2", (1_332_698, 0x864c_c3a3_0e1f_84b0), (1_332_702, 0x1cde_04d5_5ac6_c6c0)),
];

/// The pins for fixture `name`, after checking that it encoded to the pinned
/// bytes.
fn pins_for_fixture(name: &str, bytes: &[u8]) -> &'static [Pin] {
    let digest = fnv1a64(bytes);
    assert!(
        FIXTURES.contains(&(name, digest)),
        "fixture {name} encoded to unpinned bytes (FNV-1a 64 {digest:016x})"
    );
    &PINS
}

fn assert_matrix_pins(precision: TranscodePrecision) {
    let jpegs = matrix_jpegs();
    assert_eq!(jpegs.len(), FIXTURES.len(), "every fixture must be pinned");
    let mut checked = 0usize;
    for jpeg in &jpegs {
        let pins = pins_for_fixture(jpeg.name, &jpeg.bytes);
        for variant in MATRIX_VARIANTS {
            let expected = pins
                .iter()
                .find(|pin| pin.jpeg == jpeg.name && pin.variant == variant.name)
                .unwrap_or_else(|| panic!("missing pin for {} / {}", jpeg.name, variant.name));
            let (len, digest) = match precision {
                TranscodePrecision::F64 => expected.f64,
                _ => expected.f32,
            };
            let mut options = matrix_options(variant);
            options.precision = precision;

            let single = jpeg_to_htj2k(&jpeg.bytes, &options)
                .unwrap_or_else(|error| panic!("{} / {}: {error}", jpeg.name, variant.name))
                .codestream;
            assert_eq!(
                (single.len(), fnv1a64(&single)),
                (len, digest),
                "{precision:?} single-tile codestream changed for {} / {}",
                jpeg.name,
                variant.name
            );

            let tiles = [JpegTileBatchInput { bytes: &jpeg.bytes }; 2];
            let batch = jpeg_to_htj2k_batch(&tiles, &options)
                .unwrap_or_else(|error| panic!("{} / {} batch: {error}", jpeg.name, variant.name));
            for tile in batch.tiles {
                let tile = tile.unwrap_or_else(|error| {
                    panic!("{} / {} batch tile: {error}", jpeg.name, variant.name)
                });
                assert_eq!(
                    tile.codestream, single,
                    "{precision:?} batch tile differs from single tile for {} / {}",
                    jpeg.name, variant.name
                );
            }
            checked += 1;
        }
    }
    assert_eq!(
        checked,
        FIXTURES.len() * MATRIX_VARIANTS.len(),
        "every fixture and variant must be checked"
    );
}

#[test]
fn f64_precision_reproduces_the_previous_default_bytes() {
    assert_matrix_pins(TranscodePrecision::F64);
}

#[test]
fn default_f32_precision_matches_the_canonical_pins() {
    assert_eq!(
        j2k_transcode::JpegToHtj2kOptions::lossy_97().precision,
        TranscodePrecision::F32
    );
    assert_matrix_pins(TranscodePrecision::F32);
}

#[test]
fn twelve_bit_jpeg_is_rejected_at_both_precisions() {
    let jpeg = j2k_test_support::extended_12bit_grayscale_8x8_jpeg();
    for precision in [TranscodePrecision::F32, TranscodePrecision::F64] {
        let mut options = matrix_options(MATRIX_VARIANTS[0]);
        options.precision = precision;
        let error = jpeg_to_htj2k(&jpeg, &options).expect_err("12-bit JPEG is not transcodable");
        assert!(
            matches!(error, JpegToHtj2kError::Jpeg(_)),
            "{precision:?}: unexpected error {error:?}"
        );
    }
}
