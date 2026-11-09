// SPDX-License-Identifier: MIT OR Apache-2.0

//! Byte pins for the irreversible 9/7 JPEG-to-HTJ2K transcode at both
//! [`TranscodePrecision`] settings.
//!
//! `F64` must keep reproducing the binary64 CPU output of commit `c25a226b`
//! (the default before 0.14) byte for byte. `F32` is the canonical binary32
//! path that the Metal and CUDA backends reproduce; its digests change only
//! when the canonical arithmetic in `j2k_codec_math::transcode97` changes.
//!
//! The synthetic fixtures are encoded at test time by j2k-jpeg's baseline
//! encoder, whose RGB output for six of them differs between `aarch64` and
//! `x86_64`. Pins are therefore keyed by the fixture's own digest: [`PINS`]
//! belong to the `aarch64` encoding (which every other fixture shares on both
//! architectures) and [`X86_64_PINS`] to the `x86_64` encoding of those six.
//! Both sets were checked against `c25a226b` on their architecture.

use j2k_transcode::{
    jpeg_to_htj2k, jpeg_to_htj2k_batch, JpegTileBatchInput, JpegToHtj2kError, TranscodePrecision,
};
use j2k_transcode_test_support::transcode97_matrix::{
    fnv1a64, matrix_jpegs, matrix_options, MATRIX_VARIANTS,
};

struct Pin {
    jpeg: &'static str,
    variant: &'static str,
    len: usize,
    f64_digest: u64,
    f32_digest: u64,
}

const fn pin(
    jpeg: &'static str,
    variant: &'static str,
    len: usize,
    f64_digest: u64,
    f32_digest: u64,
) -> Pin {
    Pin {
        jpeg,
        variant,
        len,
        f64_digest,
        f32_digest,
    }
}

/// FNV-1a 64 of every matrix JPEG as encoded on `aarch64`; [`PINS`] belong to
/// these bytes.
#[rustfmt::skip]
const FIXTURES: [(&str, u64); 15] = [
    ("conformance_grayscale_8x8", 0x2235_331d_9ead_15bd),
    ("conformance_444_8x8", 0x82c9_1d8b_2d53_2da4),
    ("conformance_422_16x8", 0x8adb_49e2_73aa_81c6),
    ("conformance_420_16x16", 0x6e50_f26a_3f75_03b5),
    ("conformance_420_restart_32x16", 0xe073_d7f4_d4d6_176f),
    ("conformance_444_restart_67x45", 0xe865_c9b5_9984_10cd),
    ("synthetic_444_224_restart", 0xf419_6cb1_d78f_71d3),
    ("synthetic_420_224_restart", 0x8922_82b2_2df4_10d7),
    ("synthetic_420_512_restart", 0x4078_af63_9901_67f0),
    ("synthetic_gray_100x37_restart", 0x55df_ce53_e0ed_4921),
    ("synthetic_422_131x77", 0x7445_3b93_6572_5cd7),
    ("synthetic_420_97x61_restart", 0x136f_3432_7221_c74b),
    ("synthetic_444_45x67_restart", 0xa656_416e_dacf_80ca),
    ("synthetic_gray_1031x9", 0x7d18_8f72_47dc_5a56),
    ("synthetic_420_1100x1030_restart", 0x3c85_8738_2129_71b4),
];

/// FNV-1a 64 of the six synthetic RGB fixtures whose `x86_64` encoding differs;
/// [`X86_64_PINS`] belong to these bytes.
#[rustfmt::skip]
const X86_64_FIXTURES: [(&str, u64); 6] = [
    ("synthetic_444_224_restart", 0xdda6_a346_68d7_b796),
    ("synthetic_420_224_restart", 0x3b28_3c2a_8817_c0fd),
    ("synthetic_420_512_restart", 0x5a28_5352_16a9_6428),
    ("synthetic_422_131x77", 0xa1bc_2c27_27a9_fc3d),
    ("synthetic_444_45x67_restart", 0xc0f5_dcfc_8870_1909),
    ("synthetic_420_1100x1030_restart", 0xb72c_a515_0b21_e97e),
];

/// `(len, FNV-1a 64)` of every matrix codestream; the `F64` column was
/// produced by the default binary64 path at commit `c25a226b`.
#[rustfmt::skip]
const PINS: [Pin; 60] = [
    pin("conformance_grayscale_8x8", "default", 143, 0xc6d6_a1d5_ce6f_650f, 0xc6d6_a1d5_ce6f_650f),
    pin("conformance_grayscale_8x8", "levels3_cb32x16", 153, 0x3b3d_99b5_2c69_1b49, 0x3b3d_99b5_2c69_1b49),
    pin("conformance_grayscale_8x8", "levels5_scale1", 157, 0x5cd2_7a5e_4483_0984, 0x5cd2_7a5e_4483_0984),
    pin("conformance_grayscale_8x8", "levels2_layers2", 149, 0xb491_78a6_ad8c_35a8, 0xb491_78a6_ad8c_35a8),
    pin("conformance_444_8x8", "default", 254, 0x3e14_98f5_3e6a_277b, 0x3e14_98f5_3e6a_277b),
    pin("conformance_444_8x8", "levels3_cb32x16", 316, 0xa9d8_bc23_3f41_ec8d, 0xa9d8_bc23_3f41_ec8d),
    pin("conformance_444_8x8", "levels5_scale1", 344, 0x4b86_dda3_b93b_3e45, 0x4b86_dda3_b93b_3e45),
    pin("conformance_444_8x8", "levels2_layers2", 283, 0xa42d_03c1_a500_c1df, 0xa42d_03c1_a500_c1df),
    pin("conformance_422_16x8", "default", 283, 0x6550_9d3a_d9ae_bfdf, 0x6550_9d3a_d9ae_bfdf),
    pin("conformance_422_16x8", "levels3_cb32x16", 337, 0x2232_7423_0b6d_40e1, 0x2232_7423_0b6d_40e1),
    pin("conformance_422_16x8", "levels5_scale1", 371, 0xf184_6665_f2ae_9afb, 0xf184_6665_f2ae_9afb),
    pin("conformance_422_16x8", "levels2_layers2", 295, 0x7d59_a180_4374_a22b, 0x7d59_a180_4374_a22b),
    pin("conformance_420_16x16", "default", 315, 0x9194_c8a3_063e_0ccf, 0x9194_c8a3_063e_0ccf),
    pin("conformance_420_16x16", "levels3_cb32x16", 340, 0x160b_be94_be59_970c, 0x160b_be94_be59_970c),
    pin("conformance_420_16x16", "levels5_scale1", 391, 0x2ddc_3bc7_5f2b_2e79, 0x2ddc_3bc7_5f2b_2e79),
    pin("conformance_420_16x16", "levels2_layers2", 317, 0xb6be_e06e_4484_caad, 0xb6be_e06e_4484_caad),
    pin("conformance_420_restart_32x16", "default", 449, 0x007e_dcc5_917a_69df, 0x007e_dcc5_917a_69df),
    pin("conformance_420_restart_32x16", "levels3_cb32x16", 412, 0xa6fb_7b99_3d09_9718, 0xa6fb_7b99_3d09_9718),
    pin("conformance_420_restart_32x16", "levels5_scale1", 506, 0x500e_05ad_5955_badb, 0x500e_05ad_5955_badb),
    pin("conformance_420_restart_32x16", "levels2_layers2", 375, 0xed39_3ac2_d1e3_589f, 0xed39_3ac2_d1e3_589f),
    pin("conformance_444_restart_67x45", "default", 5399, 0xd3a8_0072_a4e2_c06b, 0xd3a8_0072_a4e2_c06b),
    pin("conformance_444_restart_67x45", "levels3_cb32x16", 5144, 0x1ae4_ef2c_ac2b_2059, 0x1ae4_ef2c_ac2b_2059),
    pin("conformance_444_restart_67x45", "levels5_scale1", 6266, 0xa175_b356_c54f_19b6, 0xa175_b356_c54f_19b6),
    pin("conformance_444_restart_67x45", "levels2_layers2", 4406, 0x0286_97a2_b24c_ac69, 0x0286_97a2_b24c_ac69),
    pin("synthetic_444_224_restart", "default", 132_336, 0x0dd4_7492_12dd_0cf5, 0x2a49_8351_cd4b_58b5),
    pin("synthetic_444_224_restart", "levels3_cb32x16", 131_295, 0xef72_a48f_9939_c9c1, 0x9953_2114_8a0b_1c89),
    pin("synthetic_444_224_restart", "levels5_scale1", 148_905, 0xd712_2a07_ab34_3fa8, 0x5c67_a476_e8df_610b),
    pin("synthetic_444_224_restart", "levels2_layers2", 112_883, 0xac45_398c_cc06_fc4c, 0x139c_5899_04f7_d7ec),
    pin("synthetic_420_224_restart", "default", 69_326, 0x2bf7_86c1_c43b_a108, 0x4994_db6e_42e0_4069),
    pin("synthetic_420_224_restart", "levels3_cb32x16", 69_028, 0x764a_d343_10c9_5168, 0x29cb_a2c6_4ad6_8b70),
    pin("synthetic_420_224_restart", "levels5_scale1", 77_713, 0xe1f5_6f36_d95f_df9d, 0x67e4_beff_0795_eb3c),
    pin("synthetic_420_224_restart", "levels2_layers2", 58_981, 0x4390_ef42_e4b8_8474, 0x4390_ef42_e4b8_8474),
    pin("synthetic_420_512_restart", "default", 362_303, 0x52bd_afe2_15b1_3515, 0x6041_9b61_5da0_add9),
    pin("synthetic_420_512_restart", "levels3_cb32x16", 359_558, 0xe775_4e0e_20d5_6824, 0xc21c_7752_da4b_c0dc),
    pin("synthetic_420_512_restart", "levels5_scale1", 404_250, 0x1c01_b528_e06e_d1bc, 0xfb38_f7d8_0682_d3b7),
    pin("synthetic_420_512_restart", "levels2_layers2", 307_872, 0xa4b4_32c7_5652_d3e5, 0xec38_25d7_b150_ee0a),
    pin("synthetic_gray_100x37_restart", "default", 3545, 0x630f_d7d7_6f9e_4167, 0x630f_d7d7_6f9e_4167),
    pin("synthetic_gray_100x37_restart", "levels3_cb32x16", 3547, 0xda92_b724_505e_c6c8, 0xda92_b724_505e_c6c8),
    pin("synthetic_gray_100x37_restart", "levels5_scale1", 4029, 0x4229_d12a_6f1c_0095, 0x4229_d12a_6f1c_0095),
    pin("synthetic_gray_100x37_restart", "levels2_layers2", 3044, 0x13ad_2619_6330_453f, 0x13ad_2619_6330_453f),
    pin("synthetic_422_131x77", "default", 18_559, 0x68ad_620c_830a_6ac2, 0x68ad_620c_830a_6ac2),
    pin("synthetic_422_131x77", "levels3_cb32x16", 18_650, 0x6ae6_5ef7_0dbf_3fe1, 0x6ae6_5ef7_0dbf_3fe1),
    pin("synthetic_422_131x77", "levels5_scale1", 21_050, 0xd185_8092_3452_a561, 0xd185_8092_3452_a561),
    pin("synthetic_422_131x77", "levels2_layers2", 15_898, 0x11cf_382a_2ae9_1c59, 0x11cf_382a_2ae9_1c59),
    pin("synthetic_420_97x61_restart", "default", 8321, 0xf67e_9a8e_80d0_08fe, 0xf67e_9a8e_80d0_08fe),
    pin("synthetic_420_97x61_restart", "levels3_cb32x16", 8360, 0x8613_beea_3c28_dd3a, 0x8613_beea_3c28_dd3a),
    pin("synthetic_420_97x61_restart", "levels5_scale1", 9539, 0x6aee_9a49_bc5f_6879, 0x0d78_e5e2_22ef_2abe),
    pin("synthetic_420_97x61_restart", "levels2_layers2", 7202, 0xb3ab_30ff_7fde_ea41, 0xb3ab_30ff_7fde_ea41),
    pin("synthetic_444_45x67_restart", "default", 7783, 0xca04_55f7_1d1a_eee1, 0xd8d5_72c6_0ec3_7fe0),
    pin("synthetic_444_45x67_restart", "levels3_cb32x16", 7883, 0x27aa_7014_5138_b60a, 0x27aa_7014_5138_b60a),
    pin("synthetic_444_45x67_restart", "levels5_scale1", 9079, 0x4056_e34a_5ab9_1734, 0x94ee_8170_d37b_50d5),
    pin("synthetic_444_45x67_restart", "levels2_layers2", 6719, 0xe80e_eb9c_dc1c_9aad, 0xe80e_eb9c_dc1c_9aad),
    pin("synthetic_gray_1031x9", "default", 9165, 0xa423_17ad_e862_df91, 0xa423_17ad_e862_df91),
    pin("synthetic_gray_1031x9", "levels3_cb32x16", 9241, 0xc660_39f6_ef56_7c44, 0xc660_39f6_ef56_7c44),
    pin("synthetic_gray_1031x9", "levels5_scale1", 10_309, 0x96ed_f56e_3cbb_fe6e, 0xbf8d_87b1_6585_0cc0),
    pin("synthetic_gray_1031x9", "levels2_layers2", 7963, 0x5b3e_6aec_aa43_7d45, 0x5b3e_6aec_aa43_7d45),
    pin("synthetic_420_1100x1030_restart", "default", 1_567_404, 0x2b29_f666_11e8_5260, 0xdbb9_ba49_8c31_a6e8),
    pin("synthetic_420_1100x1030_restart", "levels3_cb32x16", 1_557_456, 0x12a6_1f17_04a2_3ebc, 0xbe43_01a5_f7ba_a414),
    pin("synthetic_420_1100x1030_restart", "levels5_scale1", 1_748_566, 0x219a_b9b0_13fa_5d40, 0xf797_df9f_a341_2fcb),
    pin("synthetic_420_1100x1030_restart", "levels2_layers2", 1_332_696, 0xdb76_6c58_954c_709d, 0x7d7f_1c9b_a8af_4bb6),
];

/// Pins for the `x86_64` encoding of the fixtures in [`X86_64_FIXTURES`]; the
/// `F64` column was produced by commit `c25a226b` on `x86_64`.
#[rustfmt::skip]
const X86_64_PINS: [Pin; 24] = [
    pin("synthetic_444_224_restart", "default", 132_343, 0x6c5c_9283_deb6_bbcc, 0xdbc1_8b7b_fcea_79cc),
    pin("synthetic_444_224_restart", "levels3_cb32x16", 131_301, 0xb1cc_f984_2b78_5f23, 0x3d91_62a5_6622_6a0b),
    pin("synthetic_444_224_restart", "levels5_scale1", 148_905, 0xb874_1122_b9b5_042b, 0xe17b_bf04_268f_0018),
    pin("synthetic_444_224_restart", "levels2_layers2", 112_885, 0x587f_e7f2_8b99_fcd2, 0xa1d4_3ea2_a2a1_1772),
    pin("synthetic_420_224_restart", "default", 69_328, 0xacc6_436c_565e_b44d, 0x5fa4_9604_c865_f02c),
    pin("synthetic_420_224_restart", "levels3_cb32x16", 69_029, 0x9480_1462_bb49_b883, 0xe34a_b7c6_5eb0_a1fb),
    pin("synthetic_420_224_restart", "levels5_scale1", 77_714, 0x15d6_7c2f_b5cd_ab5a, 0xffd8_bb79_3a09_53b7),
    pin("synthetic_420_224_restart", "levels2_layers2", 58_981, 0x51fe_339d_ea1d_94b1, 0x51fe_339d_ea1d_94b1),
    pin("synthetic_420_512_restart", "default", 362_298, 0xeff8_f8d0_8323_4337, 0xc131_499b_0cdd_4683),
    pin("synthetic_420_512_restart", "levels3_cb32x16", 359_549, 0x1f73_dcb5_7498_b9b4, 0x0cb1_5f20_ff03_a64c),
    pin("synthetic_420_512_restart", "levels5_scale1", 404_250, 0x9db6_9c0c_8702_18b8, 0x2d26_5dce_6567_0aed),
    pin("synthetic_420_512_restart", "levels2_layers2", 307_866, 0xd7ca_e4ee_e29f_bbf8, 0xe428_5188_3ca9_ca9b),
    pin("synthetic_422_131x77", "default", 18_559, 0x0715_0677_c683_8520, 0x0715_0677_c683_8520),
    pin("synthetic_422_131x77", "levels3_cb32x16", 18_651, 0x5e41_7017_f096_b691, 0x5e41_7017_f096_b691),
    pin("synthetic_422_131x77", "levels5_scale1", 21_050, 0x18a3_4475_690e_acf1, 0x18a3_4475_690e_acf1),
    pin("synthetic_422_131x77", "levels2_layers2", 15_899, 0x9491_fb90_051c_7cad, 0x9491_fb90_051c_7cad),
    pin("synthetic_444_45x67_restart", "default", 7788, 0x13ac_8be1_00b0_b586, 0x8114_c579_ec49_97ed),
    pin("synthetic_444_45x67_restart", "levels3_cb32x16", 7886, 0x8a93_cd12_6c08_8f30, 0x8a93_cd12_6c08_8f30),
    pin("synthetic_444_45x67_restart", "levels5_scale1", 9077, 0x3b3a_3ca1_a4ad_045c, 0x4a12_446e_452f_4bd9),
    pin("synthetic_444_45x67_restart", "levels2_layers2", 6720, 0x0fff_22a2_226a_0acd, 0x0fff_22a2_226a_0acd),
    pin("synthetic_420_1100x1030_restart", "default", 1_567_406, 0x0bc5_0fb7_c0d1_f6e6, 0x2059_2859_c8a0_cbb6),
    pin("synthetic_420_1100x1030_restart", "levels3_cb32x16", 1_557_448, 0x804e_d3a0_a1d2_9763, 0x4701_4dd1_1da3_5167),
    pin("synthetic_420_1100x1030_restart", "levels5_scale1", 1_748_561, 0x42ac_a953_4dd5_e610, 0x3c09_3801_71e3_55c7),
    pin("synthetic_420_1100x1030_restart", "levels2_layers2", 1_332_697, 0x658d_3968_0b8e_24e1, 0x2c2b_8478_b478_efc4),
];

/// The pin set that belongs to this encoding of fixture `name`.
fn pins_for_fixture(name: &str, bytes: &[u8]) -> &'static [Pin] {
    let digest = fnv1a64(bytes);
    let known = |table: &[(&str, u64)]| table.contains(&(name, digest));
    if known(&FIXTURES) {
        &PINS
    } else if known(&X86_64_FIXTURES) {
        &X86_64_PINS
    } else {
        panic!("fixture {name} encoded to unpinned bytes (FNV-1a 64 {digest:016x})")
    }
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
            let digest = match precision {
                TranscodePrecision::F64 => expected.f64_digest,
                _ => expected.f32_digest,
            };
            let mut options = matrix_options(variant);
            options.precision = precision;

            let single = jpeg_to_htj2k(&jpeg.bytes, &options)
                .unwrap_or_else(|error| panic!("{} / {}: {error}", jpeg.name, variant.name))
                .codestream;
            assert_eq!(
                (single.len(), fnv1a64(&single)),
                (expected.len, digest),
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
