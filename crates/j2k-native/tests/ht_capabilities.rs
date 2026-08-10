// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_native::{
    encode_htj2k, encode_precomputed_htj2k_53, encode_typed_component_planes_53,
    inspect_htj2k_capabilities, DecodeError, DecodeSettings, EncodeOptions,
    EncodeTypedComponentPlane, Htj2kCapabilityMode, Image, J2kForwardDwt53Output, MarkerError,
    PrecomputedHtj2k53Component, PrecomputedHtj2k53Image,
};

#[test]
fn encoded_htj2k_capabilities_are_exposed_with_derived_bmagb() {
    // Four unsigned 12-bit midpoint samples level-shift to zero. BMAGB is
    // therefore selected from the encoded cleanup magnitudes, not precision.
    let samples = [0x00, 0x08, 0x00, 0x08, 0x00, 0x08, 0x00, 0x08];
    let codestream = encode_htj2k(
        &samples,
        2,
        2,
        1,
        12,
        false,
        &EncodeOptions {
            num_decomposition_levels: 0,
            ..Default::default()
        },
    )
    .expect("encode HTJ2K fixture");

    let capabilities = inspect_htj2k_capabilities(&codestream)
        .expect("inspect HTJ2K capabilities")
        .expect("CAP advertises Part 15");

    assert_eq!(capabilities.mode(), Htj2kCapabilityMode::HtOnly);
    assert!(!capabilities.multiple_ht_sets());
    assert!(!capabilities.roi());
    assert!(!capabilities.heterogeneous());
    assert!(!capabilities.ht_irreversible());
    assert_eq!(capabilities.magnitude_bound(), 8);
    assert!(capabilities.corresponding_profile().is_none());
}

#[test]
fn capability_inspection_preserves_cod_quality_layer_count() {
    let codestream = encode_htj2k(
        &[0, 1, 2, 3],
        2,
        2,
        1,
        8,
        false,
        &EncodeOptions {
            num_decomposition_levels: 0,
            num_layers: 3,
            ..Default::default()
        },
    )
    .expect("encode multilayer HTJ2K fixture");

    let capabilities = inspect_htj2k_capabilities(&codestream)
        .expect("inspect multilayer HTJ2K capabilities")
        .expect("CAP advertises Part 15");

    assert_eq!(capabilities.quality_layers(), 3);
}

#[test]
fn encoded_roi_is_declared_and_increases_the_magnitude_bound() {
    let codestream = encode_htj2k(
        &[0, 1, 2, 3],
        2,
        2,
        1,
        8,
        false,
        &EncodeOptions {
            num_decomposition_levels: 0,
            roi_component_shifts: vec![5],
            ..Default::default()
        },
    )
    .expect("encode HTJ2K ROI fixture");

    let capabilities = inspect_htj2k_capabilities(&codestream)
        .expect("inspect HTJ2K capabilities")
        .expect("CAP advertises Part 15");

    assert!(capabilities.roi());
    assert!(capabilities.magnitude_bound() >= 12);
}

#[test]
fn multitile_encoder_aggregates_actual_cleanup_magnitudes() {
    let codestream = encode_htj2k(
        &[0x00, 0x08, 0x00, 0x08, 0x00, 0x08, 0x00, 0x08],
        2,
        2,
        1,
        12,
        false,
        &EncodeOptions {
            num_decomposition_levels: 0,
            tile_size: Some((1, 1)),
            ..Default::default()
        },
    )
    .expect("encode multi-tile HTJ2K fixture");

    let capabilities = inspect_htj2k_capabilities(&codestream)
        .expect("inspect multi-tile HTJ2K capabilities")
        .expect("CAP advertises Part 15");
    assert_eq!(capabilities.magnitude_bound(), 8);
}

#[test]
fn precomputed_encoder_preserves_actual_cleanup_magnitude_bound() {
    let image = PrecomputedHtj2k53Image {
        width: 2,
        height: 2,
        bit_depth: 12,
        signed: false,
        components: vec![PrecomputedHtj2k53Component {
            x_rsiz: 1,
            y_rsiz: 1,
            dwt: J2kForwardDwt53Output {
                ll: vec![0.0; 4],
                ll_width: 2,
                ll_height: 2,
                levels: Vec::new(),
            },
        }],
    };
    let codestream = encode_precomputed_htj2k_53(
        &image,
        &EncodeOptions {
            num_decomposition_levels: 0,
            ..Default::default()
        },
    )
    .expect("encode precomputed HTJ2K fixture");

    let capabilities = inspect_htj2k_capabilities(&codestream)
        .expect("inspect precomputed HTJ2K capabilities")
        .expect("CAP advertises Part 15");
    assert_eq!(capabilities.magnitude_bound(), 8);
}

#[test]
fn typed_encoder_preserves_actual_cleanup_magnitude_bound() {
    let samples = [0x00, 0x08, 0x00, 0x08, 0x00, 0x08, 0x00, 0x08];
    let planes = [EncodeTypedComponentPlane {
        data: &samples,
        x_rsiz: 1,
        y_rsiz: 1,
        bit_depth: 12,
        signed: false,
    }];
    let codestream = encode_typed_component_planes_53(
        &planes,
        2,
        2,
        &EncodeOptions {
            num_decomposition_levels: 0,
            use_ht_block_coding: true,
            ..Default::default()
        },
    )
    .expect("encode typed HTJ2K fixture");

    let capabilities = inspect_htj2k_capabilities(&codestream)
        .expect("inspect typed HTJ2K capabilities")
        .expect("CAP advertises Part 15");
    assert_eq!(capabilities.magnitude_bound(), 8);
}

#[test]
fn corresponding_profile_words_and_number_are_preserved() {
    let mut codestream = encoded_fixture();
    let sot = marker_offset(&codestream, 0x90);
    codestream.splice(sot..sot, [0xFF, 0x59, 0x00, 0x04, 0x00, 0x02]);

    let capabilities = inspect_htj2k_capabilities(&codestream)
        .expect("inspect HTJ2K capabilities")
        .expect("CAP advertises Part 15");
    let profile = capabilities.corresponding_profile().expect("CPF profile");

    assert_eq!(profile.words(), &[2]);
    assert_eq!(profile.number_u64(), Some(1));
}

#[test]
fn ccap15_uses_only_the_standard_magnitude_bounds() {
    let mut codestream = encoded_fixture();
    let expected = [
        (0, 8),
        (3, 11),
        (7, 15),
        (12, 20),
        (19, 27),
        (20, 31),
        (30, 71),
        (31, 74),
    ];

    for (p, bound) in expected {
        rewrite_ccap15(&mut codestream, |value| (value & !0x1F) | p);
        let capabilities = inspect_htj2k_capabilities(&codestream)
            .expect("inspect magnitude bound")
            .expect("Part 15 capabilities");
        assert_eq!(capabilities.magnitude_bound(), bound);
    }

    let bounds = (0..=31)
        .map(|p| {
            rewrite_ccap15(&mut codestream, |value| (value & !0x1F) | p);
            inspect_htj2k_capabilities(&codestream)
                .expect("inspect magnitude bound")
                .expect("Part 15 capabilities")
                .magnitude_bound()
        })
        .collect::<Vec<_>>();
    assert!(!bounds.contains(&30), "MAGB30 is not a T.814 set");
}

#[test]
fn main_cod_ht_and_mixed_bits_are_reported_independently() {
    let mut codestream = encoded_fixture();
    let cod = marker_offset(&codestream, 0x52);
    codestream[cod + 12] = 0xC0;
    rewrite_ccap15(&mut codestream, |value| (value & 0x3FFF) | 0xC000);

    let capabilities = inspect_htj2k_capabilities(&codestream)
        .expect("inspect HT style bits")
        .expect("Part 15 capabilities");

    assert!(capabilities.default_ht_block_coding());
    assert!(capabilities.default_mixed_block_coding());
}

#[test]
fn reserved_ht_mode_is_rejected_by_inspection_and_production_decode() {
    let mut codestream = encoded_fixture();
    rewrite_ccap15(&mut codestream, |value| (value & 0x3FFF) | 0x4000);

    assert!(inspect_htj2k_capabilities(&codestream).is_err());
    assert!(matches!(
        Image::new(&codestream, &DecodeSettings::default()),
        Err(DecodeError::Marker(MarkerError::ParseFailure(
            "CAP reserved HT mode"
        )))
    ));
}

fn rewrite_ccap15(codestream: &mut [u8], rewrite: impl FnOnce(u16) -> u16) {
    let cap = marker_offset(codestream, 0x50);
    let value = u16::from_be_bytes([codestream[cap + 8], codestream[cap + 9]]);
    codestream[cap + 8..cap + 10].copy_from_slice(&rewrite(value).to_be_bytes());
}

fn encoded_fixture() -> Vec<u8> {
    encode_htj2k(
        &[0, 1, 2, 3],
        2,
        2,
        1,
        8,
        false,
        &EncodeOptions {
            num_decomposition_levels: 0,
            ..Default::default()
        },
    )
    .expect("encode HTJ2K fixture")
}

fn marker_offset(codestream: &[u8], marker: u8) -> usize {
    codestream
        .windows(2)
        .position(|bytes| bytes == [0xFF, marker])
        .expect("fixture marker")
}
