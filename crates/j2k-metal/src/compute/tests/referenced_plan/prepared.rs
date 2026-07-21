// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k_native::{encode_htj2k, DecodeSettings, DecoderContext, EncodeOptions, Image};

use crate::compute::{
    decode_prepared_ht_sub_band_group_on_cpu_profile, prepare_direct_grayscale_plan,
    prepare_referenced_htj2k_grayscale_plan, PreparedDirectGrayscaleStep, PreparedHtPayloadSource,
};

#[test]
fn referenced_metal_plan_retains_ranges_without_copying_compressed_payloads() {
    let pixels = (0_u8..64).collect::<Vec<_>>();
    let bytes = Arc::<[u8]>::from(
        encode_htj2k(
            &pixels,
            8,
            8,
            1,
            8,
            false,
            &EncodeOptions {
                reversible: true,
                num_decomposition_levels: 1,
                ..EncodeOptions::default()
            },
        )
        .expect("encode referenced Metal fixture"),
    );
    let image = Image::new(&bytes, &DecodeSettings::strict()).expect("strict HT image");
    let mut owned_context = DecoderContext::default();
    let owned = image
        .build_direct_grayscale_plan_with_context(&mut owned_context)
        .expect("owned direct plan");
    let owned_prepared = prepare_direct_grayscale_plan(&owned).expect("owned Metal plan");
    let mut context = DecoderContext::default();
    let referenced = image
        .build_referenced_htj2k_plan_region_with_context(&mut context, (0, 0, 8, 8))
        .expect("referenced direct plan");

    let prepared = prepare_referenced_htj2k_grayscale_plan(&referenced, &bytes)
        .expect("referenced Metal plan");

    for step in &prepared.steps {
        let PreparedDirectGrayscaleStep::HtSubBand(sub_band) = step else {
            continue;
        };
        let PreparedHtPayloadSource::Referenced { input, ranges } = &sub_band.payload_source else {
            panic!("referenced sub-band must retain input ranges");
        };
        assert!(Arc::ptr_eq(input, &bytes));
        assert_eq!(ranges.len(), sub_band.jobs.len());
    }
    for group in &prepared.ht_groups {
        let PreparedHtPayloadSource::Referenced { input, ranges } = &group.payload_source else {
            panic!("referenced group must retain input ranges");
        };
        assert!(Arc::ptr_eq(input, &bytes));
        assert_eq!(ranges.len(), group.jobs.len());
    }
    let step_capacity = prepared
        .steps
        .iter()
        .map(|step| match step {
            PreparedDirectGrayscaleStep::HtSubBand(sub_band) => {
                sub_band.payload_source.owned_payload_capacity()
            }
            _ => 0,
        })
        .sum::<usize>();
    let group_capacity = prepared
        .ht_groups
        .iter()
        .map(|group| group.payload_source.owned_payload_capacity())
        .sum::<usize>();
    assert_eq!(
        step_capacity + group_capacity,
        0,
        "prepared referenced HT sub-bands and groups must own no compressed payload copies",
    );
    assert_eq!(prepared.ht_groups.len(), owned_prepared.ht_groups.len());
    for (referenced_group, owned_group) in prepared.ht_groups.iter().zip(&owned_prepared.ht_groups)
    {
        let referenced_coefficients =
            decode_prepared_ht_sub_band_group_on_cpu_profile(referenced_group, None)
                .expect("referenced CPU fallback decode");
        let owned_coefficients =
            decode_prepared_ht_sub_band_group_on_cpu_profile(owned_group, None)
                .expect("owned CPU fallback decode");
        assert_eq!(referenced_coefficients, owned_coefficients);
    }
}
