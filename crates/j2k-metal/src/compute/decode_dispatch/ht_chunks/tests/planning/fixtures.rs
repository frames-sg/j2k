// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_native::{DecodeSettings, DecoderContext, EncodeOptions, Image};

use super::super::super::{
    default_metal_ht_chunk_limits, metal_ht_pipeline_kind_for_bucket, plan_metal_ht_chunks,
    HtBatchInput, MetalHtPipelineKind,
};
use crate::compute::{
    prepare_direct_color_plan, prepare_direct_grayscale_plan, PreparedDirectGrayscalePlan,
};

#[test]
fn prepared_fixtures_select_expected_dedicated_metal_ht_pipelines() {
    let cleanup_pixels = j2k_test_support::patterned_gray8(32, 32);
    let cleanup_bytes = j2k_native::encode_htj2k(
        &cleanup_pixels,
        32,
        32,
        1,
        8,
        false,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            ..EncodeOptions::default()
        },
    )
    .expect("encode cleanup-only fixture");
    let cleanup = prepare_grayscale_fixture(&cleanup_bytes, "cleanup-only");
    assert_fixture_selects_pipeline(
        "generated cleanup-only",
        &prepared_fixture_pipeline_kinds(core::slice::from_ref(&cleanup)),
        MetalHtPipelineKind::CleanupOnly,
    );

    let sigprop_bytes = j2k_test_support::openhtj2k_sigprop_fixture();
    let sigprop_image = Image::new(sigprop_bytes, &DecodeSettings::strict())
        .expect("independent OpenHTJ2K SigProp fixture");
    let mut sigprop_context = DecoderContext::default();
    let sigprop_direct = sigprop_image
        .build_direct_color_plan_with_context(&mut sigprop_context)
        .expect("independent OpenHTJ2K SigProp direct plan");
    let sigprop = prepare_direct_color_plan(&sigprop_direct)
        .expect("prepare independent OpenHTJ2K SigProp plan");
    assert_fixture_selects_pipeline(
        "independent OpenHTJ2K SigProp",
        &prepared_fixture_pipeline_kinds(&sigprop.component_plans),
        MetalHtPipelineKind::SigProp,
    );

    let magref = prepare_grayscale_fixture(
        j2k_test_support::openhtj2k_refinement_fixture(),
        "OpenHTJ2K MagRef",
    );
    assert_fixture_selects_pipeline(
        "OpenHTJ2K MagRef",
        &prepared_fixture_pipeline_kinds(core::slice::from_ref(&magref)),
        MetalHtPipelineKind::MagRef,
    );
}

fn prepare_grayscale_fixture(bytes: &[u8], name: &str) -> PreparedDirectGrayscalePlan {
    let image = Image::new(bytes, &DecodeSettings::strict())
        .unwrap_or_else(|error| panic!("parse {name} fixture: {error}"));
    let mut context = DecoderContext::default();
    let direct = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .unwrap_or_else(|error| panic!("build {name} direct plan: {error}"));
    prepare_direct_grayscale_plan(&direct)
        .unwrap_or_else(|error| panic!("prepare {name} Metal plan: {error}"))
}

fn prepared_fixture_pipeline_kinds(
    component_plans: &[PreparedDirectGrayscalePlan],
) -> Vec<MetalHtPipelineKind> {
    let mut kinds = Vec::new();
    for (source_index, component) in component_plans.iter().enumerate() {
        for group in &component.ht_groups {
            let input = HtBatchInput {
                source_index,
                payload: group.payload_source.as_ht_payload_source(),
                jobs: &group.jobs,
                output_base: 0,
                execution_owner: &group.execution_owner,
            };
            let plan = plan_metal_ht_chunks(&[input], default_metal_ht_chunk_limits())
                .expect("plan prepared fixture HT chunks");
            for chunk_index in 0..plan.chunk_count() {
                let chunk = plan
                    .pack_chunk(chunk_index)
                    .expect("pack prepared fixture HT chunk");
                let kind = metal_ht_pipeline_kind_for_bucket(chunk.bucket);
                if !kinds.contains(&kind) {
                    kinds.push(kind);
                }
            }
        }
    }
    assert!(
        !kinds.is_empty(),
        "prepared fixture must contain planned HT jobs"
    );
    kinds
}

fn assert_fixture_selects_pipeline(
    name: &str,
    actual: &[MetalHtPipelineKind],
    expected: MetalHtPipelineKind,
) {
    assert!(
        actual.contains(&expected),
        "{name} planned pipelines {actual:?} must include {expected:?}"
    );
}
