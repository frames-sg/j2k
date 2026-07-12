// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    JpegCachedPlan, JpegCachedPlanBuildError, JpegFastPacketState, JpegPlanCache,
    JpegPlanCacheError, SharedJpegFastPacket, SharedJpegInput,
};
use super::{rewrite_first_sof_quant_table_selector, unsupported_summary};
use crate::adapter::{
    build_fast420_packet, summarize_device_batch, DeviceBatchSummary, JpegFastPacketFamily,
};
use crate::{ColorSpace, Decoder};
use j2k_test_support::{
    rgb_app14_8x8_jpeg, JPEG_BASELINE_420_16X16, JPEG_BASELINE_422_16X8, JPEG_BASELINE_444_8X8,
    JPEG_GRAYSCALE_8X8,
};

#[test]
fn inspect_once_builder_produces_each_ready_family_with_canonical_summary() {
    for (bytes, expected_family) in [
        (JPEG_BASELINE_420_16X16, JpegFastPacketFamily::Fast420),
        (JPEG_BASELINE_422_16X8, JpegFastPacketFamily::Fast422),
        (JPEG_BASELINE_444_8X8, JpegFastPacketFamily::Fast444),
    ] {
        let input = SharedJpegInput::try_copy_from_slice(bytes).expect("copy fixture");
        let retained_input = input.clone();
        let plan = JpegCachedPlan::build(input).expect("build cached JPEG plan");
        let expected_summary =
            summarize_device_batch(&Decoder::new(bytes).expect("construct fixture decoder"), 4);

        assert_eq!(plan.batch_summary(), expected_summary);
        assert_eq!(
            plan.fast_packet()
                .expect("ready fixture packet")
                .as_packet()
                .family(),
            expected_family
        );
        assert!(SharedJpegInput::ptr_eq(plan.input(), &retained_input));
        let cloned = plan.clone();
        assert!(SharedJpegFastPacket::ptr_eq(
            plan.fast_packet().expect("original packet"),
            cloned.fast_packet().expect("cloned packet")
        ));
    }
}

#[test]
fn app14_rgb_444_retains_color_mode_and_one_validated_family() {
    let bytes = rgb_app14_8x8_jpeg();
    let input = SharedJpegInput::try_copy_from_slice(&bytes).expect("copy RGB fixture");
    let plan = JpegCachedPlan::build(input).expect("build RGB cached plan");

    assert_eq!(plan.color_space(), ColorSpace::Rgb);
    assert!(plan.batch_summary().matches_fast_444);
    assert!(!plan.batch_summary().matches_fast_420);
    assert!(!plan.batch_summary().matches_fast_422);
    assert_eq!(
        plan.fast_packet()
            .expect("ready RGB packet")
            .as_packet()
            .family(),
        JpegFastPacketFamily::Fast444
    );
}

#[test]
fn grayscale_is_an_explicit_unsupported_plan_with_no_fast_flags() {
    let input = SharedJpegInput::try_copy_from_slice(JPEG_GRAYSCALE_8X8).expect("copy fixture");
    let plan = JpegCachedPlan::build(input).expect("build unsupported plan");
    let expected = summarize_device_batch(
        &Decoder::new(JPEG_GRAYSCALE_8X8).expect("construct fixture decoder"),
        4,
    );

    assert_eq!(plan.batch_summary(), expected);
    assert!(matches!(
        plan.packet_state(),
        JpegFastPacketState::Unsupported
    ));
    assert!(plan.fast_packet().is_none());
}

#[test]
fn malformed_decode_and_hard_packet_errors_remain_typed_and_uncached() {
    let mut cache = JpegPlanCache::new();
    let malformed = SharedJpegInput::try_copy_from_slice(&[0xff, 0xd8]).expect("copy malformed");
    assert!(matches!(
        JpegCachedPlan::build(malformed),
        Err(JpegCachedPlanBuildError::Decode(_))
    ));

    let invalid_table =
        rewrite_first_sof_quant_table_selector(JPEG_BASELINE_420_16X16.to_vec(), u8::MAX);
    let input = SharedJpegInput::try_copy_from_slice(&invalid_table).expect("copy invalid table");
    assert!(matches!(
        JpegCachedPlan::build(input),
        Err(JpegCachedPlanBuildError::FastPacket(_))
    ));
    assert_eq!(cache.diagnostics().entries, 0);
    assert!(cache.get(&invalid_table).is_none());
}

#[test]
fn plan_state_rejects_ambiguous_or_contradictory_family_summaries() {
    let unsupported = JpegCachedPlan::try_new(
        SharedJpegInput::try_copy_from_slice(b"unsupported").expect("copy input"),
        DeviceBatchSummary {
            matches_fast_420: true,
            ..unsupported_summary()
        },
        ColorSpace::YCbCr,
        JpegFastPacketState::Unsupported,
    );
    assert!(matches!(unsupported, Err(JpegPlanCacheError::Invariant(_))));

    let packet = SharedJpegFastPacket::try_new(
        build_fast420_packet(JPEG_BASELINE_420_16X16)
            .expect("build packet")
            .into(),
    )
    .expect("share packet");
    for summary in [
        unsupported_summary(),
        DeviceBatchSummary {
            matches_fast_420: false,
            matches_fast_422: true,
            ..unsupported_summary()
        },
        DeviceBatchSummary {
            matches_fast_420: true,
            matches_fast_422: true,
            ..unsupported_summary()
        },
    ] {
        let result = JpegCachedPlan::try_new(
            SharedJpegInput::try_copy_from_slice(JPEG_BASELINE_420_16X16).expect("copy fixture"),
            summary,
            ColorSpace::YCbCr,
            JpegFastPacketState::Ready(packet.clone()),
        );
        assert!(matches!(result, Err(JpegPlanCacheError::Invariant(_))));
    }
}
