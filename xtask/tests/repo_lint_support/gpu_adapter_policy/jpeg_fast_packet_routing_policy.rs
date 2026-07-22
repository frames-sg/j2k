// SPDX-License-Identifier: MIT OR Apache-2.0
//! Single-family, typed-failure JPEG Metal fast-packet routing policy.
use std::fs;

use crate::repo_lint_support::{assert_pattern_checks, read_source_files, repo_root, PatternCheck};

mod error_contract;

#[test]
fn jpeg_metal_fast_packet_selection_builds_and_caches_one_typed_family() {
    let root = repo_root();
    error_contract::assert_policy(root);
    let selection = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/fast_packets.rs"))
        .expect("read JPEG Metal fast-packet selection");
    let metal_error = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/error.rs"))
        .expect("read JPEG Metal error source");
    let session = read_source_files(
        root,
        &[
            "crates/j2k-jpeg-metal/src/session.rs",
            "crates/j2k-jpeg-metal/src/session/tests.rs",
        ],
    );
    let batch = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/batch.rs"))
        .expect("read JPEG Metal batch source");

    assert_pattern_checks(&[
        PatternCheck::new("single-family JPEG Metal packet selection", &selection)
            .required(&[
                "SharedJpegFastPacket",
                "#[cfg(test)]\npub(crate) fn build_shared_fast_packet(",
                "match classify_color_fast_packet_family(decoder)",
                "SharedJpegFastPacket::try_new(JpegFastPacket::Fast420(packet))",
                "Err(source) if source.is_capability_mismatch() => Ok(None)",
                "Err(source) => Err(Error::FastPacket { source })",
                "supported_sampling_selects_exactly_one_matching_family",
                "malformed_packet_failures_remain_typed_hard_errors",
            ])
            .forbidden(&["enum SharedJpegFastPacket", "Arc<[u8]>"]),
        PatternCheck::new("typed JPEG Metal fast-packet crossing", &metal_error).required(&[
            "FastPacket {",
            "source: FastPacketError",
            "Self::FastPacket { source } => Some(source)",
            "JpegPlanCache(#[from] JpegPlanCacheError)",
            "impl From<JpegCachedPlanBuildError> for Error",
            "JpegCachedPlanBuildError::FastPacket(source) => Self::FastPacket { source }",
        ]),
        PatternCheck::new("one inspect-once JPEG Metal session cache", &session)
            .required(&[
                "jpeg_plans: JpegPlanCache",
                "pub(crate) fn resolve_jpeg_plan(",
                "self.jpeg_plans\n            .resolve_with_external_live(input, adapter_live_bytes)",
                "resolve_with_decoder_and_external_live(input, adapter_live_bytes)",
                "input: plan.input().clone()",
                "fast_packet: plan.fast_packet().cloned()",
                "batch::BatchShape::from_summary(plan.batch_summary(), plan.color_space())",
                "repeated_plan_hits_share_input_packet_and_eager_shape",
                "reused_source_pointer_with_new_valid_bytes_never_cross_hits",
                "oversized_plan_is_returned_without_retention_and_diagnostics_are_stable",
            ])
            .forbidden(&[
                "VecDeque",
                "CachedBatchShape",
                "CachedFastPackets",
                "CachedInputAlias",
                "resolve_batch_shape",
                "resolve_fast_packets",
                "build_shared_fast_packet_from_bytes",
            ]),
        PatternCheck::new("eager shared JPEG Metal queued plans", &batch)
            .required(&[
                "input: SharedJpegInput",
                "fast_packet: Option<SharedJpegFastPacket>",
                "shape: BatchShape",
                "pub(crate) fn key(&self) -> BatchKey",
                "shape: self.shape",
            ])
            .forbidden(&["session.resolve_batch_shape", "pub(crate) input: Arc<[u8]>"]),
    ]);

    for builder in [
        "build_fast420_packet(bytes)",
        "build_fast422_packet(bytes)",
        "build_fast444_packet(bytes)",
    ] {
        assert_eq!(
            selection.matches(builder).count(),
            1,
            "central selection must invoke `{builder}` exactly once"
        );
    }
}

#[test]
fn jpeg_metal_routes_do_not_probe_or_erase_individual_packet_builders() {
    let root = repo_root();
    let routes = read_source_files(
        root,
        &[
            "crates/j2k-jpeg-metal/src/decoder.rs",
            "crates/j2k-jpeg-metal/src/lib.rs",
            "crates/j2k-jpeg-metal/src/codec_batch.rs",
            "crates/j2k-jpeg-metal/src/tile_batch.rs",
            "crates/j2k-jpeg-metal/src/routing.rs",
            "crates/j2k-jpeg-metal/src/viewport.rs",
            "crates/j2k-jpeg-metal/src/viewport/policy.rs",
            "crates/j2k-jpeg-metal/src/viewport/resident.rs",
        ],
    );

    assert_pattern_checks(&[
        PatternCheck::new("central JPEG Metal fast-packet routes", &routes)
            .required(&[
                "cache.resolve_with_decoder_and_external_live(input, 0)?",
                "JpegCachedPlan::build_from_view_with_decoder(view, 0)?",
                "resolve_jpeg_plan_with_decoder_and_external_live(",
                "resolve_jpeg_plan_with_external_live(",
                "JpegFastPackets::from_shared(fast_packet.as_ref())",
                "fn has_direct_viewport_packet(decoder: &CpuDecoder<'_>)",
                "plans.resolve_from_decoder_with_external_live(decoder, decoder_live_bytes)?",
                "decoder.fast_packets()",
            ])
            .forbidden(&[
                "build_fast420_packet",
                "build_fast422_packet",
                "build_fast444_packet",
                "None => fast_packets::build_shared_fast_packet(&decoder)?",
                "let fast420_packet =",
                "let fast422_packet =",
                "let fast444_packet =",
                "build_shared_fast_packet(",
                "SharedJpegFastPacket::try_new(",
            ]),
    ]);

    assert!(
        include_str!("jpeg_fast_packet_routing_policy.rs")
            .lines()
            .count()
            < 150,
        "JPEG fast-packet routing policy must stay focused"
    );
}
