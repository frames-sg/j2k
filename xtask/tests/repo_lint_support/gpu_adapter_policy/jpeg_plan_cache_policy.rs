// SPDX-License-Identifier: MIT OR Apache-2.0

//! Backend-neutral JPEG plan ownership and cache structure policy.

use std::fs;

use super::super::rust_function_policy::FunctionCalls;
use super::super::{assert_pattern_checks, contains_normalized, repo_root, PatternCheck};

const CACHE_ROOT: &str = "crates/j2k-jpeg/src/adapter/fast_packet/cache";

#[test]
fn jpeg_plan_cache_is_flat_randomized_full_input_lru_with_optional_admission() {
    let store = read("store.rs");
    let resolve = [
        read("store/resolve.rs"),
        read("store/resolve/borrowed.rs"),
        read("store/resolve/shared.rs"),
        read("store/resolve/existing_decoder.rs"),
        read("store/resolve/miss.rs"),
        read("store/resolve/accounting.rs"),
    ]
    .join("\n");
    let state_owner = read("store/state.rs");
    let state_admission = read("store/state/admission.rs");
    let state = format!("{state_owner}\n{state_admission}");
    let diagnostics = read("store/diagnostics.rs");
    let resolve_miss = read("store/resolve/miss.rs");
    let combined = format!("{store}\n{resolve}\n{state}\n{diagnostics}");

    assert_pattern_checks(&[
        PatternCheck::new("neutral JPEG plan cache defaults and identity", &combined).required(&[
            "DEFAULT_JPEG_PLAN_CACHE_ENTRIES: usize = 8",
            "DEFAULT_JPEG_PLAN_CACHE_HOST_BYTES: usize = 64 * 1024 * 1024",
            "RandomState::new()",
            "pub fn resolve(",
            "entries: Vec<JpegPlanCacheEntry>",
            "hash_one(input)",
            "entry.digest == digest && entry.plan.input().as_bytes() == input",
            "min_by_key(|(index, entry)| (entry.last_used, *index))",
        ]),
        PatternCheck::new("neutral JPEG plan cache optional outcomes", &combined).required(&[
            "SkippedDisabled",
            "SkippedOversized",
            "metadata_allocation_failures",
            "oversized_rejections",
            "disabled_rejections",
            "evictions",
            "peak_bytes",
            "peak_entries",
            "metadata_capacity_bytes",
        ]),
        PatternCheck::new("neutral JPEG plan metadata allocation", &state).required(&[
            "try_reserve_exact(self.entry_limit)",
            "metadata_bytes(entries.capacity())",
            "size_of::<JpegPlanCacheEntry>()",
            "JpegPlanCacheError::allocation(\n                \"JPEG plan cache entry metadata\"",
        ]),
        PatternCheck::new("neutral JPEG plan cache container exclusions", &combined).forbidden(&[
            "VecDeque",
            "HashMap",
            "#[allow(",
            "#[expect(",
        ]),
    ]);

    assert!(
        contains_normalized(
            &state,
            "if !fits_within(requested_metadata, retained_bytes, self.host_byte_limit) { return Ok(self.reject_oversized()); } if !self.ensure_metadata_capacity()?",
        ),
        "oversized admission must be rejected before metadata allocation"
    );
    assert!(
        contains_normalized(
            &state,
            "self.evict_until_fits(retained_bytes, None)?; let stamp = self.next_access_stamp(); self.entries.push",
        ),
        "deterministic eviction must complete before a new plan is committed"
    );
    FunctionCalls::parse(
        "neutral JPEG borrowed-input cache miss",
        &resolve_miss,
        "resolve_miss_with_decoder",
    )
    .assert_ordered(
        "resolve must preflight, parse, copy, build, and admit in one typed sequence",
        &[
            "prepare_for_miss",
            "operation_live_bytes",
            "JpegView::parse_with_external_live",
            "retained_allocation_bytes",
            "checked_live_bytes",
            "SharedJpegInput::try_copy_from_slice_with_external_live_and_cap",
            "JpegCachedPlan::build_shared_from_view_with_decoder",
            "insert",
        ],
    );
    assert!(resolve.contains("pub fn resolve_shared("));
}

#[test]
fn jpeg_plan_cache_owns_fallible_shared_input_and_exact_packet_capacities() {
    let input = read("shared_input.rs");
    let construction = [
        read("shared_input/construction.rs"),
        read("shared_input/construction/copied.rs"),
        read("shared_input/construction/adopted.rs"),
    ]
    .join("\n");
    let copied_construction = read("shared_input/construction/copied.rs");
    let input_accounting = read("shared_input/accounting.rs");
    let packet = format!("{}\n{}", read("packet.rs"), read("packet/accounting.rs"));
    let allocation = read("shared_allocation.rs");
    let combined = format!("{input}\n{construction}\n{input_accounting}\n{packet}\n{allocation}");

    assert_pattern_checks(&[
        PatternCheck::new("fallible shared JPEG input", &combined).required(&[
            "enum SharedJpegInputStorage",
            "Copied(Arc<SharedJpegInputInner>)",
            "ArcSlice(Arc<[u8]>)",
            "DEFAULT_MAX_HOST_ALLOCATION_BYTES",
            "try_copy_from_slice_with_cap",
            "pub fn try_from_arc(input: Arc<[u8]>)",
            "try_from_arc_with_cap",
            "JpegPlanCacheError::Limit",
            "let mut bytes = Vec::new()",
            "bytes.try_reserve_exact(input.len())",
            "pub fn as_bytes(&self) -> &[u8]",
            "impl AsRef<[u8]> for SharedJpegInput",
            "shared_owner_bytes::<SharedJpegInputInner>(input.bytes.capacity())",
            "shared_slice_owner_bytes(input.len())",
            "Arc::ptr_eq(left, right)",
            "Stable Rust exposes neither fallible Arc",
            "allocator usable-size",
        ]),
        PatternCheck::new("one-family shared JPEG packet", &packet).required(&[
            "pub enum JpegFastPacket",
            "Fast420(JpegFast420PacketV1)",
            "Fast422(JpegFast422PacketV1)",
            "Fast444(JpegFast444PacketV1)",
            "SharedJpegFastPacket(Arc<JpegFastPacket>)",
            "self.0.nested_capacity_bytes()?",
            "packet.restart_offsets",
            "packet.entropy_checkpoints",
            "packet.entropy_bytes",
        ]),
        PatternCheck::new("shared JPEG ownership exclusions", &combined).forbidden(&[
            "Arc::from(",
            ".ok()",
            "#[allow(",
            "#[expect(",
        ]),
    ]);
    assert_eq!(
        packet
            .matches("Self::Fast420(packet) => color_packet_capacity_bytes(")
            .count(),
        1
    );
    FunctionCalls::parse(
        "shared JPEG copied-input construction",
        &copied_construction,
        "try_copy_from_slice_with_external_live_and_cap",
    )
    .assert_ordered(
        "shared JPEG input must preflight before allocator entry and postcheck actual capacity",
        &[
            "checked_live_bytes",
            "Vec::new",
            "try_reserve_exact",
            "checked_live_bytes",
            "extend_from_slice",
        ],
    );
    assert_eq!(
        packet
            .matches("Self::Fast422(packet) => color_packet_capacity_bytes(")
            .count(),
        1
    );
    assert_eq!(
        packet
            .matches("Self::Fast444(packet) => color_packet_capacity_bytes(")
            .count(),
        1
    );
}

#[test]
fn jpeg_cached_plan_builds_once_with_canonical_key_semantics_and_typed_errors() {
    let build_facade = read("build.rs");
    let build_plan = read("build/plan.rs");
    let build_assemble = read("build/assemble.rs");
    let build_packet = read("build/packet.rs");
    let build_shared_input = read("build/shared_input.rs");
    let build = [
        build_facade.as_str(),
        build_plan.as_str(),
        build_assemble.as_str(),
        build_packet.as_str(),
        build_shared_input.as_str(),
    ]
    .join("\n");
    let plan = read("plan.rs");
    let packet_build = read_path("crates/j2k-jpeg/src/adapter/fast_packet/build.rs");
    let packet_allocation = read_path("crates/j2k-jpeg/src/adapter/fast_packet/allocation.rs");
    let packet_allocation_tests =
        read_path("crates/j2k-jpeg/src/adapter/fast_packet/tests/allocation.rs");
    let cache = read_path("crates/j2k-jpeg/src/adapter/fast_packet/cache.rs");
    let adapter = read_path("crates/j2k-jpeg/src/adapter/mod.rs");

    assert!(build_facade.contains("mod assemble;"));
    assert!(build_facade.contains("mod shared_input;"));
    assert_pattern_checks(&[
        PatternCheck::new("canonical inspect-once JPEG plan", &build)
            .required(&[
                "const JPEG_PLAN_CACHE_CADENCE_MCUS: u32 = 4",
                "pub fn build(input: SharedJpegInput)",
                "pub fn build_from_view_with_decoder(",
                "build_shared_from_view_with_decoder(",
                "summarize_device_batch(decoder, JPEG_PLAN_CACHE_CADENCE_MCUS)",
                "build_selected_packet(",
                "owner_live_bytes",
                "error.is_capability_mismatch()",
                "clear_fast_families(&mut summary)",
                "pub fn decoder_with_external_live(",
                "pub fn decode_request_with_external_live(",
            ])
            .forbidden(&["cadence_mcus:", "#[allow(", "#[expect("]),
        PatternCheck::new("explicit JPEG packet state invariant", &plan).required(&[
            "JpegFastPacketState::Unsupported if family_count != 0",
            "JpegFastPacketState::Ready(packet)",
            "if family_count != 1 || !matches",
        ]),
        PatternCheck::new("typed JPEG cached-plan build errors", &cache).required(&[
            "Decode(#[from] JpegError)",
            "FastPacket(#[from] FastPacketError)",
            "Cache(#[from] JpegPlanCacheError)",
            "#[derive(Debug, Clone, thiserror::Error)]",
        ]),
        PatternCheck::new("adapter-neutral JPEG cache exports", &adapter).required(&[
            "JpegCachedPlan",
            "JpegCachedPlanBuildError",
            "JpegPlanCacheDiagnostics",
            "SharedJpegFastPacket",
            "SharedJpegInput",
            "DEFAULT_JPEG_PLAN_CACHE_HOST_BYTES",
        ]),
        PatternCheck::new("cached packet external-live baseline", &packet_build).required(&[
            "build_color_fast_packet_from_decoder(bytes, &decoder, header, 0)",
            "external_live_bytes: usize",
            "checked_color_packet_initial_live_bytes(",
            "let mut live_bytes = initial_live_bytes",
        ]),
        PatternCheck::new(
            "external input plus decoder checked helper",
            &packet_allocation,
        )
        .required(&["pub(super) fn checked_color_packet_initial_live_bytes("]),
        PatternCheck::new(
            "external-live exact boundary regression",
            &packet_allocation_tests,
        )
        .required(&["cached_packet_external_input_and_decoder_share_one_exact_initial_boundary"]),
    ]);
}

#[test]
fn jpeg_plan_cache_source_and_contract_tests_stay_focused() {
    for (relative, maximum) in [
        ("../cache.rs", 100),
        ("build.rs", 30),
        ("build/assemble.rs", 100),
        ("build/packet.rs", 130),
        ("build/plan.rs", 100),
        ("build/shared_input.rs", 80),
        ("packet.rs", 220),
        ("packet/accounting.rs", 50),
        ("plan.rs", 140),
        ("shared_allocation.rs", 50),
        ("shared_input.rs", 100),
        ("shared_input/accounting.rs", 80),
        ("shared_input/construction.rs", 30),
        ("shared_input/construction/adopted.rs", 90),
        ("shared_input/construction/copied.rs", 100),
        ("store.rs", 130),
        ("store/diagnostics.rs", 80),
        ("store/resolve.rs", 30),
        ("store/resolve/accounting.rs", 80),
        ("store/resolve/borrowed.rs", 90),
        ("store/resolve/existing_decoder.rs", 90),
        ("store/resolve/miss.rs", 110),
        ("store/resolve/shared.rs", 100),
        ("store/state.rs", 220),
        ("store/state/admission.rs", 220),
    ] {
        let source = read(relative);
        assert!(
            source.lines().count() < maximum,
            "{CACHE_ROOT}/{relative} exceeds its focused line ratchet"
        );
    }

    let tests = [
        read("tests/build.rs"),
        read("tests/ownership.rs"),
        read("tests/resolve.rs"),
        read("tests/store.rs"),
    ]
    .join("\n");
    assert_pattern_checks(&[
        PatternCheck::new("JPEG plan cache contract regressions", &tests).required(&[
            "reused_source_pointer_with_new_bytes_cannot_cross_hit",
            "forced_digest_collision_still_requires_full_byte_equality",
            "exact_retained_limit_is_cached_and_one_byte_over_is_not_retained",
            "oversized_admission_does_not_evict_or_replace_existing_content",
            "deterministic_lru_evicts_the_oldest_entry_after_a_hit",
            "metadata_reservation_failure_preserves_source_and_diagnostics",
            "disabled_cache_is_a_non_error_without_metadata_allocation",
            "shared_one_family_packets_charge_every_nested_vector_capacity_exactly",
            "shared_input_limit_accepts_exact_bytes_and_rejects_one_over_before_reserve",
            "arc_input_moves_without_payload_copy_and_charges_its_fixed_slice_owner",
            "arc_input_limit_accepts_exact_length_and_rejects_one_over",
            "malformed_decode_and_hard_packet_errors_remain_typed_and_uncached",
            "resolve_hit_reuses_the_authoritative_input_and_packet_owners",
            "resolve_shared_preserves_arc_owner_and_full_equality_hits_existing_owner",
            "resolve_returns_current_plan_when_cache_is_disabled",
            "resolve_returns_current_plan_when_admission_is_one_byte_oversized",
            "resolve_never_caches_decode_or_fast_packet_hard_errors",
            "resolve_preflights_impossible_metadata_before_allocator_entry",
            "prepopulated_cache_and_external_baseline_use_one_checked_operation_ledger",
            "resolve_reports_malformed_input_before_copy_limit_failure",
        ]),
    ]);
}

fn read(relative: &str) -> String {
    read_path(&format!("{CACHE_ROOT}/{relative}"))
}

fn read_path(relative: &str) -> String {
    fs::read_to_string(repo_root().join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"))
}
