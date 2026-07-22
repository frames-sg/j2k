// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed native-encode allocation and phase-ownership ratchets.

use std::fs;

use super::{assert_pattern_checks, read_source_files, repo_root, PatternCheck};

mod precomputed97;
mod standard_multitile;
mod standard_plan;
mod tier1_rate_control;

fn read(relative: &str) -> String {
    fs::read_to_string(repo_root().join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"))
}

fn production_source(relative: &str) -> String {
    read(relative)
        .split("\n#[cfg(test)]\nmod tests")
        .next()
        .expect("production prefix")
        .to_owned()
}

fn assert_does_not_derive_clone(source: &str, declaration: &str) {
    let declaration_offset = source
        .find(declaration)
        .unwrap_or_else(|| panic!("missing public owner declaration `{declaration}`"));
    let prefix = &source[..declaration_offset];
    let derive_offset = prefix
        .rfind("#[derive(")
        .unwrap_or_else(|| panic!("missing derive for public owner `{declaration}`"));
    let derive = prefix[derive_offset..].lines().next().expect("derive line");
    assert!(
        !derive.contains("Clone"),
        "{declaration} must remain move-only; found `{derive}`"
    );
}

#[test]
fn large_encode_owner_graphs_remain_move_only() {
    let types = read("crates/j2k-types/src/lib.rs");
    for declaration in [
        "pub struct EncodedJ2kCodeBlock",
        "pub struct EncodedHtJ2kCodeBlock",
        "pub struct J2kPacketizationSubband",
        "pub struct J2kPacketizationResolution",
        "pub struct J2kForwardDwt53Output",
        "pub struct J2kForwardDwt53Level",
        "pub struct J2kForwardDwt97Output",
        "pub struct J2kForwardDwt97Level",
        "pub struct PrecomputedHtj2k53Component",
        "pub struct PrecomputedHtj2k53Image",
        "pub struct PrecomputedHtj2k97Component",
        "pub struct PrecomputedHtj2k97Image",
        "pub struct PrequantizedHtj2k97Image",
        "pub struct PrequantizedHtj2k97Component",
        "pub struct PrequantizedHtj2k97Resolution",
        "pub struct PrequantizedHtj2k97Subband",
        "pub struct PrequantizedHtj2k97CodeBlock",
        "pub struct PreencodedHtj2k97Image",
        "pub struct PreencodedHtj2k97Component",
        "pub struct PreencodedHtj2k97Resolution",
        "pub struct PreencodedHtj2k97Subband",
        "pub struct PreencodedHtj2k97CodeBlock",
        "pub struct PreencodedHtj2k97CompactImage",
        "pub struct PreencodedHtj2k97CompactComponent",
        "pub struct PreencodedHtj2k97CompactResolution",
        "pub struct PreencodedHtj2k97CompactSubband",
        "pub struct PreencodedHtj2k97CompactCodeBlock",
    ] {
        assert_does_not_derive_clone(&types, declaration);
    }

    let transcode_contracts = read("crates/j2k-transcode/src/accelerator_contracts.rs");
    for declaration in [
        "pub struct PreencodedHtj2k97CompactBatch",
        "pub struct PreencodedHtj2k97CompactBatchGroups",
    ] {
        assert_does_not_derive_clone(&transcode_contracts, declaration);
    }

    let transcode = read("crates/j2k-transcode/src/lib.rs");
    for declaration in [
        "pub struct Dwt53TwoDimensional",
        "pub struct Dwt97TwoDimensional",
        "pub struct ReversibleDwt53FirstLevel",
    ] {
        assert_does_not_derive_clone(&transcode, declaration);
    }

    let transcode_outputs = read("crates/j2k-transcode/src/jpeg_to_htj2k/output.rs");
    assert_does_not_derive_clone(&transcode_outputs, "pub struct EncodedTranscode");

    let transcode_test_support = read("crates/j2k-transcode-test-support/src/dct53_multilevel.rs");
    assert_does_not_derive_clone(&transcode_test_support, "pub struct Dwt53MultiLevel");

    let native_params = read("crates/j2k-native/src/j2c/codestream_write.rs");
    assert_does_not_derive_clone(&native_params, "pub(crate) struct EncodeParams");

    let native_roi = read("crates/j2k-native/src/j2c/encode/roi_plan.rs");
    assert_does_not_derive_clone(&native_roi, "pub(super) struct ComponentRoiEncodePlan");
}

#[test]
fn native_encode_errors_preserve_resource_and_stage_categories() {
    let error = read("crates/j2k-native/src/error.rs");
    let facade = read("crates/j2k-native/src/lib.rs");
    let pipeline = read("crates/j2k-native/src/j2c/encode/retained_input.rs");
    assert_pattern_checks(&[
        PatternCheck::new("native encode error contract", &error).required(&[
            "pub enum EncodeError",
            "InvalidInput",
            "Unsupported",
            "ArithmeticOverflow",
            "AllocationTooLarge",
            "HostAllocationFailed",
            "Accelerator",
            "CodestreamValidation",
            "InternalInvariant",
            "pub type EncodeResult<T>",
        ]),
        PatternCheck::new("native encode error export", &facade)
            .required(&["EncodeError, EncodeResult"]),
        PatternCheck::new("native encode pipeline categories", &pipeline)
            .required(&[
                "pub(crate) enum NativeEncodePipelineError",
                "pub(crate) const fn invalid_input(",
                "pub(crate) const fn unsupported(",
                "pub(crate) const fn arithmetic_overflow(",
                "pub(crate) const fn internal_invariant(",
            ])
            .forbidden(&[
                "impl From<&'static str> for NativeEncodePipelineError",
                "Self::InternalInvariant(detail)",
            ]),
    ]);
}

#[test]
fn native_encode_error_boundary_regression_stays_present() {
    let error = read("crates/j2k-native/src/error.rs");
    assert_pattern_checks(
        &[PatternCheck::new("native encode error regression", &error)
            .required(&["encode_resource_errors_keep_cap_and_allocator_failures_distinct"])],
    );
}

#[test]
fn native_transform_stage_keeps_focused_real_module_boundaries() {
    const MODULES: &[(&str, usize)] = &[
        ("crates/j2k-native/src/j2c/encode/transform.rs", 40),
        (
            "crates/j2k-native/src/j2c/encode/transform/accelerated_dwt.rs",
            340,
        ),
        (
            "crates/j2k-native/src/j2c/encode/transform/component_samples.rs",
            150,
        ),
        (
            "crates/j2k-native/src/j2c/encode/transform/dwt53_output.rs",
            120,
        ),
        ("crates/j2k-native/src/j2c/encode/transform/mct.rs", 50),
        (
            "crates/j2k-native/src/j2c/encode/transform/reversible.rs",
            80,
        ),
    ];
    for (relative, ceiling) in MODULES {
        let source = read(relative);
        let lines = source.lines().count();
        assert!(
            lines <= *ceiling,
            "{relative} has {lines} lines; transform module ceiling is {ceiling}"
        );
    }

    let facade = read("crates/j2k-native/src/j2c/encode/transform.rs");
    let accelerated =
        production_source("crates/j2k-native/src/j2c/encode/transform/accelerated_dwt.rs");
    let components =
        production_source("crates/j2k-native/src/j2c/encode/transform/component_samples.rs");
    let typed_output =
        production_source("crates/j2k-native/src/j2c/encode/transform/dwt53_output.rs");
    let color = production_source("crates/j2k-native/src/j2c/encode/transform/mct.rs");
    let reversible = production_source("crates/j2k-native/src/j2c/encode/transform/reversible.rs");

    assert_pattern_checks(&[
        PatternCheck::new("native transform module wiring", &facade)
            .required(&[
                "mod accelerated_dwt;",
                "mod component_samples;",
                "mod dwt53_output;",
                "mod mct;",
                "mod reversible;",
                "pub(super) use accelerated_dwt::{",
                "pub(super) use component_samples::{",
                "pub(super) use dwt53_output::try_forward_dwt53_output_from_decomposition;",
                "pub(super) use mct::{try_encode_forward_ict, try_encode_forward_rct};",
                "pub(super) use reversible::{",
            ])
            .forbidden(&[
                "#[path =",
                "fn encode_forward_dwt(",
                "fn try_component_plane_to_f32_for_session(",
                "fn reversible_guard_bits_for_marker_limit(",
            ]),
        PatternCheck::new("accelerated DWT responsibility", &accelerated).required(&[
            "struct ForwardDwtRequest",
            "trait AcceleratedDwtLevel",
            "fn convert_accelerated_dwt_output",
            "fn accelerated_dwt_output_retained_bytes",
            "fn validate_band_len",
        ]),
        PatternCheck::new("component sample responsibility", &components).required(&[
            "fn validate_deinterleaved_components",
            "fn try_component_plane_to_f32_for_session",
            "fn validate_component_sample_info",
        ]),
        PatternCheck::new("typed 5/3 output responsibility", &typed_output).required(&[
            "fn try_forward_dwt53_output_from_decomposition",
            "typed component DWT conversion overlap",
        ]),
        PatternCheck::new("MCT accelerator responsibility", &color).required(&[
            "fn try_encode_forward_rct",
            "fn try_encode_forward_ict",
            "J2kForwardRctJob",
            "J2kForwardIctJob",
        ]),
        PatternCheck::new("reversible marker responsibility", &reversible).required(&[
            "fn reversible_guard_bits_for_marker_limit",
            "fn adjust_component_step_sizes_for_guard_delta",
            "fn adjust_reversible_step_sizes_for_guard_delta",
        ]),
    ]);
}

#[test]
fn native_packet_encoder_keeps_real_module_boundaries_and_line_ratchets() {
    const MODULES: &[(&str, usize)] = &[
        ("crates/j2k-native/src/j2c/packet_encode.rs", 140),
        (
            "crates/j2k-native/src/j2c/packet_encode/accelerator_ownership.rs",
            90,
        ),
        ("crates/j2k-native/src/j2c/packet_encode/form.rs", 560),
        ("crates/j2k-native/src/j2c/packet_encode/header.rs", 520),
        ("crates/j2k-native/src/j2c/packet_encode/ownership.rs", 140),
        ("crates/j2k-native/src/j2c/packet_encode/state.rs", 310),
        ("crates/j2k-native/src/j2c/packet_encode/view.rs", 220),
    ];
    for (relative, ceiling) in MODULES {
        let source = read(relative);
        let lines = source.lines().count();
        assert!(
            lines <= *ceiling,
            "{relative} has {lines} lines; packet module ceiling is {ceiling}"
        );
    }

    let facade = read("crates/j2k-native/src/j2c/packet_encode.rs");
    assert_pattern_checks(&[PatternCheck::new("native packet module wiring", &facade)
        .required(&[
            "mod accelerator_ownership;",
            "mod form;",
            "mod header;",
            "mod ownership;",
            "mod state;",
            "mod view;",
            "#[cfg(test)]\nmod tests;",
        ])
        .forbidden(&["#[path =", "struct FormedPacket", "merged: Vec<u8>"])]);
}

#[test]
fn native_packet_owners_share_typed_checked_allocation_contract() {
    let root = repo_root();
    let packet = read_source_files(
        root,
        &[
            "crates/j2k-native/src/j2c/packet_encode.rs",
            "crates/j2k-native/src/j2c/packet_encode/accelerator_ownership.rs",
            "crates/j2k-native/src/j2c/packet_encode/form.rs",
            "crates/j2k-native/src/j2c/packet_encode/header.rs",
            "crates/j2k-native/src/j2c/packet_encode/ownership.rs",
            "crates/j2k-native/src/j2c/packet_encode/state.rs",
            "crates/j2k-native/src/j2c/packet_encode/view.rs",
        ],
    );
    let allocation = production_source("crates/j2k-native/src/j2c/encode/allocation.rs");
    let writer = production_source("crates/j2k-native/src/writer.rs");
    let tag_tree = production_source("crates/j2k-native/src/j2c/tag_tree_encode.rs");
    let scalar_source = production_source("crates/j2k-native/src/scalar/encode.rs");
    let scalar = scalar_source
        .split_once("pub fn encode_j2k_packetization_scalar(")
        .map(|(_, body)| {
            let mut function = String::from("pub fn encode_j2k_packetization_scalar(");
            function.push_str(body);
            function
        })
        .expect("scalar packetization function must remain present");

    assert_pattern_checks(&[
        PatternCheck::new("typed packet ownership", &packet)
            .required(&[
                "form_tile_bitstream_with_public_descriptors_and_retained_baseline",
                "owned_packet_retained_bytes_for_public_descriptors",
                "borrowed_scalar_retained_bytes",
                "packet_metadata_retained_bytes",
                "packetized_tile_retained_bytes",
                "EncodeAllocationLedger::new(retained_baseline_bytes)?",
                "BudgetedHeaderStore",
                "payload_claim: EncodeAllocationClaim",
                "code_blocks: BudgetedVec<'a, PacketCodeBlockState>",
                "let mut data =",
                ".try_vec_with_capacity(plan.tile_len",
                "allocations.seal()?",
                "allocations.finalize()?",
                "use explicit packet descriptors for multidimensional packetization",
            ])
            .forbidden(&[
                "Vec::with_capacity(",
                "vec![",
                ".to_vec(",
                ".collect::<",
                ".collect()",
                "BitWriter::new(",
                "struct FormedPacket",
                "merged: Vec<u8>",
                "body: Vec<u8>",
                "Result<Vec<u8>, &'static str>",
                "Result<PacketizedTileData, &'static str>",
                "unreachable!(",
                ".expect(",
            ]),
        PatternCheck::new("shared encode allocation ledger", &allocation)
            .required(&[
                "DEFAULT_MAX_CODEC_BYTES",
                "const SEALED_BIT",
                "struct EncodeAllocationLedger",
                "struct EncodeAllocationClaim",
                "struct BudgetedVec",
                "try_reserve_exact(count)",
                "claim.reconcile(",
                "compare_exchange_weak(",
                "try_update(",
                "poisoned.store(true",
            ])
            .forbidden(&["Cell<", "saturating_sub(", "512 * 1024 * 1024"]),
        PatternCheck::new("checked packet bit writer", &writer)
            .required(&[
                "struct CheckedBitWriter<'a>",
                "data: BudgetedVec<'a, u8>",
                "impl FallibleBitWriter for CheckedBitWriter<'_>",
                "#[cfg(test)]\nimpl FallibleBitWriter for BitWriter",
            ])
            .forbidden(&["#[derive(Debug, Clone)]\npub(crate) struct BitWriter"]),
        PatternCheck::new("checked tag-tree owner", &tag_tree)
            .required(&[
                "nodes: BudgetedVec<'a, TagNode>",
                "pub(crate) fn try_new(",
                "allocations.try_vec_with_capacity(",
                "let mut path = [0usize; MAX_TAG_TREE_LEVELS]",
            ])
            .forbidden(&["Vec::with_capacity(", "pub(crate) fn new(", "vec!["]),
        PatternCheck::new("borrowed scalar packet fallback", &scalar)
            .required(&[
                "pub fn encode_j2k_packetization_scalar(",
                ") -> crate::EncodeResult<Vec<u8>>",
                "form_borrowed_packetization_scalar(job, 0)",
            ])
            .forbidden(&["code_block.data.to_vec()", ".collect::<Vec<_>>()"]),
    ]);
}

#[test]
fn native_packet_boundary_and_parity_regressions_stay_present() {
    let coverage = read_source_files(
        repo_root(),
        &[
            "crates/j2k-native/src/j2c/encode/allocation.rs",
            "crates/j2k-native/src/j2c/packet_encode/tests.rs",
            "crates/j2k-native/src/j2c/tag_tree_encode.rs",
        ],
    );
    assert_pattern_checks(&[
        PatternCheck::new("native packet regressions", &coverage).required(&[
            "exact_cap_claim_is_accepted_and_released",
            "one_byte_over_cap_is_rejected_without_changing_live_bytes",
            "element_byte_overflow_is_typed",
            "allocator_failure_mapping_is_typed_and_source_specific",
            "stale_preseal_observation_cannot_commit_after_concurrent_seal",
            "classic_pass_count_boundary_headers_are_bit_exact",
            "implicit_single_layer_component_progressions_preserve_packet_bytes",
            "implicit_progression_rejects_multidimensional_packetization",
            "explicit_packet_descriptors_reject_sparse_max_state_before_allocation",
        ]),
    ]);
}

#[test]
fn native_precinct_split_modules_stay_under_focused_ceilings() {
    const MODULES: &[(&str, usize)] = &[
        ("crates/j2k-native/src/j2c/encode/packet_plan.rs", 350),
        (
            "crates/j2k-native/src/j2c/encode/packet_plan/precinct.rs",
            360,
        ),
        (
            "crates/j2k-native/src/j2c/encode/packet_plan/precinct/geometry.rs",
            210,
        ),
        (
            "crates/j2k-native/src/j2c/encode/packet_plan/precinct/geometry/block_mapping.rs",
            170,
        ),
        (
            "crates/j2k-native/src/j2c/encode/packet_plan/precinct/distribution.rs",
            230,
        ),
        (
            "crates/j2k-native/src/j2c/encode/packet_plan/precinct/ownership.rs",
            250,
        ),
        (
            "crates/j2k-native/src/j2c/encode/packet_plan/precinct/tests.rs",
            220,
        ),
    ];
    for (relative, ceiling) in MODULES {
        let source = read(relative);
        let lines = source.lines().count();
        assert!(
            lines <= *ceiling,
            "{relative} has {lines} lines; precinct ownership ceiling is {ceiling}"
        );
    }
}

#[test]
fn native_precinct_split_moves_one_graph_under_one_actual_capacity_plan() {
    let facade = production_source("crates/j2k-native/src/j2c/encode/packet_plan.rs");
    let split = read_source_files(
        repo_root(),
        &[
            "crates/j2k-native/src/j2c/encode/packet_plan/precinct.rs",
            "crates/j2k-native/src/j2c/encode/packet_plan/precinct/distribution.rs",
        ],
    );
    let geometry = read_source_files(
        repo_root(),
        &[
            "crates/j2k-native/src/j2c/encode/packet_plan/precinct/geometry.rs",
            "crates/j2k-native/src/j2c/encode/packet_plan/precinct/geometry/block_mapping.rs",
        ],
    );
    let ownership =
        production_source("crates/j2k-native/src/j2c/encode/packet_plan/precinct/ownership.rs");
    let encode = production_source("crates/j2k-native/src/j2c/encode.rs");
    let call_sites = read_source_files(
        repo_root(),
        &[
            "crates/j2k-native/src/j2c/encode/i64_packetize.rs",
            "crates/j2k-native/src/j2c/encode/single_tile/tile_encode.rs",
        ],
    );
    let coverage = read("crates/j2k-native/src/j2c/encode/packet_plan/precinct/tests.rs");

    assert_pattern_checks(&[
        PatternCheck::new("precinct ownership module boundary", &facade).required(&[
            "mod precinct;",
            "split_component_resolution_packets_by_precinct_for_session",
        ]),
        PatternCheck::new("move-only prepared precinct distribution", &split)
            .required(&[
                "for (block_index, block) in code_blocks.into_iter().enumerate()",
                "fn move_preencoded_blocks_to_precincts(",
                "try_destination_vec(",
                "release_source_capacity::<PreparedEncodeCodeBlock>",
                "release_source_capacity::<crate::EncodedHtJ2kCodeBlock>",
                "precinct code-block grid count mismatch",
            ])
            .forbidden(&[
                ".cloned()",
                ".clone()",
                "Vec::with_capacity(",
                ".collect::<",
                ".collect()",
            ]),
        PatternCheck::new("checked precinct geometry", &geometry).required(&[
            "precinct dimensions must not reduce encoder code-block dimensions",
            "checked_mul(",
            "checked_add(",
            "precinct_index_for_block(",
        ]),
        PatternCheck::new("actual-capacity precinct high-water", &ownership)
            .required(&[
                "source_structural_bytes",
                "destination_structural_bytes",
                "payload_bytes",
                "session.checked_phase(",
                "try_reserve_exact(count)",
                "values.capacity()",
                "PreparedCodeBlockCoefficients::I32(values)",
                "PreparedCodeBlockCoefficients::I64(values)",
                "values.capacity()",
                "block.data.capacity()",
                "prepared_tree_ownership(output, output_capacity)",
            ])
            .forbidden(&["Vec::with_capacity(", "saturating_", ".clone()"]),
        PatternCheck::new("prepared Tier-1 owners are move-only", &encode).forbidden(&[
            "#[derive(Clone)]\nstruct PreparedEncodeCodeBlock",
            "#[derive(Clone)]\nstruct PreparedEncodeSubband",
        ]),
        PatternCheck::new("typed precinct split call sites", &call_sites).required(&[
            "split_component_resolution_packets_by_precinct_for_session(",
            "session,",
            "retained_base_bytes,",
        ]),
        PatternCheck::new("precinct split behavior and cap regressions", &coverage).required(&[
            "precinct_split_moves_tier1_payloads_without_clone",
            "precinct_split_exact_peak_accepts_cap_and_rejects_one_byte_less",
            "assert_eq!(moved_coefficient_ptrs, coefficient_ptrs)",
            "assert_eq!(moved_preencoded_ptrs, preencoded_ptrs)",
        ]),
    ]);
}

#[test]
fn native_encode_retained_inputs_are_typed_lifetime_bound_and_phase_seeded() {
    const MODULES: &[(&str, usize)] = &[
        ("crates/j2k-native/src/j2c/encode/retained_input.rs", 230),
        (
            "crates/j2k-native/src/j2c/encode/retained_input/child_session.rs",
            40,
        ),
        (
            "crates/j2k-native/src/j2c/encode/retained_input/tests.rs",
            130,
        ),
        ("crates/j2k-native/src/j2c/encode/retained_api.rs", 80),
        ("crates/j2k-native/src/color/allocation.rs", 60),
        ("crates/j2k-native/src/j2c/recode/tests.rs", 100),
    ];
    for (relative, ceiling) in MODULES {
        let source = read(relative);
        let lines = source.lines().count();
        assert!(
            lines <= *ceiling,
            "{relative} has {lines} lines; retained-input module ceiling is {ceiling}"
        );
    }

    let retained = read_source_files(
        repo_root(),
        &[
            "crates/j2k-native/src/j2c/encode/retained_input.rs",
            "crates/j2k-native/src/j2c/encode/retained_input/child_session.rs",
        ],
    );
    let owners = read_source_files(
        repo_root(),
        &[
            "crates/j2k-native/src/color/allocation.rs",
            "crates/j2k-native/src/j2c/recode.rs",
            "crates/j2k-native/src/j2c/encode/precomputed/api53.rs",
            "crates/j2k/src/recode/coefficient.rs",
        ],
    );
    let entrypoints = read_source_files(
        repo_root(),
        &[
            "crates/j2k-native/src/j2c/encode.rs",
            "crates/j2k-native/src/j2c/encode/retained_api.rs",
            "crates/j2k-native/src/j2c/encode/precomputed/api53.rs",
            "crates/j2k-native/src/j2c/encode/single_tile.rs",
            "crates/j2k-native/src/j2c/encode/single_tile/tile_encode.rs",
            "crates/j2k-native/src/j2c/encode/i64_packetize.rs",
            "crates/j2k-native/src/j2c/encode/packet_plan.rs",
        ],
    );
    let facade = read("crates/j2k-native/src/lib.rs");

    assert_pattern_checks(&[
        PatternCheck::new("lifetime-bound retained encode token", &retained)
            .required(&[
                "pub(crate) struct NativeEncodeRetainedInput<'a>",
                "owners: PhantomData<&'a ()>",
                "pub(crate) fn from_owner_bytes<O: ?Sized>",
                "pub(crate) struct NativeEncodeSession<'a>",
                "EncodeAllocationLedger::with_cap(retained_input.bytes(), cap)?",
                "checked_phase_retained_bytes",
                "checked_child_session",
            ])
            .forbidden(&[
                "#[derive(Clone",
                "#[derive(Copy",
                "impl Clone for NativeEncodeRetainedInput",
                "impl Copy for NativeEncodeRetainedInput",
                "pub fn from_owner_bytes",
                "pub fn from_bytes",
                "pub fn bytes(",
            ]),
        PatternCheck::new("specialized retained encode owners", &owners)
            .required(&[
                "impl RawBitmap",
                "pub(crate) fn allocated_bytes(&self) -> Option<usize>",
                "bit_capacity_bytes(self.component_signed.capacity())?",
                "impl Reversible53CoefficientImage",
                "pub fn encode_htj2k(&self, options: &EncodeOptions)",
                "let retained_bytes = self.checked_retained_capacity_bytes()?",
                "encode_precomputed_htj2k_53_with_mct_and_retained_owner",
                "NativeEncodeRetainedInput::from_owner_bytes(owner, retained_bytes)",
                ".encode_htj2k(&encode_options)",
            ])
            .forbidden(&["try_from_vec(", "try_include_vec("]),
        PatternCheck::new("retained baseline entrypoint propagation", &entrypoints)
            .required(&[
                "pub(crate) fn encode_with_accelerator_and_retained_input",
                "NativeEncodeRetainedInput::none()",
                "session: &NativeEncodeSession<'_>",
                "packetize_resolution_packets_with_options_for_session",
                "retained native encode inputs and packet ownership",
            ])
            .forbidden(&["pub fn encode_with_accelerator_and_retained_input"]),
        PatternCheck::new("retained encode test-only root token", &facade).required(&[
            "#[cfg(test)]",
            "pub(crate) use j2c::encode::NativeEncodeRetainedInput;",
            "NativeEncodeRetainedInput",
        ]),
    ]);
}

#[test]
fn native_encode_retained_input_boundary_regressions_stay_present() {
    let retained = read_source_files(
        repo_root(),
        &[
            "crates/j2k-native/src/j2c/encode/retained_input.rs",
            "crates/j2k-native/src/j2c/encode/retained_input/tests.rs",
            "crates/j2k-native/src/color/allocation.rs",
            "crates/j2k-native/src/j2c/recode/tests.rs",
        ],
    );
    assert_pattern_checks(&[
        PatternCheck::new("retained encode boundary regressions", &retained).required(&[
            "retained_baseline_accepts_exact_cap_and_rejects_one_byte_over",
            "checked_phase_baseline_includes_every_retained_owner",
            "packed_bitmap_counts_bit_vector_capacity_in_bytes",
            "coefficient_tree_baseline_accepts_exact_cap_and_rejects_one_byte_over",
            "accelerator_output_phase_accepts_exact_cap_and_rejects_one_byte_over",
            "component_signed.capacity().div_ceil(8)",
        ]),
    ]);
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "accelerator ownership checks remain one fail-closed source and regression matrix"
)]
fn native_accelerator_outputs_reconcile_before_codestream_finalization() {
    const MODULES: &[(&str, usize)] = &[
        (
            "crates/j2k-native/src/j2c/encode/single_tile/ownership.rs",
            110,
        ),
        (
            "crates/j2k-native/src/j2c/encode/single_tile/ownership/params.rs",
            70,
        ),
        (
            "crates/j2k-native/src/j2c/encode/precomputed/compact97.rs",
            500,
        ),
        (
            "crates/j2k-native/src/j2c/encode/precomputed/compact97/tests.rs",
            320,
        ),
        (
            "crates/j2k-native/src/j2c/encode/precomputed/compact97/accelerator_tests.rs",
            190,
        ),
        (
            "crates/j2k-native/src/j2c/encode/precomputed/compact97/construction.rs",
            420,
        ),
        (
            "crates/j2k-native/src/j2c/codestream_write/accounting.rs",
            140,
        ),
        (
            "crates/j2k-native/src/j2c/codestream_write/accounting/header.rs",
            125,
        ),
        (
            "crates/j2k-native/src/j2c/codestream_write/accounting/tests.rs",
            150,
        ),
        (
            "crates/j2k-native/src/j2c/codestream_write/packet_markers.rs",
            400,
        ),
        (
            "crates/j2k-native/src/j2c/encode/multitile/finalize/tests.rs",
            130,
        ),
        ("crates/j2k-native/src/j2c/encode/packet_plan/tests.rs", 40),
        (
            "crates/j2k-native/src/j2c/encode/packet_plan/tests/accelerator_ownership.rs",
            220,
        ),
        ("crates/j2k-native/src/j2c/encode/single_tile/tests.rs", 320),
        (
            "crates/j2k-native/src/j2c/encode/single_tile/tests/whole_tile.rs",
            300,
        ),
        (
            "crates/j2k-native/src/j2c/encode/single_tile/tests/resident_errors.rs",
            80,
        ),
    ];
    for (relative, ceiling) in MODULES {
        let source = read(relative);
        let lines = source.lines().count();
        assert!(
            lines <= *ceiling,
            "{relative} has {lines} lines; accelerator ownership ceiling is {ceiling}"
        );
    }

    let session = production_source("crates/j2k-native/src/j2c/encode/retained_input.rs");
    let packet = production_source("crates/j2k-native/src/j2c/encode/packet_plan.rs");
    let packet_owners =
        production_source("crates/j2k-native/src/j2c/packet_encode/accelerator_ownership.rs");
    let compact = production_source("crates/j2k-native/src/j2c/encode/precomputed/compact97.rs");
    let compact_construction =
        production_source("crates/j2k-native/src/j2c/encode/precomputed/compact97/construction.rs");
    let writer = production_source("crates/j2k-native/src/j2c/codestream_write.rs");
    let writer_accounting = read_source_files(
        repo_root(),
        &[
            "crates/j2k-native/src/j2c/codestream_write/accounting.rs",
            "crates/j2k-native/src/j2c/codestream_write/accounting/header.rs",
        ],
    );
    let writer_packet_markers =
        production_source("crates/j2k-native/src/j2c/codestream_write/packet_markers.rs");
    let tile = production_source("crates/j2k-native/src/j2c/encode/single_tile/accelerator.rs");
    let tile_owners = read_source_files(
        repo_root(),
        &[
            "crates/j2k-native/src/j2c/encode/single_tile/ownership.rs",
            "crates/j2k-native/src/j2c/encode/single_tile/ownership/params.rs",
        ],
    );
    let tile_tests = read("crates/j2k-native/src/j2c/encode/single_tile/tests.rs");
    let resident = production_source("crates/j2k-native/src/j2c/encode/resident_contract.rs");
    let resident_route =
        production_source("crates/j2k-native/src/j2c/encode/single_tile/resident.rs");
    let coverage = read_source_files(
        repo_root(),
        &[
            "crates/j2k-native/src/j2c/encode/retained_input.rs",
            "crates/j2k-native/src/j2c/encode/retained_input/tests.rs",
            "crates/j2k-native/src/j2c/encode/packet_plan/tests.rs",
            "crates/j2k-native/src/j2c/encode/packet_plan/tests/accelerator_ownership.rs",
            "crates/j2k-native/src/j2c/encode/precomputed/compact97/tests.rs",
            "crates/j2k-native/src/j2c/encode/precomputed/compact97/accelerator_tests.rs",
            "crates/j2k-native/src/j2c/encode/single_tile/accelerator/tests.rs",
            "crates/j2k-native/src/j2c/encode/single_tile/tests.rs",
            "crates/j2k-native/src/j2c/encode/single_tile/tests/resident_errors.rs",
            "crates/j2k-native/src/j2c/encode/single_tile/tests/whole_tile.rs",
            "crates/j2k-native/src/j2c/codestream_write/accounting/tests.rs",
            "crates/j2k-native/src/j2c/encode/multitile/finalize/tests.rs",
            "crates/j2k-native/src/j2c/encode/tile_parts/tests.rs",
            "crates/j2k-native/tests/multitile_tile_parts.rs",
            "crates/j2k-native/src/j2c/encode_tests.rs",
        ],
    );

    assert_pattern_checks(&[
        PatternCheck::new("checked accelerator output phase", &session).required(&[
            "pub(crate) struct NativeEncodePhase",
            "cap: usize",
            "EncodeAllocationLedger::with_phase_cap(retained_bytes, self.cap, what)?",
            "reconcile_accelerator_output_bytes(",
            "reconcile_accelerator_vec<T>(",
        ]),
        PatternCheck::new("packet accelerator output ownership", &packet).required(&[
            "packet_metadata_retained_bytes(",
            "session.checked_phase(",
            "try_packetization_accelerator(",
            "packetized_tile_retained_bytes(&packetized)",
            "accelerator packetization output",
        ]),
        PatternCheck::new("nested packet accelerator owners", &packet_owners).required(&[
            "resolution_capacity",
            "resolution.subbands.capacity()",
            "subband.code_blocks.capacity()",
            "tile.data.capacity()",
            "tile.packet_lengths.capacity()",
            "tile.packet_headers.capacity()",
            "header.capacity()",
        ]),
        PatternCheck::new("compact preencoded 9/7 retained ownership", &compact).required(&[
            "compact_image_retained_bytes(&image)",
            "NativeEncodeRetainedInput::from_owner_bytes",
            "packet_metadata_retained_bytes(",
            "construction::try_public_packet_metadata(",
            "try_compact_packetization_accelerator(",
            "drop(image);",
            "write_codestream_accounted_with_peak_check(",
            "reconcile_compact_final_codestream(",
        ]),
        PatternCheck::new("compact 9/7 fallible construction", &compact_construction).required(&[
            "ConstructionTracker::new(session)",
            "fn try_prepared_packets<'a>(",
            "fn try_plan_metadata(",
            "try_reserve_exact(count)",
            "host_allocation_failed(what, requested_bytes)",
            "self.session.checked_phase(requested_live, CONSTRUCTION)",
            "self.session.checked_phase(actual_live, CONSTRUCTION)",
            "try_public_packet_metadata<'a>(",
        ]),
        PatternCheck::new(
            "codestream writer preflight before fallible reserve",
            &writer,
        )
        .required(&[
            "write_codestream_accounted_with_peak_check(",
            "check_writer_peak(output_len)?",
            "try_reserve_exact(output_len)",
            "check_writer_peak(output_capacity)?",
            "write_main_header_prefix(",
        ]),
        PatternCheck::new("general codestream writer peak contract", &writer)
            .required(&[
                "write_codestream_tiles_accounted_with_peak_check(",
                "codestream_tiles_output_len(",
                "check_writer_peak(output_len)?",
                "check_writer_peak(output_capacity)?",
                "write_plm_markers(&mut out, tiles)?",
                "write_ppm_markers(&mut out, tiles)?",
                "write_plt_markers(&mut out, tile.packet_lengths)?",
                "write_ppt_markers(&mut out, tile.packet_headers)?",
            ])
            .forbidden(&[
                "fn write_codestream_tiles(",
                "main_header_packet_lengths",
                "main_header_packet_headers",
                "PreparedTilePart",
            ]),
        PatternCheck::new("codestream writer checked exact sizing", &writer_accounting).required(
            &[
                "single_tile_output_len(",
                "codestream_tiles_output_len(",
                "tile_part_len(",
                "checked_element_bytes::<[u8; 3]>(",
                "quantization_marker_bytes(",
                "tile-part length exceeds u32",
            ],
        ),
        PatternCheck::new(
            "allocation-free packet marker streaming",
            &writer_packet_markers,
        )
        .required(&[
            "plt_marker_bytes(",
            "plm_marker_bytes(",
            "ppm_marker_bytes(",
            "ppt_marker_bytes(",
            "write_packet_length_chunks(",
            "struct HeaderCursor",
        ])
        .forbidden(&[
            "Vec::new()",
            "Vec::with_capacity(",
            ".to_vec()",
            "packet_length_bytes(",
            "let mut chunk = Vec",
        ]),
        PatternCheck::new("whole-tile accelerator output ownership", &tile).required(&[
            "single_tile_plan_retained_bytes(plan)?",
            "session.checked_phase(",
            "reconcile_accelerator_vec(",
            "operation: \"whole-tile HTJ2K encode\"",
            "resident_error_from_encode_error",
        ]),
        PatternCheck::new("single-tile plan owner coverage", &tile_owners).required(&[
            "plan.step_sizes.capacity()",
            "plan.component_step_sizes.capacity()",
            "roi.regions.capacity()",
            "params.component_quantization_step_sizes.capacity()",
            "params.precinct_exponents.capacity()",
        ]),
        PatternCheck::new("single-tile test module boundary", &tile_tests)
            .required(&["mod resident_errors;", "mod whole_tile;"])
            .forbidden(&[
                "struct FakeWholeTileAccelerator",
                "fn whole_tile_fixture(",
                "fn resident_whole_tile_over_cap_keeps_resource_category(",
            ]),
        PatternCheck::new("general writer boundary regressions", &coverage).required(&[
            "marker_multitile_writer_accepts_exact_peak_and_rejects_cap_minus_one",
            "marker_multitile_finalizer_counts_exact_writer_peak",
            "borrowed_finalization_accepts_exact_peak_and_rejects_one_byte_over",
            "scratch_free_finalization_is_byte_exact_and_enforces_writer_peak",
            "multi_tile_packet_limit_splits_only_parent_tile_parts_and_round_trips",
        ]),
        PatternCheck::new("resident error categories", &resident).required(&[
            "InvalidInput(&'static str)",
            "Resource(crate::EncodeError)",
            "Backend(crate::EncodeError)",
        ]),
        PatternCheck::new("resident typed error classifier", &resident_route)
            .required(&[
                "crate::EncodeError::InvalidInput { what } => ResidentHtj2kEncodeError::InvalidInput(what)",
                "crate::EncodeError::Unsupported { what } => ResidentHtj2kEncodeError::Unsupported(what)",
                "ResidentHtj2kEncodeError::Resource(error)",
                "ResidentHtj2kEncodeError::Accelerator(source)",
                "ResidentHtj2kEncodeError::Backend(error)",
            ])
            .forbidden(&["map_err(ResidentHtj2kEncodeError::Resource)"]),
        PatternCheck::new("accelerator output boundary regressions", &coverage).required(&[
            "packet_accelerator_output_accepts_exact_cap_without_copying",
            "packet_accelerator_output_rejects_one_byte_over_without_fallback",
            "packet_accelerator_decline_and_failure_keep_distinct_categories",
            "packetized_accelerator_output_counts_nested_metadata_capacities",
            "packet_accelerator_phase_counts_nested_public_metadata_capacities",
            "compact_accelerator_packet_output_accepts_exact_cap_without_copying",
            "compact_accelerator_packet_output_rejects_cap_minus_one_without_scalar_fallback",
            "compact_accelerator_packet_failure_keeps_accelerator_category",
            "compact_accelerator_decline_runs_the_checked_scalar_fallback",
            "compact_request_rejects_every_ignored_output_option",
            "compact_marker_field_validation_rejects_option_overflow",
            "compact_image_retained_owner_counts_every_nested_actual_capacity",
            "compact_packet_phase_counts_nested_metadata_actual_capacities",
            "compact_final_codestream_high_water_accepts_exact_cap_and_rejects_one_byte_over",
            "compact_accounted_finalizer_preserves_codestream_byte_parity",
            "compact_final_writer_rejects_preflight_before_reservation",
            "compact_preencoded_packetization_borrows_payload_ranges",
            "whole_tile_accelerator_output_accepts_exact_cap_without_copying",
            "whole_tile_accelerator_output_rejects_one_byte_over_without_fallback",
            "whole_tile_accelerator_decline_and_failure_keep_distinct_categories",
            "resident_whole_tile_over_cap_keeps_resource_category",
            "resident_invalid_options_do_not_masquerade_as_resource_failures",
            "malformed_accelerator_output_keeps_the_accelerator_category",
        ]),
    ]);
}

#[test]
fn native_precomputed_53_borrows_coefficients_without_sample_or_dwt_copies() {
    const MODULES: &[(&str, usize)] = &[
        (
            "crates/j2k-native/src/j2c/encode/single_tile/coefficient_source.rs",
            150,
        ),
        (
            "crates/j2k-native/src/j2c/encode/single_tile/coefficient_source/contiguous.rs",
            150,
        ),
        (
            "crates/j2k-native/src/j2c/encode/single_tile/coefficient_source/packed.rs",
            120,
        ),
        (
            "crates/j2k-native/src/j2c/encode/single_tile/precomputed.rs",
            150,
        ),
        ("crates/j2k-native/src/j2c/recode/tests.rs", 110),
    ];
    for (relative, ceiling) in MODULES {
        let source = read(relative);
        let lines = source.lines().count();
        assert!(
            lines <= *ceiling,
            "{relative} has {lines} lines; precomputed 5/3 module ceiling is {ceiling}"
        );
    }

    let api = production_source("crates/j2k-native/src/j2c/encode/precomputed/api53.rs");
    let source = read_source_files(
        repo_root(),
        &[
            "crates/j2k-native/src/j2c/encode/single_tile/coefficient_source.rs",
            "crates/j2k-native/src/j2c/encode/single_tile/coefficient_source/contiguous.rs",
            "crates/j2k-native/src/j2c/encode/single_tile/coefficient_source/packed.rs",
        ],
    );
    let direct = read("crates/j2k-native/src/j2c/encode/single_tile/precomputed.rs");
    let accelerator =
        production_source("crates/j2k-native/src/j2c/encode/precomputed/accelerator.rs");
    let coverage = read_source_files(
        repo_root(),
        &[
            "crates/j2k-native/src/j2c/encode_tests.rs",
            "crates/j2k-native/src/j2c/recode/tests.rs",
        ],
    );

    assert_pattern_checks(&[
        PatternCheck::new("direct precomputed 5/3 API", &api)
            .required(&["encode_precomputed_53_single_tile("])
            .forbidden(&[
                "zero_pixel_buffer(",
                "PrecomputedDwtAccelerator",
                ".dwt.clone()",
                "component.dwt.clone()",
                "encode_with_accelerator_and_component_sample_info_for_session(",
            ]),
        PatternCheck::new("borrowed 5/3 coefficient source", &source).required(&[
            "trait DwtComponentSource",
            "impl DwtComponentSource for PrecomputedHtj2k53Component",
            "Ok(band(&self.dwt.ll, self.dwt.ll_width, self.dwt.ll_height))",
            "level_view(",
            "&level.hl",
            "&level.lh",
            "&level.hh",
        ]),
        PatternCheck::new("direct precomputed single-tile orchestration", &direct).required(&[
            "validate_non_pixel_single_tile_request(",
            "NonPixelSingleTileRequest",
            "build_single_tile_plan(",
            "encode_tile_packets(",
            "&image.components",
            "finalize_precomputed_codestream(",
        ]),
        PatternCheck::new("obsolete 5/3 DWT adapter removal", &accelerator).forbidden(&[
            "struct PrecomputedDwtAccelerator",
            "fn encode_forward_dwt53(",
        ]),
        PatternCheck::new("precomputed 5/3 ownership regressions", &coverage).required(&[
            "assert_eq!(accelerator.deinterleave, 0)",
            "assert_eq!(accelerator.forward_dwt53, 0)",
            "precomputed_htj2k53_borrowed_coefficients_match_pixel_pipeline_codestream",
            "coefficient_tree_baseline_accepts_exact_cap_and_rejects_one_byte_over",
        ]),
    ]);
}

const HIGH_BIT_OWNER_MODULES: &[(&str, usize)] = &[
    ("crates/j2k-native/src/j2c/encode/typed_i64.rs", 80),
    (
        "crates/j2k-native/src/j2c/encode/typed_i64/validation.rs",
        80,
    ),
    ("crates/j2k-native/src/j2c/encode/typed_i64/geometry.rs", 80),
    ("crates/j2k-native/src/j2c/encode/typed_i64/single.rs", 190),
    (
        "crates/j2k-native/src/j2c/encode/typed_i64/multitile.rs",
        260,
    ),
    (
        "crates/j2k-native/src/j2c/encode/typed_i64/multitile/input.rs",
        200,
    ),
    ("crates/j2k-native/src/j2c/encode/typed_i64/plan.rs", 280),
    (
        "crates/j2k-native/src/j2c/encode/typed_i64/plan/accounting.rs",
        110,
    ),
    (
        "crates/j2k-native/src/j2c/encode/typed_i64/plan/construction.rs",
        270,
    ),
    (
        "crates/j2k-native/src/j2c/encode/typed_i64/plan/options.rs",
        90,
    ),
    (
        "crates/j2k-native/src/j2c/encode/typed_i64/plan/transition.rs",
        140,
    ),
    ("crates/j2k-native/src/j2c/encode/typed_i64/prepare.rs", 310),
    (
        "crates/j2k-native/src/j2c/encode/typed_i64/prepare/subband.rs",
        300,
    ),
    (
        "crates/j2k-native/src/j2c/encode/typed_i64/prepare/subband/plan.rs",
        130,
    ),
    (
        "crates/j2k-native/src/j2c/encode/typed_i64/prepare/typed.rs",
        220,
    ),
    (
        "crates/j2k-native/src/j2c/encode/typed_i64/prepare/transform.rs",
        100,
    ),
    (
        "crates/j2k-native/src/j2c/encode/single_tile/reversible_i64.rs",
        110,
    ),
    (
        "crates/j2k-native/src/j2c/encode/single_tile/reversible_i64/input.rs",
        190,
    ),
    (
        "crates/j2k-native/src/j2c/encode/single_tile/reversible_i64/prepare.rs",
        170,
    ),
    (
        "crates/j2k-native/src/j2c/encode/single_tile/reversible_i64/prepare/accounting.rs",
        100,
    ),
    (
        "crates/j2k-native/src/j2c/encode/tile_parts/consume.rs",
        300,
    ),
];

#[test]
fn high_bit_routes_keep_focused_module_boundaries() {
    for (relative, ceiling) in HIGH_BIT_OWNER_MODULES {
        let source = read(relative);
        let lines = source.lines().count();
        assert!(
            lines <= *ceiling,
            "{relative} has {lines} lines; high-bit owner module ceiling is {ceiling}"
        );
    }
}

#[test]
fn high_bit_plan_and_preparation_remain_fallible_and_packed() {
    let facade = production_source("crates/j2k-native/src/j2c/encode/typed_i64.rs");
    let plan = read_source_files(
        repo_root(),
        &[
            "crates/j2k-native/src/j2c/encode/typed_i64/plan.rs",
            "crates/j2k-native/src/j2c/encode/typed_i64/plan/accounting.rs",
            "crates/j2k-native/src/j2c/encode/typed_i64/plan/construction.rs",
            "crates/j2k-native/src/j2c/encode/typed_i64/plan/options.rs",
            "crates/j2k-native/src/j2c/encode/typed_i64/plan/transition.rs",
        ],
    );
    let preparation = read_source_files(
        repo_root(),
        &[
            "crates/j2k-native/src/j2c/encode/typed_i64/prepare.rs",
            "crates/j2k-native/src/j2c/encode/typed_i64/prepare/subband.rs",
            "crates/j2k-native/src/j2c/encode/typed_i64/prepare/subband/plan.rs",
            "crates/j2k-native/src/j2c/encode/typed_i64/prepare/transform.rs",
            "crates/j2k-native/src/j2c/encode/typed_i64/prepare/typed.rs",
        ],
    );

    assert_pattern_checks(&[
        PatternCheck::new("focused typed high-bit facade", &facade)
            .required(&[
                "mod geometry;",
                "mod multitile;",
                "mod plan;",
                "mod prepare;",
                "mod single;",
                "mod validation;",
            ])
            .forbidden(&[
                "struct TypedI64HighBitPlan",
                "Vec::with_capacity(",
                "options.clone()",
                "codestream_write::",
            ]),
        PatternCheck::new("fallible typed high-bit plan", &plan)
            .required(&[
                "pub(super) fn try_new(",
                "try_reserve_exact(",
                "host_allocation_failed(",
                "session.checked_phase(",
                "try_into_execution(",
                "try_high_bit_options(",
                "mod transition;",
            ])
            .forbidden(&["Vec::with_capacity(", ".collect::<", "options.clone()"]),
        PatternCheck::new("packed exact i64 preparation", &preparation)
            .required(&[
                "try_forward_dwt_packed_i64(",
                "PackedDwtGeometry::try_new(",
                "PreparedCodeBlockCoefficients::I64(",
                "prepared_packet_tree_ownership(",
                "try_reserve_exact(",
                "session.checked_phase(",
            ])
            .forbidden(&[
                "forward_dwt_i64(",
                "prepare_subband_i64(",
                "Vec::with_capacity(",
                ".to_vec()",
                ".collect::<",
            ]),
    ]);
}

#[test]
fn high_bit_routes_keep_phase_bounded_consuming_handoffs() {
    let routes = read_source_files(
        repo_root(),
        &[
            "crates/j2k-native/src/j2c/encode/typed_i64/single.rs",
            "crates/j2k-native/src/j2c/encode/typed_i64/multitile.rs",
            "crates/j2k-native/src/j2c/encode/typed_i64/multitile/input.rs",
            "crates/j2k-native/src/j2c/encode/single_tile/reversible_i64.rs",
            "crates/j2k-native/src/j2c/encode/single_tile/reversible_i64/input.rs",
            "crates/j2k-native/src/j2c/encode/single_tile/reversible_i64/prepare.rs",
            "crates/j2k-native/src/j2c/encode/single_tile/reversible_i64/prepare/accounting.rs",
        ],
    );
    let consume = production_source("crates/j2k-native/src/j2c/encode/tile_parts/consume.rs");
    let coverage = read_source_files(
        repo_root(),
        &[
            "crates/j2k-native/src/j2c/encode/typed_i64/tests.rs",
            "crates/j2k-native/src/j2c/encode/typed_i64/plan/tests.rs",
            "crates/j2k-native/src/j2c/encode/typed_i64/prepare/tests.rs",
            "crates/j2k-native/src/j2c/encode/single_tile/reversible_i64/input/tests.rs",
            "crates/j2k-native/src/j2c/encode/tile_parts/consume/tests.rs",
        ],
    );

    assert_pattern_checks(&[
        PatternCheck::new("phase-bounded high-bit routes", &routes)
            .required(&[
                "retained_base_bytes:",
                "single_tile_plan_retained_bytes(&plan)?",
                "consume_packetized_tile_into_tile_parts(",
                "append_encoded_tile_parts(",
                "finalize_multitile_codestream(",
                "write_single_tile_packetized_codestream_for_session(",
                "mod accounting;",
            ])
            .forbidden(&[
                "Vec::with_capacity(",
                "options.clone()",
                "encode_i64_component_resolution_packets(",
                "write_codestream_tiles_accounted_with_peak_check(",
                "|_| Ok(())",
            ]),
        PatternCheck::new("consuming tile-part handoff", &consume)
            .required(&[
                "consume_packetized_tile_into_tile_parts(",
                "drop(packetized);",
                "try_reserve_exact(",
                "encoded_tile_parts_retained_bytes(",
                "packetized_tile_retained_bytes(",
            ])
            .forbidden(&["Vec::with_capacity(", ".to_vec()", ".clone()"]),
        PatternCheck::new("high-bit cap and behavior regressions", &coverage).required(&[
            "plan_construction_accepts_exact_live_peak_and_rejects_one_byte_less",
            "packed_component_preparation_is_i64_exact_and_enforces_peak",
            "deinterleave_is_value_exact_and_enforces_actual_capacity_peak",
            "split_transition_accepts_exact_overlap_peak_and_rejects_one_byte_less",
            "unsplit_transition_moves_packetized_owners_without_payload_copy",
            "high_bit_multitile_ppt_round_trips_without_legacy_writer_or_split_clones",
        ]),
    ]);
}
