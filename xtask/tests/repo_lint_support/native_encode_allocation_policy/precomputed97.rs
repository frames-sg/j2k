// SPDX-License-Identifier: MIT OR Apache-2.0

//! Legacy precomputed 9/7 and aggregate batch ownership ratchets.

use super::{production_source, read};
use crate::repo_lint_support::{assert_pattern_checks, read_source_files, repo_root, PatternCheck};

const MODULES: &[(&str, usize)] = &[
    ("crates/j2k-native/src/j2c/encode/precomputed/api53.rs", 300),
    ("crates/j2k-native/src/j2c/encode/precomputed/api97.rs", 450),
    (
        "crates/j2k-native/src/j2c/encode/precomputed/limits.rs",
        140,
    ),
    (
        "crates/j2k-native/src/j2c/encode/precomputed/limits/tests.rs",
        100,
    ),
    (
        "crates/j2k-native/src/j2c/encode/precomputed/options.rs",
        100,
    ),
    (
        "crates/j2k-native/src/j2c/encode/precomputed/allocation.rs",
        300,
    ),
    (
        "crates/j2k-native/src/j2c/encode/precomputed/orchestrator.rs",
        350,
    ),
    (
        "crates/j2k-native/src/j2c/encode/precomputed/packets.rs",
        380,
    ),
    (
        "crates/j2k-native/src/j2c/encode/precomputed/batch97.rs",
        100,
    ),
    (
        "crates/j2k-native/src/j2c/encode/precomputed/batch97/prepare.rs",
        300,
    ),
    (
        "crates/j2k-native/src/j2c/encode/precomputed/batch97/finalize.rs",
        250,
    ),
    (
        "crates/j2k-native/src/j2c/encode/precomputed/compact97.rs",
        500,
    ),
    ("crates/j2k-native/src/j2c/encode/precomputed_batch.rs", 350),
];

struct Sources {
    api53: String,
    api97: String,
    limits: String,
    coefficient_source: String,
    allocation: String,
    packets: String,
    orchestrator: String,
    batch_shell: String,
    batch_prepare: String,
    batch_finalize: String,
    direct_batch: String,
    transcode_batch: String,
    transcode_error: String,
    transcode_single: String,
    coverage: String,
    typed_coverage: String,
}

impl Sources {
    fn read() -> Self {
        let direct_batch =
            production_source("crates/j2k-native/src/j2c/encode/precomputed_batch.rs")
                .split("pub(super) fn copy_code_block_coefficients(")
                .next()
                .expect("session-aware batch preparation prefix")
                .to_owned();
        Self {
            api53: production_source("crates/j2k-native/src/j2c/encode/precomputed/api53.rs"),
            api97: production_source("crates/j2k-native/src/j2c/encode/precomputed/api97.rs"),
            limits: production_source("crates/j2k-native/src/j2c/encode/precomputed/limits.rs"),
            coefficient_source: read_source_files(
                repo_root(),
                &[
                    "crates/j2k-native/src/j2c/encode/single_tile/coefficient_source.rs",
                    "crates/j2k-native/src/j2c/encode/single_tile/coefficient_source/contiguous.rs",
                    "crates/j2k-native/src/j2c/encode/single_tile/coefficient_source/packed.rs",
                ],
            ),
            allocation: production_source(
                "crates/j2k-native/src/j2c/encode/precomputed/allocation.rs",
            ),
            packets: production_source("crates/j2k-native/src/j2c/encode/precomputed/packets.rs"),
            orchestrator: production_source(
                "crates/j2k-native/src/j2c/encode/precomputed/orchestrator.rs",
            ),
            batch_shell: production_source(
                "crates/j2k-native/src/j2c/encode/precomputed/batch97.rs",
            ),
            batch_prepare: production_source(
                "crates/j2k-native/src/j2c/encode/precomputed/batch97/prepare.rs",
            ),
            batch_finalize: production_source(
                "crates/j2k-native/src/j2c/encode/precomputed/batch97/finalize.rs",
            ),
            direct_batch,
            transcode_batch: [
                production_source("crates/j2k-transcode/src/jpeg_to_htj2k/batch/encode.rs"),
                production_source(
                    "crates/j2k-transcode/src/jpeg_to_htj2k/batch/encode/precomputed.rs",
                ),
            ]
            .concat(),
            transcode_error: [
                production_source("crates/j2k-transcode/src/jpeg_to_htj2k/error.rs"),
                production_source("crates/j2k-transcode/src/jpeg_to_htj2k/error/native_encode.rs"),
            ]
            .concat(),
            transcode_single: production_source(
                "crates/j2k-transcode/src/jpeg_to_htj2k/single_tile_encode.rs",
            ),
            coverage: read_source_files(
                repo_root(),
                &[
                    "crates/j2k-native/src/j2c/encode/precomputed/limits/tests.rs",
                    "crates/j2k-native/src/j2c/encode/precomputed/api97/tests.rs",
                    "crates/j2k-native/src/j2c/encode/precomputed/batch97/tests.rs",
                ],
            ),
            typed_coverage: read_source_files(
                repo_root(),
                &[
                    "crates/j2k-native/src/j2c/encode/precomputed/api53/tests.rs",
                    "crates/j2k-native/src/j2c/encode/precomputed/api97/tests.rs",
                    "crates/j2k-native/src/j2c/encode/precomputed/batch97/tests.rs",
                    "crates/j2k-transcode/src/jpeg_to_htj2k/error.rs",
                ],
            ),
        }
    }
}

#[test]
fn legacy_precomputed_97_packet_inputs_and_batch_share_one_typed_owner_plan() {
    for &(relative, ceiling) in MODULES {
        let source = read(relative);
        let lines = source.lines().count();
        assert!(
            lines <= ceiling,
            "{relative} has {lines} lines; legacy 9/7 ownership ceiling is {ceiling}"
        );
    }
    let sources = Sources::read();
    assert_typed_public_errors(&sources);
    assert_direct_paths(&sources);
    assert_batch_paths(&sources);
    assert_regressions(&sources);
}

fn assert_direct_paths(sources: &Sources) {
    assert_pattern_checks(&[
        PatternCheck::new("borrowed direct precomputed 9/7 API", &sources.api97)
            .required(&[
                "encode_precomputed_htj2k_97_with_accelerator_and_max_host_bytes(",
                "encode_precomputed_for_session(",
                "encode_precomputed_97_single_tile(",
                "PrecomputedStageAccelerator",
            ])
            .forbidden(&[
                "zero_pixel_buffer(",
                "PrecomputedDwt97Accelerator",
                "component.dwt.clone()",
                "encode_with_accelerator(",
            ]),
        PatternCheck::new("lowered-cap precomputed 9/7 adapter", &sources.limits).required(&[
            "precomputed_97_image_retained_bytes(image)",
            "NativeEncodeRetainedInput::from_owner_bytes(image, retained_bytes)",
            "NativeEncodeSession::try_with_lowered_cap(retained_input, max_host_bytes)",
            "encode_precomputed_for_session(image, options, &session, accelerator)",
        ]),
        PatternCheck::new(
            "borrowed 9/7 coefficient source",
            &sources.coefficient_source,
        )
        .required(&[
            "impl DwtComponentSource for PrecomputedHtj2k97Component",
            "Ok(band(&self.dwt.ll, self.dwt.ll_width, self.dwt.ll_height))",
            "level_view(",
            "&level.hl",
            "&level.lh",
            "&level.hh",
        ]),
        PatternCheck::new("legacy 9/7 actual-capacity accounting", &sources.allocation).required(
            &[
                "struct ConstructionTracker",
                "try_reserve_exact(count)",
                "values.capacity()",
                "precomputed_97_images_retained_bytes(",
                "block.coefficients.capacity()",
                "block.encoded.data.capacity()",
            ],
        ),
        PatternCheck::new("legacy 9/7 fallible packet construction", &sources.packets)
            .required(&[
                "try_prepared_packets_from_prequantized_component(",
                "try_prepared_packets_from_preencoded_component(",
                "try_preencoded_owned_skeleton(",
                "move_preencoded_payloads_into_skeleton(",
                "target_payloads.push(source_block.encoded)",
            ])
            .forbidden(&[
                ".clone()",
                ".collect::<",
                "Vec::with_capacity(",
                ".to_vec()",
            ]),
        PatternCheck::new(
            "legacy 9/7 typed packet orchestration",
            &sources.orchestrator,
        )
        .required(&[
            "encode_prepared_resolution_packets_for_session(",
            "packetize_resolution_packets_with_options_for_session(",
            "write_single_tile_packetized_codestream_for_session(",
            "NativeEncodePipelineResult<Vec<u8>>",
        ]),
    ]);
}

fn assert_typed_public_errors(sources: &Sources) {
    assert_typed_signatures("precomputed 5/3", &sources.api53, 8);
    assert_typed_signatures("precomputed 9/7", &sources.api97, 8);
    assert_typed_signatures("precomputed 9/7 batch", &sources.batch_shell, 2);
    let native_apis = format!("{}{}{}", sources.api53, sources.api97, sources.batch_shell);
    assert_pattern_checks(&[
        PatternCheck::new("typed precomputed public adapters", &native_apis)
            .required(&["NativeEncodePipelineError::into_encode_error"])
            .forbidden(&[
                "Result<Vec<u8>, &'static str>",
                "Result<Vec<Vec<u8>>, &'static str>",
                "NativeEncodePipelineError::into_legacy_detail",
            ]),
        PatternCheck::new("typed transcode encode mapping", &sources.transcode_error)
            .required(&[
                "Encode(Htj2kEncodeError)",
                "Self::Encode(err) => Some(err)",
                "fn map_encode_error(value: j2k_native::EncodeError)",
                "EncodeError::AllocationTooLarge { requested, cap, .. }",
                "EncodeError::HostAllocationFailed { bytes, .. }",
                "pub struct Htj2kEncodeError",
                "source: EncodeError",
                "pub const fn kind(&self) -> Htj2kEncodeErrorKind",
                "Some(&self.source)",
            ])
            .forbidden(&[
                "Encode(&'static str)",
                "Encode(j2k_native::EncodeError)",
                "Self::Encode(_) => None",
                "source.to_string()",
            ]),
    ]);
    let transcode_calls = format!("{}{}", sources.transcode_single, sources.transcode_batch);
    assert_pattern_checks(&[
        PatternCheck::new("typed transcode precomputed call sites", &transcode_calls)
            .required(&[".map_err(map_encode_error)"])
            .forbidden(&[
                ".map_err(JpegToHtj2kError::Encode)",
                "Err(JpegToHtj2kError::Encode(error))",
            ]),
        PatternCheck::new("typed public-error regressions", &sources.typed_coverage).required(&[
            "public_precomputed_53_keeps_invalid_input_and_accelerator_categories",
            "public_precomputed_97_keeps_accelerator_error_category",
            "public_compact_preencoded_97_keeps_invalid_input_category",
            "public_precomputed_97_batch_keeps_accelerator_error_category",
            "native_encode_resource_errors_lift_into_transcode_resource_categories",
            "native_encode_semantic_errors_remain_typed_and_are_error_sources",
        ]),
    ]);
}

fn assert_typed_signatures(label: &str, source: &str, expected_count: usize) {
    let signatures = source
        .split("pub fn encode_")
        .skip(1)
        .map(|tail| tail.split('{').next().expect("public encode signature"))
        .collect::<Vec<_>>();
    assert_eq!(
        signatures.len(),
        expected_count,
        "{label} public encode API count changed; review the typed-error ratchet"
    );
    for signature in signatures {
        assert!(
            signature.contains("-> crate::EncodeResult<"),
            "{label} public encode API is not typed: pub fn encode_{signature}"
        );
    }
}

fn assert_batch_paths(sources: &Sources) {
    assert_pattern_checks(&[
        PatternCheck::new(
            "owned batch releases source before Tier-1",
            &sources.batch_shell,
        )
        .required(&[
            "encode_precomputed_htj2k_97_batch_owned_with_accelerator(",
            "fn prepare_owned_batch_plans(",
            "let plans = {",
            "let input_session = NativeEncodeSession::try_with_lowered_cap(",
            "prepare_batch_plans(&images, options, &input_session)?",
            "drop(images);",
            "encode_prepared_batch(plans, &session, accelerator)",
        ])
        .forbidden(&["drop(input_session);"]),
        PatternCheck::new("owned single-image lexical handoff", &sources.api97).required(&[
            "prepare_owned_preencoded_handoff(&image, options, max_host_bytes)?",
            "move_preencoded_payloads_into_skeleton(image, &mut components)",
            "fn prepare_owned_preencoded_handoff(",
            "let input_session = NativeEncodeSession::try_with_lowered_cap(",
        ]),
        PatternCheck::new("bounded shared Tier-1 batch", &sources.batch_prepare)
            .required(&[
                "try_reserve_exact(images.len())",
                "encode_prepared_resolution_packets_for_session(",
                "split_encoded_packets(",
                "packet_counts.capacity()",
            ])
            .forbidden(&[
                "par_iter()",
                "into_par_iter()",
                "Vec::with_capacity(",
                ".collect()",
            ]),
        PatternCheck::new("aggregate batch finalization", &sources.batch_finalize).required(&[
            "struct BatchTailOwners<'a>",
            "plans: &'a [Prepared97PacketPlan]",
            "groups: &'a [Vec<ResolutionPacket>]",
            "let remaining = BatchTailOwners {",
            "remaining.codestreams",
            "codestream.capacity()",
            "packetize_resolution_packets_with_options_for_session(",
            "write_single_tile_packetized_codestream_for_session(",
        ]),
        PatternCheck::new(
            "session-aware borrowed precomputed batch preparation",
            &sources.direct_batch,
        )
        .required(&[
            "ConstructionTracker::new(session, retained_base_bytes)",
            "prepare_subband_for_session(",
            "prepared_subbands_ownership(",
        ])
        .forbidden(&[
            "prepare_subband_cpu_quantized(",
            "Vec::with_capacity(",
            ".collect::<",
        ]),
    ]);
}

fn assert_regressions(sources: &Sources) {
    assert_pattern_checks(&[
        PatternCheck::new(
            "transcode consumes its prepared batch",
            &sources.transcode_batch,
        )
        .required(&[
            "encode_precomputed_htj2k_97_batch_owned_with_accelerator_and_max_host_bytes(",
            "images,",
            "native_host_cap,",
        ]),
        PatternCheck::new("legacy 9/7 ownership regressions", &sources.coverage).required(&[
            "lowered_public_cap_accepts_measured_exact_peak_and_rejects_one_less",
            "requested_cap_above_process_ceiling_is_clamped",
            "precomputed_97_quantization_borrows_the_source_dwt_allocation",
            "borrowed_precomputed_97_accepts_its_measured_exact_peak_and_rejects_one_byte_less",
            "owned_preencoded_97_moves_payloads_through_packetization_without_copying",
            "copied_packet_inputs_enforce_exact_aggregate_caps",
            "owned_precomputed_97_batch_keeps_one_tier1_batch_and_byte_parity",
            "aggregate_precomputed_97_batch_accepts_measured_peak_and_rejects_one_byte_less",
            "batch_preserves_progression_markers_and_tile_part_behavior",
        ]),
    ]);
}
