// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, path::Path};

use crate::repo_lint_support::{assert_pattern_checks, repo_root, PatternCheck};

struct CudaEncodeStructureSources {
    htj2k_root: String,
    htj2k_types: String,
    htj2k_validation: String,
    htj2k_code_blocks: String,
    htj2k_resident: String,
    htj2k_tile_packets: String,
    packet_root: String,
    packet_types: String,
    packet_tag_tree: String,
    packet_tag_tree_allocation: String,
    packet_tag_tree_allocation_tests: String,
    packet_state: String,
    packet_state_count: String,
    packet_flatten: String,
    packet_runtime: String,
    packet_tests: String,
    packet_ht_segment_tests: String,
}

impl CudaEncodeStructureSources {
    fn read(root: &Path) -> Self {
        let read = |relative: &str| {
            fs::read_to_string(root.join(relative))
                .unwrap_or_else(|error| panic!("read {relative}: {error}"))
        };
        Self {
            htj2k_root: read("crates/j2k-cuda/src/encode/htj2k.rs"),
            htj2k_types: read("crates/j2k-cuda/src/encode/htj2k/types.rs"),
            htj2k_validation: read("crates/j2k-cuda/src/encode/htj2k/validation.rs"),
            htj2k_code_blocks: read("crates/j2k-cuda/src/encode/htj2k/code_blocks.rs"),
            htj2k_resident: read("crates/j2k-cuda/src/encode/htj2k/resident.rs"),
            htj2k_tile_packets: read("crates/j2k-cuda/src/encode/htj2k/tile_packets.rs"),
            packet_root: read("crates/j2k-cuda/src/encode/packetization.rs"),
            packet_types: read("crates/j2k-cuda/src/encode/packetization/types.rs"),
            packet_tag_tree: read("crates/j2k-cuda/src/encode/packetization/tag_tree.rs"),
            packet_tag_tree_allocation: read(
                "crates/j2k-cuda/src/encode/packetization/tag_tree/allocation.rs",
            ),
            packet_tag_tree_allocation_tests: read(
                "crates/j2k-cuda/src/encode/packetization/tag_tree/allocation/tests.rs",
            ),
            packet_state: read("crates/j2k-cuda/src/encode/packetization/state.rs"),
            packet_state_count: read("crates/j2k-cuda/src/encode/packetization/state/count.rs"),
            packet_flatten: read("crates/j2k-cuda/src/encode/packetization/flatten.rs"),
            packet_runtime: read("crates/j2k-cuda/src/encode/packetization/runtime.rs"),
            packet_tests: read("crates/j2k-cuda/src/encode/packetization/tests.rs"),
            packet_ht_segment_tests: read(
                "crates/j2k-cuda/src/encode/packetization/tests/ht_segment.rs",
            ),
        }
    }

    fn capped_sources(&self) -> [(&str, &str, usize); 17] {
        [
            ("encode/htj2k.rs", &self.htj2k_root, 40),
            ("encode/htj2k/types.rs", &self.htj2k_types, 125),
            ("encode/htj2k/validation.rs", &self.htj2k_validation, 100),
            ("encode/htj2k/code_blocks.rs", &self.htj2k_code_blocks, 325),
            ("encode/htj2k/resident.rs", &self.htj2k_resident, 600),
            (
                "encode/htj2k/tile_packets.rs",
                &self.htj2k_tile_packets,
                200,
            ),
            ("encode/packetization.rs", &self.packet_root, 45),
            ("encode/packetization/types.rs", &self.packet_types, 140),
            (
                "encode/packetization/tag_tree.rs",
                &self.packet_tag_tree,
                225,
            ),
            (
                "encode/packetization/tag_tree/allocation.rs",
                &self.packet_tag_tree_allocation,
                100,
            ),
            (
                "encode/packetization/tag_tree/allocation/tests.rs",
                &self.packet_tag_tree_allocation_tests,
                50,
            ),
            ("encode/packetization/state.rs", &self.packet_state, 375),
            (
                "encode/packetization/state/count.rs",
                &self.packet_state_count,
                50,
            ),
            ("encode/packetization/flatten.rs", &self.packet_flatten, 450),
            ("encode/packetization/runtime.rs", &self.packet_runtime, 150),
            ("encode/packetization/tests.rs", &self.packet_tests, 80),
            (
                "encode/packetization/tests/ht_segment.rs",
                &self.packet_ht_segment_tests,
                80,
            ),
        ]
    }
}

fn assert_focused_real_modules(sources: &CudaEncodeStructureSources) {
    for (relative, source, max_lines) in sources.capped_sources() {
        assert!(
            source.lines().count() < max_lines,
            "CUDA encode owner {relative} exceeded its focused line-count ratchet"
        );
        assert!(
            !source.contains("include!(")
                && !source.lines().any(|line| {
                    let line = line.trim_start();
                    line.starts_with("use ") && line.contains("::*")
                }),
            "CUDA encode owner {relative} must use explicit real-module boundaries"
        );
    }
}

fn assert_htj2k_ownership(sources: &CudaEncodeStructureSources) {
    assert_pattern_checks(&[
        PatternCheck::new("CUDA HTJ2K facade", &sources.htj2k_root)
            .required(&[
                "mod code_blocks;",
                "mod resident;",
                "mod tile_packets;",
                "mod types;",
                "mod validation;",
                "pub(crate) use self::code_blocks::cuda_htj2k_encode_tables;",
            ])
            .forbidden(&[
                "fn cuda_encode_ht_code_block(",
                "struct CudaEncodedHtj2kTile",
                "fn cuda_encode_htj2k_tile_body(",
                "fn cuda_packetize_tile_body(",
            ]),
        PatternCheck::new("CUDA HTJ2K result types", &sources.htj2k_types).required(&[
            "struct CudaEncodedHtj2kTile",
            "struct CudaEncodedHtSubband",
            "struct CudaHtj2kTileEncodeStats",
        ]),
        PatternCheck::new("CUDA HTJ2K validation", &sources.htj2k_validation).required(&[
            "fn resident_job_from_host(",
            "fn validate_cuda_htj2k_tile_job(",
        ]),
        PatternCheck::new("CUDA HTJ2K code-block launches", &sources.htj2k_code_blocks).required(
            &[
                "fn cuda_encode_ht_code_block(",
                "fn cuda_encode_ht_code_blocks(",
                "fn cuda_encode_ht_subband(",
                "fn encoded_ht_code_blocks_from_cuda(",
                "pub(crate) fn cuda_htj2k_encode_tables(",
            ],
        ),
        PatternCheck::new("CUDA HTJ2K resident DWT", &sources.htj2k_resident).required(&[
            "fn cuda_encode_htj2k_device_tile_body(",
            "fn cuda_encode_htj2k_resident_components_body(",
            "fn cuda_encode_dwt_component_packets(",
            "fn cuda_encode_tile_subband_region(",
        ]),
        PatternCheck::new(
            "CUDA HTJ2K tile packet assembly",
            &sources.htj2k_tile_packets,
        )
        .required(&[
            "fn cuda_packetize_tile_body(",
            "fn cuda_tile_packet_descriptors(",
        ]),
    ]);
}

fn assert_packetization_ownership(sources: &CudaEncodeStructureSources) {
    assert_pattern_checks(&[
        PatternCheck::new("CUDA packetization facade", &sources.packet_root)
            .required(&[
                "mod flatten;",
                "mod runtime;",
                "mod state;",
                "mod tag_tree;",
                "mod types;",
                "mod tests;",
            ])
            .forbidden(&[
                "struct CudaHtj2kPacketizationPlan",
                "fn flatten_cuda_htj2k_packetization_job(",
                "fn cuda_packetization_packets(",
            ]),
        PatternCheck::new("CUDA packetization ABI types", &sources.packet_types)
            .required(&[
                "#[derive(Debug, PartialEq, Eq)]\npub(in crate::encode) struct CudaHtj2kPacketizationPlan",
                "struct CudaHtj2kPacketizationPlanTagNodeState",
            ])
            .forbidden(&[
                "#[derive(Debug, Clone, PartialEq, Eq)]\npub(in crate::encode) struct CudaHtj2kPacketizationPlan",
            ]),
        PatternCheck::new("CUDA packetization tag tree", &sources.packet_tag_tree)
            .required(&[
                "#[derive(Debug, PartialEq, Eq)]\npub(super) struct CudaHtj2kPacketizationTagTreeState",
                "try_tag_tree_vec_filled(",
                "checked_tag_tree_retained_bytes(",
                "fn propagate(&mut self)",
                "fn append_snapshot(",
            ])
            .forbidden(&["vec![0; total_nodes]", "Vec::with_capacity(self.widths.len())"]),
        PatternCheck::new(
            "CUDA packetization tag-tree allocation",
            &sources.packet_tag_tree_allocation,
        )
        .required(&[
            "host_budget: &mut HostPhaseBudget",
            ".try_vec_with_capacity(capacity)",
            ".try_vec_filled(len, value)",
            "checked_tag_tree_retained_bytes(",
        ]),
        PatternCheck::new(
            "CUDA packetization tag-tree allocation regressions",
            &sources.packet_tag_tree_allocation_tests,
        )
        .required(&[
            "tag_tree_oversized_request_is_rejected_before_allocation",
            "tag_tree_actual_capacity_has_exact_and_one_over_boundaries",
        ]),
        PatternCheck::new("CUDA packetization state", &sources.packet_state)
            .required(&[
                "#[derive(Debug, PartialEq, Eq)]\npub(super) struct CudaHtj2kPacketizationSubbandState",
                "#[derive(Debug, PartialEq, Eq)]\npub(super) struct CudaHtj2kPacketizationState",
                "fn seed_cuda_htj2k_packetization_state(",
                "fn update_cuda_htj2k_packetization_state_after_block(",
                "fn cuda_ht_segment_lengths(",
            ])
            .forbidden(&[
                "#[derive(Debug, Clone, PartialEq, Eq)]\npub(super) struct CudaHtj2kPacketizationSubbandState",
                "#[derive(Debug, Clone, PartialEq, Eq)]\npub(super) struct CudaHtj2kPacketizationState",
            ]),
        PatternCheck::new("CUDA packetization state count", &sources.packet_state_count)
            .required(&[
                "fn cuda_packetization_state_count(",
                "fn checked_cuda_packetization_state_count(",
                ".checked_add(1)",
            ]),
        PatternCheck::new("CUDA packet descriptor flattening", &sources.packet_flatten).required(
            &[
                "fn flatten_cuda_htj2k_packetization_job_classified(",
                "fn flatten_cuda_htj2k_packet_inner(",
            ],
        ),
        PatternCheck::new("CUDA packet runtime conversion", &sources.packet_runtime).required(&[
            "fn cuda_packetization_packets(",
            "fn cuda_packetization_blocks(",
            "fn cuda_packetization_tag_nodes(",
        ]),
        PatternCheck::new("CUDA packet boundary test facade", &sources.packet_tests).required(&[
            "mod ht_segment;",
            "sparse_descriptor_state_index_is_rejected_before_state_allocation",
            "descriptor_state_count_addition_is_checked",
            "packetization_plan_allocation_failure_keeps_its_typed_category",
        ]),
        PatternCheck::new(
            "CUDA HT segment category tests",
            &sources.packet_ht_segment_tests,
        )
        .required(&["ht_segment_validation_errors_keep_invalid_and_overflow_categories"]),
    ]);
}

#[test]
fn cuda_htj2k_and_packetization_owners_stay_split_and_focused() {
    let sources = CudaEncodeStructureSources::read(repo_root());
    assert_focused_real_modules(&sources);
    assert_htj2k_ownership(&sources);
    assert_packetization_ownership(&sources);
}
