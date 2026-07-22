// SPDX-License-Identifier: MIT OR Apache-2.0

//! ALLOC-018 policy for caller-sized J2K/JPEG Metal batch ownership.

use super::super::{assert_pattern_checks, read_source_files, repo_root, PatternCheck};

mod inventory;

#[test]
fn metal_batch_allocators_keep_checked_aggregate_typed_contracts() {
    let root = repo_root();
    let core_allocator = read_source_files(root, &["crates/j2k-core/src/batch/allocation.rs"]);
    let allocators = read_source_files(
        root,
        &[
            "crates/j2k-metal/src/batch_allocation.rs",
            "crates/j2k-jpeg-metal/src/batch_allocation.rs",
        ],
    );
    let j2k_error = read_source_files(root, &["crates/j2k-metal/src/error.rs"]);
    let jpeg_error = read_source_files(root, &["crates/j2k-jpeg-metal/src/error.rs"]);
    let entropy = read_source_files(
        root,
        &["crates/j2k-jpeg-metal/src/compute/batch_support.rs"],
    );

    assert_pattern_checks(&[
        PatternCheck::new("shared core batch allocation budget", &core_allocator)
            .required(&[
                "pub struct BatchAllocationBudget",
                "pub struct BatchAllocationRequest",
                "checked_mul",
                "checked_add",
                "try_host_vec_with_capacity",
                "try_host_vec_filled",
                "account_capacity",
                "values.capacity()",
                "BatchInfrastructureError::AllocationTooLarge",
                "BatchInfrastructureError::HostAllocationFailed",
                "actual_capacities_are_cumulative_and_allocator_failure_is_exact",
            ])
            .forbidden(&[
                "Error::MetalKernel",
                "JpegEncodeError",
                "BufferError::HostAllocationFailed",
                "saturating_mul",
            ]),
        PatternCheck::new("thin Metal adapter batch allocation contexts", &allocators)
            .required(&[
                "BatchAllocationBudget as BatchMetadataBudget",
                "BatchAllocationRequest as BatchMetadataRequest",
                "allocator_failure_keeps_batch_infrastructure_source_and_category",
                "grouped_result_plan",
            ])
            .forbidden(&[
                "struct BatchMetadataBudget",
                "try_host_vec_with_capacity",
                "fn reconcile_capacity",
            ]),
        PatternCheck::new("J2K Metal batch infrastructure variant", &j2k_error)
            .required(&["| Self::BatchInfrastructure(_)"])
            .normalized_required(&[
                "BatchInfrastructure( #[from] #[source] BatchInfrastructureError,",
            ]),
        PatternCheck::new("JPEG Metal batch infrastructure variant", &jpeg_error)
            .required(&["| Self::BatchInfrastructure(_)"])
            .normalized_required(&[
                "BatchInfrastructure( #[from] #[source] BatchInfrastructureError,",
            ]),
        PatternCheck::new("JPEG Metal entropy aggregate planning", &entropy)
            .required(&[
                "checked_count_product(",
                "BatchMetadataRequest::of::<u8>(total_entropy_len)",
                "BatchMetadataRequest::of::<u32>(tile_count)",
                "BatchMetadataRequest::of::<JpegEntropyCheckpointHost>(checkpoint_count)",
                "JPEG Metal batch entropy checkpoints",
            ])
            .forbidden(&[
                "tile_count * segment_count",
                "Vec::with_capacity",
                "Error::MetalKernel {\n            message: labels.total_len_overflow",
            ]),
    ]);
}

#[test]
fn remediated_metal_batch_paths_do_not_regress_to_infallible_collections() {
    let root = repo_root();
    let jpeg_paths = [
        "crates/j2k-jpeg-metal/src/batch.rs",
        "crates/j2k-jpeg-metal/src/codec_batch.rs",
        "crates/j2k-jpeg-metal/src/lib.rs",
        "crates/j2k-jpeg-metal/src/session.rs",
        "crates/j2k-jpeg-metal/src/surface.rs",
        "crates/j2k-jpeg-metal/src/surface/batch_buffer.rs",
        "crates/j2k-jpeg-metal/src/surface/batch_texture.rs",
        "crates/j2k-jpeg-metal/src/surface/resident_tile.rs",
        "crates/j2k-jpeg-metal/src/surface/texture_tile.rs",
        "crates/j2k-jpeg-metal/src/tile_batch.rs",
        "crates/j2k-jpeg-metal/src/compute/batch_entry.rs",
        "crates/j2k-jpeg-metal/src/compute/batch_plan.rs",
        "crates/j2k-jpeg-metal/src/compute/batch_support.rs",
        "crates/j2k-jpeg-metal/src/compute/fast_packets/params.rs",
        "crates/j2k-jpeg-metal/src/compute/region_scaled_plan.rs",
        "crates/j2k-jpeg-metal/src/compute/batch_full/rgb.rs",
        "crates/j2k-jpeg-metal/src/compute/batch_full/texture.rs",
        "crates/j2k-jpeg-metal/src/compute/batch_full/texture_grouped.rs",
        "crates/j2k-jpeg-metal/src/compute/batch_region/rgb.rs",
        "crates/j2k-jpeg-metal/src/compute/batch_region/texture/fast444.rs",
        "crates/j2k-jpeg-metal/src/compute/batch_region/texture/subsampled.rs",
        "crates/j2k-jpeg-metal/src/compute/pack_dispatch/grouped_output.rs",
        "crates/j2k-jpeg-metal/src/compute/pack_dispatch/texture.rs",
    ];
    let j2k_paths = [
        "crates/j2k-metal/src/batch/cpu.rs",
        "crates/j2k-metal/src/batch/execute.rs",
        "crates/j2k-metal/src/batch/heuristics.rs",
        "crates/j2k-metal/src/decoder/adapters.rs",
        "crates/j2k-metal/src/decoder/direct_paths.rs",
        "crates/j2k-metal/src/encode/batch.rs",
        "crates/j2k-metal/src/encode/packet_plan.rs",
        "crates/j2k-metal/src/encode/resident_schedule.rs",
        "crates/j2k-metal/src/encode/resident_submit.rs",
        "crates/j2k-metal/src/encode/resident_wait.rs",
        "crates/j2k-metal/src/encode/submitted.rs",
        "crates/j2k-metal/src/hybrid.rs",
        "crates/j2k-metal/src/hybrid/plan_resolution.rs",
        "crates/j2k-metal/src/tile_batch.rs",
        "crates/j2k-metal/src/compute/decode_cleanup.rs",
        "crates/j2k-metal/src/compute/decode_dispatch/classic_cleanup.rs",
        "crates/j2k-metal/src/compute/decode_dispatch/classic_cleanup/distinct_allocation.rs",
        "crates/j2k-metal/src/compute/decode_dispatch/classic_cleanup/distinct_batch.rs",
        "crates/j2k-metal/src/compute/decode_dispatch/ht_distinct.rs",
        "crates/j2k-metal/src/compute/direct_execute.rs",
        "crates/j2k-metal/src/compute/direct_cpu.rs",
        "crates/j2k-metal/src/compute/direct_grayscale_execute.rs",
        "crates/j2k-metal/src/compute/direct_grayscale_execute/allocation.rs",
        "crates/j2k-metal/src/compute/direct_grayscale_execute/component_plane.rs",
        "crates/j2k-metal/src/compute/direct_grayscale_execute/component_plane/execution.rs",
        "crates/j2k-metal/src/compute/direct_grayscale_execute/single.rs",
        "crates/j2k-metal/src/compute/direct_plane_pack.rs",
        "crates/j2k-metal/src/compute/direct_stacked_batch/command_submission.rs",
        "crates/j2k-metal/src/compute/direct_stacked_batch/repeated_grayscale/execution.rs",
        "crates/j2k-metal/src/compute/direct_stacked_batch/resources.rs",
        "crates/j2k-metal/src/compute/direct_flattened.rs",
        "crates/j2k-metal/src/compute/direct_prepare.rs",
        "crates/j2k-metal/src/compute/direct_roi.rs",
        "crates/j2k-metal/src/compute/direct_surface_pack.rs",
        "crates/j2k-metal/src/compute/resident_packet_plan.rs",
        "crates/j2k-metal/src/compute/resident_tier1/counter_validation/validate.rs",
        "crates/j2k-metal/src/compute/resident_tier1/readback.rs",
        "crates/j2k-metal/src/compute/resident_codestream/classic_tier1.rs",
        "crates/j2k-metal/src/compute/resident_codestream/ht_tier1.rs",
        "crates/j2k-metal/src/compute/resident_codestream/resident_single.rs",
        "crates/j2k-metal/src/compute/resident_codestream/tier2_packetization.rs",
        "crates/j2k-metal/src/compute/tier1_encode.rs",
    ];
    let jpeg = read_source_files(root, &jpeg_paths);
    let j2k = read_source_files(root, &j2k_paths);

    assert_pattern_checks(&[
        PatternCheck::new("JPEG Metal remediated batch paths", &jpeg)
            .required(&["BatchMetadataBudget::new(", "try_reserve_for_push("])
            .forbidden(&["Vec::with_capacity", ".collect::<Vec", ".collect()"]),
        PatternCheck::new("J2K Metal remediated batch paths", &j2k)
            .required(&["BatchMetadataBudget::new(", "try_reserve_for_push("])
            .forbidden(&[
                "Vec::with_capacity",
                "Vec::<usize>::with_capacity",
                "HashMap::<",
                ".collect::<Vec",
                ".collect()",
                "vec![0.0_f32;",
            ]),
    ]);
}

#[test]
fn metal_submission_queues_share_one_fallible_aggregate_owner() {
    let root = repo_root();
    let shared = read_source_files(root, &["crates/j2k-metal-support/src/submission_queue.rs"]);
    let adapters = read_source_files(
        root,
        &[
            "crates/j2k-metal/src/tile_batch.rs",
            "crates/j2k-jpeg-metal/src/tile_batch.rs",
            "crates/j2k-metal/src/decoder/adapters.rs",
            "crates/j2k-metal/src/compute/resident_tier1/readback.rs",
            "crates/j2k-metal/src/encode/submitted.rs",
        ],
    );

    assert_pattern_checks(&[
        PatternCheck::new("shared fallible Metal submission queue", &shared)
            .required(&[
                "pub struct FallibleSubmissionQueue<S>",
                "try_batch_reserve_to(&mut self.submissions",
                "try_batch_reserve_for_push(&mut self.submissions",
                "pub fn try_push_with<E>",
                "pub fn try_finish<O, E>",
                "account_capacity::<S>(self.submissions.capacity())",
                "BatchAllocationRequest::of::<O>(output_count)",
                "finish_budget_counts_live_submission_and_output_capacities",
                "oversized_hint_fails_before_submission_construction",
            ])
            .forbidden(&["Vec::with_capacity", ".collect::<Vec", ".collect()"]),
        PatternCheck::new("Metal submission queue adapters", &adapters)
            .required(&[
                "FallibleSubmissionQueue<batch::MetalSubmission>",
                "FallibleSubmissionQueue::with_capacity_hint",
                ".try_push_with(",
                ".try_finish(",
                "FallibleSubmissionQueue::from_retained",
            ])
            .forbidden(&[
                "submissions: Vec<batch::MetalSubmission>",
                "capacity_hint: usize",
                "try_reserve_to(&mut self.submissions",
                "try_reserve_for_push(&mut self.submissions",
                ".collect::<Vec",
                ".collect()",
            ]),
    ]);
}

#[test]
fn j2k_direct_batch_metadata_preflights_nested_live_sets() {
    let root = repo_root();
    let sources = read_source_files(
        root,
        &[
            "crates/j2k-metal/src/compute/decode_cleanup.rs",
            "crates/j2k-metal/src/compute/decode_dispatch/classic_cleanup.rs",
            "crates/j2k-metal/src/compute/decode_dispatch/classic_cleanup/distinct_allocation.rs",
            "crates/j2k-metal/src/compute/decode_dispatch/classic_cleanup/distinct_batch.rs",
            "crates/j2k-metal/src/compute/decode_dispatch/classic_cleanup/distinct_metadata_tests.rs",
            "crates/j2k-metal/src/compute/decode_dispatch/ht_distinct.rs",
            "crates/j2k-metal/src/compute/decode_dispatch/ht_chunks.rs",
            "crates/j2k-metal/src/compute/decode_dispatch/ht_chunks/tests.rs",
            "crates/j2k-metal/src/compute/decode_dispatch/ht_chunks/tests/planning.rs",
            "crates/j2k-metal/src/compute/direct_execute.rs",
            "crates/j2k-metal/src/compute/direct_grayscale_execute.rs",
            "crates/j2k-metal/src/compute/direct_grayscale_execute/allocation.rs",
            "crates/j2k-metal/src/compute/direct_grayscale_execute/component_plane.rs",
            "crates/j2k-metal/src/compute/direct_grayscale_execute/component_plane/execution.rs",
            "crates/j2k-metal/src/compute/direct_grayscale_execute/single.rs",
            "crates/j2k-metal/src/compute/direct_plane_pack.rs",
            "crates/j2k-metal/src/compute/direct_stacked_batch.rs",
            "crates/j2k-metal/src/compute/direct_stacked_batch/command_submission.rs",
            "crates/j2k-metal/src/compute/direct_stacked_batch/command_submission/classic_tier1.rs",
            "crates/j2k-metal/src/compute/direct_stacked_batch/command_submission/ht_tier1.rs",
            "crates/j2k-metal/src/compute/direct_stacked_batch/repeated_grayscale/execution.rs",
            "crates/j2k-metal/src/compute/direct_stacked_batch/resources.rs",
            "crates/j2k-metal/src/compute/direct_stacked_batch/validation.rs",
            "crates/j2k-metal/src/compute/lossless_prepare/batch_item.rs",
            "crates/j2k-metal/src/compute/lossless_prepare/single.rs",
            "crates/j2k-metal/src/compute/resident_tier1/counter_validation/validate.rs",
            "crates/j2k-metal/src/encode/packet_plan.rs",
            "crates/j2k-metal/src/compute/tier1_encode.rs",
        ],
    );

    assert_pattern_checks(&[
        PatternCheck::new("J2K direct batch nested metadata", &sources)
            .required(&[
                "classic J2K Metal cleanup segment metadata",
                "classic J2K MetalDirect distinct color submission",
                "distinct_classic_metadata_honors_exact_cap_and_one_byte_over",
                "HTJ2K MetalDirect distinct chunk submission",
                "HTJ2K Metal packed chunk metadata",
                "packed_ht_chunk_metadata_honors_exact_cap_and_one_byte_over",
                "J2K MetalDirect prepared color component plans",
                "allocate_direct_execution_metadata(",
                "direct_execution_resources_honor_exact_cap_and_one_byte_over",
                "J2K MetalDirect component band metadata",
                "J2K MetalDirect grayscale band metadata",
                "J2K Metal stacked component band metadata",
                "stacked_band_graph_honors_exact_cap_and_one_byte_over",
                "J2K Metal repeated grayscale band metadata",
                "allocate_direct_surface_handles(",
                "J2K Metal stacked component plan references",
                "planned_cpu_input_count(",
                "try_collect_submission_items(",
                "BatchMetadataRequest::of::<ClassicCpuDecodeInput<'_>>",
                "BatchMetadataRequest::of::<HtCpuDecodeInput<'_>>",
                "J2K Metal lossless prepare code-block metadata",
                "J2K Metal resident batch coefficient job metadata",
                "classic J2K Metal encoded segment metadata",
                "J2K Metal classic Tier-1 token-pack length metadata",
                "J2K Metal resident packetization metadata",
                "J2K Metal CPU packetization metadata",
            ])
            .forbidden(&[
                ".collect::<Vec",
                ".collect::<Result<Vec",
                ".collect()",
                "expect(\"preflight validated",
            ]),
    ]);
}

#[test]
fn jpeg_entropy_staging_owns_one_converted_checkpoint_graph() {
    let root = repo_root();
    let sources = read_source_files(
        root,
        &[
            "crates/j2k-jpeg-metal/src/compute/batch_support.rs",
            "crates/j2k-jpeg-metal/src/compute/batch_full/rgb.rs",
            "crates/j2k-jpeg-metal/src/compute/fast_packets/params.rs",
        ],
    );

    assert_pattern_checks(&[PatternCheck::new(
        "JPEG Metal converted checkpoint ownership",
        &sources,
    )
    .required(&[
        "checkpoints: Vec<JpegEntropyCheckpointHost>",
        "actual_checkpoint_count != checkpoint_count",
        "JPEG Metal batch entropy metadata shape mismatch",
        ".map(JpegEntropyCheckpointHost::from)",
        "account_capacity::<JpegEntropyCheckpointHost>(",
        "JPEG Metal entropy checkpoint upload bytes",
        "batch_entropy_shape_mismatch_fails_before_owner_growth",
    ])
    .forbidden(&[
        "checkpoints: Vec<JpegEntropyCheckpointV1>",
        "fn entropy_checkpoint_hosts(",
        ".collect::<Vec",
        ".collect()",
    ])]);
}

#[test]
fn jpeg_viewport_and_texture_owners_keep_aggregate_move_only_contracts() {
    let root = repo_root();
    let viewport = read_source_files(
        root,
        &[
            "crates/j2k-jpeg-metal/src/viewport.rs",
            "crates/j2k-jpeg-metal/src/viewport/model.rs",
            "crates/j2k-jpeg-metal/src/viewport/cpu.rs",
            "crates/j2k-jpeg-metal/src/viewport/tests.rs",
            "crates/j2k-jpeg-metal/src/viewport/tests/budget.rs",
        ],
    );
    let viewport_compose = read_source_files(
        root,
        &["crates/j2k-jpeg-metal/src/compute/viewport_compose.rs"],
    );
    let texture_output = read_source_files(
        root,
        &["crates/j2k-jpeg-metal/src/surface/batch_texture.rs"],
    );

    assert_pattern_checks(&[
        PatternCheck::new("move-only aggregate viewport ownership", &viewport)
            .required(&[
                "/// Move-only planned viewport decode",
                "cpu_viewport_allocation_budget_with_cap(",
                "workload.tiles.capacity()",
                "JPEG Metal CPU viewport live allocation",
                "budget.try_filled(viewport_len",
                "budget.try_filled(max_tile_len",
                "cpu_viewport_live_budget_honors_exact_cap_and_one_byte_over",
                "cpu_viewport_live_budget_reports_count_overflow",
            ])
            .forbidden(&[
                "#[derive(Debug, Clone, PartialEq, Eq)]\n/// Move-only planned viewport decode",
                "let mut viewport = vec!",
                "let mut tile_bytes = vec!",
            ]),
        PatternCheck::new("viewport GPU allocation preflight", &viewport_compose)
            .required(&[
                "validate_viewport_tile_count(tiles, external_live_bytes)?",
                "let mut stage = cached_plane_stage(",
            ])
            .forbidden(&["Vec::with_capacity", "vec!["]),
        PatternCheck::new("shared fallible texture allocation set", &texture_output)
            .required(&[
                "set: Arc<MetalBatchTextureSet>",
                "struct MetalBatchTextureSet",
                "heap_texture_size_and_align(&descriptor)",
                "budget.try_vec(tile_capacity",
                "budget.account_capacity::<u8>(texture_bytes)?",
                "Arc::ptr_eq(&self.set, &other.set)",
            ])
            .forbidden(&[
                "pub struct MetalBatchTextureOutput {\n    textures: Vec<Texture>",
                "let mut textures = Vec::with_capacity(tile_capacity)",
            ]),
    ]);
    assert_eq!(
        viewport
            .matches("Vec::with_capacity((VIEWPORT_TILE_COLS * VIEWPORT_TILE_ROWS) as usize)")
            .count(),
        1,
        "only the fixed 6x2 suggestion may retain raw viewport capacity"
    );
}

#[test]
fn resident_encode_metadata_is_moved_between_lifecycle_stages() {
    let root = repo_root();
    let code_block_routes = [
        (
            "device-resident",
            read_source_files(root, &["crates/j2k-metal/src/encode/device_resident.rs"]),
        ),
        (
            "hybrid-resident",
            read_source_files(root, &["crates/j2k-metal/src/encode/resident_hybrid.rs"]),
        ),
        (
            "batch-resident",
            read_source_files(root, &["crates/j2k-metal/src/encode/resident_prepare.rs"]),
        ),
    ];
    let code_block_owner = read_source_files(root, &["crates/j2k-metal/src/encode/plan.rs"]);
    let code_block_ownership = read_source_files(
        root,
        &[
            "crates/j2k-metal/src/encode/plan.rs",
            "crates/j2k-metal/src/encode/device_resident.rs",
            "crates/j2k-metal/src/encode/resident_hybrid.rs",
            "crates/j2k-metal/src/encode/resident_prepare.rs",
        ],
    );
    let packet_ownership = read_source_files(
        root,
        &[
            "crates/j2k-metal/src/encode/resident_types.rs",
            "crates/j2k-metal/src/encode/resident_submit.rs",
            "crates/j2k-metal/src/compute/resident_tier1/types.rs",
        ],
    );

    assert_pattern_checks(&[
        PatternCheck::new(
            "resident code-block ownership handoff",
            &code_block_ownership,
        )
        .required(&[
            "fn take_code_blocks(&mut self)",
            "std::mem::take(&mut self.code_blocks)",
            "code_block_ownership_transfer_preserves_allocation_without_clone",
        ])
        .forbidden(&["code_blocks.clone()", "component[resolution].clone()"]),
        PatternCheck::new(
            "resident packet metadata ownership handoff",
            &packet_ownership,
        )
        .required(&[
            "fn take_packet_descriptors(&mut self)",
            "fn take_packetization_resolutions(",
            "metadata.take_packet_descriptors()",
            "metadata.take_packetization_resolutions()",
        ])
        .forbidden(&[
            "metadata.packet_descriptors.clone()",
            "metadata.packetization_resolutions.clone()",
            "#[derive(Clone, Debug)]\npub(crate) struct J2kResidentPacketizationResolution",
        ]),
    ]);
    for (route, source) in code_block_routes {
        assert_eq!(
            source.matches(".take_code_blocks()").count(),
            1,
            "the {route} prepare route must move the code-block owner exactly once"
        );
    }
    assert_eq!(
        code_block_owner.matches(".take_code_blocks()").count(),
        1,
        "the ownership regression test must exercise the move exactly once"
    );
}

#[test]
fn resident_and_roi_maps_use_fallible_vec_backed_metadata() {
    let root = repo_root();
    let resident_single = read_source_files(
        root,
        &["crates/j2k-metal/src/compute/resident_codestream/resident_single.rs"],
    );
    let tier2 = read_source_files(
        root,
        &["crates/j2k-metal/src/compute/resident_codestream/tier2_packetization.rs"],
    );
    let direct_roi = read_source_files(root, &["crates/j2k-metal/src/compute/direct_roi.rs"]);

    assert_pattern_checks(&[
        PatternCheck::new("resident single aggregate metadata", &resident_single)
            .required(&[
                "struct ResidentPacketAllocationCounts",
                "J2K Metal resident single packet metadata",
                "BatchMetadataRequest::of::<(u32, u32, usize)>",
                "state_block_offsets",
                ".find(|(state_index, _, _)|",
            ])
            .forbidden(&[
                "HashMap::<u32, (u32, usize)>",
                "Vec::<J2kPacketStateBlock>::new()",
                "Vec::<J2kPacketDescriptor>::with_capacity",
            ]),
        PatternCheck::new("Tier-2 aggregate packet metadata", &tier2)
            .required(&[
                "struct Tier2PacketAllocationCounts",
                "J2K Metal Tier-2 packet metadata",
                "BatchMetadataRequest::of::<u8>(counts.payload_bytes)",
                "state_block_offsets",
                ".find(|(state_index, _, _)|",
            ])
            .forbidden(&[
                "HashMap::<u32, (u32, usize)>",
                "Vec::<J2kPacketStateBlock>::new()",
                "Vec::<J2kPacketDescriptor>::with_capacity",
            ]),
        PatternCheck::new("MetalDirect ROI fallible small maps", &direct_roi)
            .required(&[
                "type BandRequiredRegions = Vec<",
                "J2K MetalDirect ROI band maps",
                "try_reserve_for_push(",
                "required_region(",
            ])
            .forbidden(&[
                "HashMap::<J2kDirectBandId",
                ".entry(band_id)",
                "idwt_output_windows.insert(",
            ]),
    ]);
}
