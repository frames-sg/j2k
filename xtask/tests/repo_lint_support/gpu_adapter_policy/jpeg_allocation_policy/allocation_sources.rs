// SPDX-License-Identifier: MIT OR Apache-2.0

//! Source loading for the JPEG host-allocation policy matrix.

use std::{fs, path::Path};

use super::{JpegAllocationSources, JpegEncodeAllocationSources};
use crate::repo_lint_support::repo_root;

impl JpegAllocationSources {
    pub(super) fn read() -> Self {
        let root = repo_root();
        let checkpoints = CheckpointSourceGroup::read(root);
        let fast_packet = FastPacketSourceGroup::read(root);
        Self {
            checkpoint: checkpoints.facade,
            checkpoint_allocation: checkpoints.allocation,
            checkpoint_build: checkpoints.build,
            checkpoint_cache: checkpoints.cache,
            checkpoint_cache_allocation: checkpoints.cache_allocation,
            checkpoint_eoi: checkpoints.eoi,
            checkpoint_planning: checkpoints.planning,
            checkpoint_tests: checkpoints.tests,
            checkpoint_allocation_tests: checkpoints.allocation_tests,
            checkpoint_build_tests: checkpoints.build_tests,
            checkpoint_eoi_tests: checkpoints.eoi_tests,
            fast_packet: fast_packet.facade,
            fast_packet_types: fast_packet.types,
            fast_packet_allocation: fast_packet.allocation,
            fast_packet_build_owner: fast_packet.build_owner,
            fast_packet_build_gray: fast_packet.build_gray,
            fast_packet_build_materialization: fast_packet.build_materialization,
            fast_packet_build: fast_packet.build,
            fast_packet_checkpoints: fast_packet.checkpoints,
            fast_packet_entropy: fast_packet.entropy,
            fast_packet_header: fast_packet.header,
            fast_packet_tests: fast_packet.tests,
            fast_packet_allocation_tests: fast_packet.allocation_tests,
            fast_packet_behavior_tests: fast_packet.behavior_tests,
            fast_packet_checkpoint_tests: fast_packet.checkpoint_tests,
            fast_packet_source_tests: fast_packet.source_tests,
            device_plan: read_source(
                root,
                "crates/j2k-jpeg/src/adapter/device_plan.rs",
                "JPEG device-plan source",
            ),
            decoder_sequential: read_source(
                root,
                "crates/j2k-jpeg/src/decoder/sequential.rs",
                "JPEG sequential decoder source",
            ),
            owned_decode: read_source(
                root,
                "crates/j2k-jpeg-cuda/src/owned_decode.rs",
                "JPEG CUDA owned-decode source",
            ),
            owned_decode_plan: read_source(
                root,
                "crates/j2k-jpeg-cuda/src/owned_decode/plan.rs",
                "JPEG CUDA owned-decode plan source",
            ),
            owned_decode_tests: read_source(
                root,
                "crates/j2k-jpeg-cuda/src/owned_decode/tests.rs",
                "JPEG CUDA owned-decode tests",
            ),
        }
    }
}

struct CheckpointSourceGroup {
    facade: String,
    allocation: String,
    build: String,
    cache: String,
    cache_allocation: String,
    eoi: String,
    planning: String,
    tests: String,
    allocation_tests: String,
    build_tests: String,
    eoi_tests: String,
}

impl CheckpointSourceGroup {
    fn read(root: &Path) -> Self {
        let source = |relative, label| read_source(root, relative, label);
        Self {
            facade: source("crates/j2k-jpeg/src/internal/checkpoint.rs", "checkpoint"),
            allocation: source(
                "crates/j2k-jpeg/src/internal/checkpoint/allocation.rs",
                "checkpoint allocation",
            ),
            build: source(
                "crates/j2k-jpeg/src/internal/checkpoint/build.rs",
                "checkpoint build",
            ),
            cache: source(
                "crates/j2k-jpeg/src/internal/checkpoint/cache.rs",
                "checkpoint cache",
            ),
            cache_allocation: source(
                "crates/j2k-jpeg/src/internal/checkpoint/cache/allocation.rs",
                "checkpoint cache allocation",
            ),
            eoi: source(
                "crates/j2k-jpeg/src/internal/checkpoint/eoi.rs",
                "checkpoint EOI",
            ),
            planning: source(
                "crates/j2k-jpeg/src/internal/checkpoint/planning.rs",
                "checkpoint planning",
            ),
            tests: source(
                "crates/j2k-jpeg/src/internal/checkpoint/tests.rs",
                "checkpoint tests",
            ),
            allocation_tests: source(
                "crates/j2k-jpeg/src/internal/checkpoint/tests/allocation.rs",
                "checkpoint allocation tests",
            ),
            build_tests: source(
                "crates/j2k-jpeg/src/internal/checkpoint/tests/build.rs",
                "checkpoint build tests",
            ),
            eoi_tests: source(
                "crates/j2k-jpeg/src/internal/checkpoint/tests/eoi.rs",
                "checkpoint EOI tests",
            ),
        }
    }
}

struct FastPacketSourceGroup {
    facade: String,
    types: String,
    allocation: String,
    build_owner: String,
    build_gray: String,
    build_materialization: String,
    build: String,
    checkpoints: String,
    entropy: String,
    header: String,
    tests: String,
    allocation_tests: String,
    behavior_tests: String,
    checkpoint_tests: String,
    source_tests: String,
}

impl FastPacketSourceGroup {
    fn read(root: &Path) -> Self {
        let source = |relative, label| read_source(root, relative, label);
        let build_owner = source(
            "crates/j2k-jpeg/src/adapter/fast_packet/build.rs",
            "fast packet build owner",
        );
        let build_gray = source(
            "crates/j2k-jpeg/src/adapter/fast_packet/build/gray.rs",
            "fast packet grayscale build",
        );
        let build_materialization = source(
            "crates/j2k-jpeg/src/adapter/fast_packet/build/materialization.rs",
            "fast packet materialization accounting",
        );
        let build = build_owner.clone();
        Self {
            facade: source("crates/j2k-jpeg/src/adapter/fast_packet.rs", "fast packet"),
            types: source(
                "crates/j2k-jpeg/src/adapter/fast_packet/types.rs",
                "fast packet types",
            ),
            allocation: source(
                "crates/j2k-jpeg/src/adapter/fast_packet/allocation.rs",
                "fast packet allocation",
            ),
            build_owner,
            build_gray,
            build_materialization,
            build,
            checkpoints: source(
                "crates/j2k-jpeg/src/adapter/fast_packet/checkpoints.rs",
                "fast packet checkpoints",
            ),
            entropy: source(
                "crates/j2k-jpeg/src/adapter/fast_packet/entropy.rs",
                "fast packet entropy",
            ),
            header: source(
                "crates/j2k-jpeg/src/adapter/fast_packet/header.rs",
                "fast packet header",
            ),
            tests: source(
                "crates/j2k-jpeg/src/adapter/fast_packet/tests.rs",
                "fast packet tests",
            ),
            allocation_tests: source(
                "crates/j2k-jpeg/src/adapter/fast_packet/tests/allocation.rs",
                "fast packet allocation tests",
            ),
            behavior_tests: source(
                "crates/j2k-jpeg/src/adapter/fast_packet/tests/behavior.rs",
                "fast packet behavior tests",
            ),
            checkpoint_tests: source(
                "crates/j2k-jpeg/src/adapter/fast_packet/tests/checkpoints.rs",
                "fast packet checkpoint tests",
            ),
            source_tests: source(
                "crates/j2k-jpeg/src/adapter/fast_packet/tests/source.rs",
                "fast packet source tests",
            ),
        }
    }
}

impl JpegEncodeAllocationSources {
    pub(super) fn read() -> Self {
        let root = repo_root();
        let read = |relative: &str| {
            fs::read_to_string(root.join(relative))
                .unwrap_or_else(|error| panic!("read {relative}: {error}"))
        };
        let orchestrate_batch_owner =
            read("crates/j2k-jpeg/src/adapter/baseline_encode/orchestrate/batch.rs");
        let orchestrate_batch_group =
            read("crates/j2k-jpeg/src/adapter/baseline_encode/orchestrate/batch/group.rs");
        let orchestrate_batch = orchestrate_batch_owner.clone();
        let planning_owner = read("crates/j2k-jpeg/src/adapter/baseline_encode/planning.rs");
        let planning_batch = read("crates/j2k-jpeg/src/adapter/baseline_encode/planning/batch.rs");
        let planning = planning_owner.clone();
        Self {
            encode_allocation: read("crates/j2k-jpeg/src/adapter/baseline_encode/allocation.rs"),
            encode_allocation_tests: read(
                "crates/j2k-jpeg/src/adapter/baseline_encode/allocation/tests.rs",
            ),
            encoded_output: read("crates/j2k-jpeg/src/encoded_output.rs"),
            encoded_output_tests: read("crates/j2k-jpeg/src/encoded_output/tests.rs"),
            encoder: read("crates/j2k-jpeg/src/encoder.rs"),
            encoder_contract: read("crates/j2k-jpeg/src/baseline_encode_contract.rs"),
            encoder_planning: read("crates/j2k-jpeg/src/encoder/planning.rs"),
            encoder_tests: read("crates/j2k-jpeg/src/encoder/tests.rs"),
            baseline_entropy: read("crates/j2k-jpeg/src/baseline_entropy.rs"),
            shared_allocation: read("crates/j2k-jpeg/src/allocation.rs"),
            entropy: read("crates/j2k-jpeg/src/encoder/entropy.rs"),
            entropy_restart: read("crates/j2k-jpeg/src/encoder/entropy/restart.rs"),
            entropy_workspace: read("crates/j2k-jpeg/src/encoder/entropy/workspace.rs"),
            frame: read("crates/j2k-jpeg/src/adapter/baseline_encode/frame.rs"),
            orchestrate: read("crates/j2k-jpeg/src/adapter/baseline_encode/orchestrate.rs"),
            orchestrate_batch_owner,
            orchestrate_batch_group,
            orchestrate_batch,
            planning_owner,
            planning_batch,
            planning,
            transcode: read("crates/j2k-jpeg/src/transcode.rs"),
            types: read("crates/j2k-jpeg/src/adapter/baseline_encode/types.rs"),
            adapter_tests: read("crates/j2k-jpeg/src/adapter/baseline_encode/tests.rs"),
        }
    }
}

fn read_source(root: &Path, relative: &str, label: &str) -> String {
    fs::read_to_string(root.join(relative))
        .unwrap_or_else(|error| panic!("read JPEG {label} source: {error}"))
}
