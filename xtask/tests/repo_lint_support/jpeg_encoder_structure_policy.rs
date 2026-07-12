// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cohesive ownership ratchets for the baseline JPEG encoder.

use std::fs;

use super::{assert_contains_all, assert_not_contains_all, repo_root};

fn read(relative: &str) -> String {
    fs::read_to_string(repo_root().join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"))
}

#[test]
fn jpeg_encoder_modules_keep_cohesive_ownership() {
    let facade = read("crates/j2k-jpeg/src/encoder.rs");
    assert!(
        facade.lines().count() <= 260,
        "encoder.rs must remain a thin public-contract and orchestration facade"
    );
    assert_contains_all(
        "JPEG encoder facade",
        &facade,
        &[
            "mod api;",
            "mod planning;",
            "mod profile;",
            "mod sample_planes;",
            "mod transform;",
            "pub use self::api::{",
            "fn encode_jpeg_baseline_cpu(",
        ],
    );
    assert_not_contains_all(
        "JPEG encoder facade",
        &facade,
        &[
            "pub enum JpegEncodeError",
            "fn try_vec_with_live_budget",
            "fn rgb_to_ycbcr",
            "fn fdct_quantize",
            "struct BitWriter",
            "fn emit_cpu_encode_profile",
            "mod tests {",
        ],
    );

    assert_module_sizes();
    assert_module_owners();
    assert_shared_entropy_dependency_direction();
    assert_shared_allocation_ownership();
}

fn assert_module_sizes() {
    for (module, ceiling) in [
        ("api.rs", 190usize),
        ("planning.rs", 180),
        ("profile.rs", 100),
        ("sample_planes.rs", 220),
        ("transform.rs", 125),
        ("tests.rs", 260),
    ] {
        let source = read(&format!("crates/j2k-jpeg/src/encoder/{module}"));
        assert!(
            source.lines().count() <= ceiling,
            "encoder/{module} has {} lines; cohesive ownership ceiling is {ceiling}",
            source.lines().count()
        );
    }
}

fn assert_shared_entropy_dependency_direction() {
    let api = read("crates/j2k-jpeg/src/encoder/api.rs");
    let facade = read("crates/j2k-jpeg/src/encoder.rs");
    let entropy = read("crates/j2k-jpeg/src/baseline_entropy.rs");
    let encode_contract = read("crates/j2k-jpeg/src/baseline_encode_contract.rs");
    let dct_contract = read("crates/j2k-jpeg/src/dct_contract.rs");
    let transcode = read("crates/j2k-jpeg/src/transcode.rs");
    let coefficients = read("crates/j2k-jpeg/src/transcode/validation/coefficients.rs");

    assert_contains_all(
        "shared baseline entropy owner",
        &entropy,
        &["struct BitWriter", "fn encode_block", "fn magnitude"],
    );
    assert_contains_all(
        "independent JPEG encode error owner",
        &encode_contract,
        &[
            "pub enum JpegBackend",
            "pub enum JpegSubsampling",
            "pub enum JpegEncodeError",
            "crate::dct_contract::JpegDctImageError",
        ],
    );
    assert_contains_all(
        "independent DCT contract owner",
        &dct_contract,
        &["pub enum JpegDctCodingMode", "pub enum JpegDctImageError"],
    );
    assert_contains_all(
        "transcode shared entropy imports",
        &transcode,
        &["crate::baseline_entropy::{encode_block, BitWriter}"],
    );
    assert_contains_all(
        "DCT validation shared entropy imports",
        &coefficients,
        &["crate::baseline_entropy::magnitude"],
    );
    for (label, source) in [
        ("JPEG encoder API", api.as_str()),
        ("JPEG encoder facade", facade.as_str()),
        ("shared baseline entropy", entropy.as_str()),
        ("JPEG encode contract", encode_contract.as_str()),
        ("DCT contract", dct_contract.as_str()),
    ] {
        assert_not_contains_all(
            label,
            source,
            &["crate::transcode", "super::bitstream", "mod bitstream;"],
        );
    }
}

fn assert_shared_allocation_ownership() {
    let allocation = read("crates/j2k-jpeg/src/allocation.rs");
    let planning = read("crates/j2k-jpeg/src/encoder/planning.rs");
    let sample_planes = read("crates/j2k-jpeg/src/encoder/sample_planes.rs");

    assert_contains_all(
        "shared JPEG allocation owner",
        &allocation,
        &[
            "enum AllocationBudgetError",
            "fn try_new_vec_with_live_budget",
        ],
    );
    assert_contains_all(
        "JPEG encode allocation mapping boundary",
        &planning,
        &["AllocationBudgetError", "fn map_allocation_budget_error"],
    );
    assert_contains_all(
        "JPEG sample-plane shared allocation use",
        &sample_planes,
        &["try_new_vec_with_live_budget"],
    );
    assert_not_contains_all(
        "JPEG encoder planning",
        &planning,
        &["fn try_vec_with_live_budget", "try_reserve_exact"],
    );
}

fn assert_module_owners() {
    let api = read("crates/j2k-jpeg/src/encoder/api.rs");
    let contract = read("crates/j2k-jpeg/src/baseline_encode_contract.rs");
    assert_contains_all(
        "JPEG encoder public contracts",
        &contract,
        &[
            "pub enum JpegBackend",
            "pub enum JpegSubsampling",
            "pub struct JpegEncodeOptions",
            "pub enum JpegSamples<'a>",
            "pub struct EncodedJpeg",
            "pub enum JpegEncodeError",
        ],
    );
    assert_contains_all(
        "JPEG encoder API re-export",
        &api,
        &["pub use crate::baseline_encode_contract::{"],
    );

    let planning = read("crates/j2k-jpeg/src/encoder/planning.rs");
    assert_contains_all(
        "JPEG encoder capacity planning",
        &planning,
        &[
            "struct CpuEncodeCapacityPlan",
            "checked_cpu_encode_capacity_plan",
            "checked_sample_byte_len",
            "component_plane_capacity_bytes",
        ],
    );

    let sample_planes = read("crates/j2k-jpeg/src/encoder/sample_planes.rs");
    assert_contains_all(
        "JPEG encoder sample planes",
        &sample_planes,
        &[
            "validate_sample_layout",
            "component_planes",
            "rgb_to_ycbcr",
            "sample_block",
        ],
    );

    let bitstream = read("crates/j2k-jpeg/src/baseline_entropy.rs");
    assert_contains_all(
        "JPEG encoder entropy bitstream",
        &bitstream,
        &[
            "struct BitWriter",
            "encode_block",
            "write_huffman_symbol",
            "magnitude",
        ],
    );

    let transform = read("crates/j2k-jpeg/src/encoder/transform.rs");
    assert_contains_all(
        "JPEG encoder transform",
        &transform,
        &["fdct_quantize", "cosine_table"],
    );
}
