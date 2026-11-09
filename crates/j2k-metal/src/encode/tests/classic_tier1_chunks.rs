// SPDX-License-Identifier: MIT OR Apache-2.0

//! Classic Tier-1 batches larger than one Metal allocation.

use super::*;

/// Encodes `pixels` with classic block coding on the CPU and through the
/// Metal stage accelerator, requires the same codestream bytes, and returns
/// the number of code blocks given to the Metal Tier-1 stage.
fn assert_classic_lossless_rgb8_matches_cpu(width: u32, height: u32, pixels: &[u8]) -> usize {
    let cpu_options = lossless_options! {
        backend: EncodeBackendPreference::CpuOnly,
        block_coding_mode: J2kBlockCodingMode::Classic,
        validation: J2kEncodeValidation::External,
    };
    let expected = encode_j2k_lossless(
        J2kLosslessSamples::new(pixels, width, height, 3, 8, false).expect("valid CPU samples"),
        &cpu_options,
    )
    .expect("CPU classic lossless encode");
    let mut accelerator = MetalEncodeStageAccelerator::default();
    let actual = encode_j2k_lossless_with_accelerator(
        J2kLosslessSamples::new(pixels, width, height, 3, 8, false).expect("valid Metal samples"),
        &cpu_options.with_backend(EncodeBackendPreference::RequireDevice),
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("Metal classic lossless encode");

    assert!(accelerator.tier1_code_block_dispatches() > 0);
    assert_same_codestream(
        &format!("{width}x{height} classic lossless"),
        &actual.codestream,
        &expected.codestream,
    );
    accelerator.tier1_code_block_attempts()
}

/// Requires `actual` to equal the CPU's `expected` codestream, reporting the
/// first differing byte rather than both buffers.
fn assert_same_codestream(label: &str, actual: &[u8], expected: &[u8]) {
    if actual == expected {
        return;
    }
    let first_difference = actual
        .iter()
        .zip(expected)
        .position(|(actual, expected)| actual != expected);
    panic!(
        "{label} codestream differs from the CPU: lengths {} / {}, first difference at byte \
         {first_difference:?}",
        actual.len(),
        expected.len(),
    );
}

/// Restores this thread's classic Tier-1 buffer limit when dropped.
struct BufferLimitOverride;

impl BufferLimitOverride {
    fn set(limit: usize) -> Self {
        compute::set_classic_tier1_buffer_limit_for_test(limit);
        Self
    }
}

impl Drop for BufferLimitOverride {
    fn drop(&mut self) {
        compute::set_classic_tier1_buffer_limit_for_test(0);
    }
}

/// A 2048x2048 RGB image's classic Tier-1 output buffers (about 1 GB at the
/// conservative per-block capacity) exceed one Metal allocation; the batch
/// must still encode, in several runs, byte for byte as the CPU does.
#[test]
fn metal_classic_lossless_rgb8_2048_encode_matches_cpu_codestream() {
    if !should_run_metal_runtime() {
        return;
    }

    let pixels = j2k_test_support::patterned_rgb8(2048, 2048);
    compute::reset_classic_tier1_runs_for_test();
    assert_classic_lossless_rgb8_matches_cpu(2048, 2048, &pixels);
    assert!(compute::classic_tier1_runs_for_test() > 1);
}

/// The irreversible encode of a 2048x2048 RGB image asks for even more output
/// (about 1.2 GB) and must likewise encode in runs, byte for byte as the CPU
/// does.
#[test]
fn metal_classic_lossy_rgb8_2048_encode_matches_cpu_codestream() {
    if !should_run_metal_runtime() {
        return;
    }

    let pixels = j2k_test_support::patterned_rgb8(2048, 2048);
    let samples =
        J2kLossySamples::new(&pixels, 2048, 2048, 3, 8, false).expect("valid RGB samples");
    let options = J2kLossyEncodeOptions::default()
        .with_block_coding_mode(J2kBlockCodingMode::Classic)
        .with_validation(J2kEncodeValidation::External);
    let expected = j2k::encode_j2k_lossy(
        samples,
        &options
            .clone()
            .with_backend(EncodeBackendPreference::CpuOnly),
    )
    .expect("CPU classic lossy encode");
    compute::reset_classic_tier1_runs_for_test();
    let mut accelerator = MetalEncodeStageAccelerator::default();
    let actual = encode_j2k_lossy_with_accelerator(
        samples,
        &options.with_backend(EncodeBackendPreference::RequireDevice),
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("Metal classic lossy encode");

    assert_eq!(actual.backend, BackendKind::Metal);
    assert!(compute::classic_tier1_runs_for_test() > 1);
    assert_same_codestream(
        "2048x2048 classic lossy",
        &actual.codestream,
        &expected.codestream,
    );
}

/// With a 1 MiB buffer limit a small image's batch splits into many runs,
/// with edge blocks and uneven run lengths; the codestream must not change.
#[test]
fn metal_classic_tier1_runs_under_a_small_buffer_limit_match_cpu_codestream() {
    if !should_run_metal_runtime() {
        return;
    }

    let pixels = j2k_test_support::patterned_rgb8(320, 200);
    let _limit = BufferLimitOverride::set(1 << 20);
    compute::reset_classic_tier1_runs_for_test();
    assert_classic_lossless_rgb8_matches_cpu(320, 200, &pixels);
    assert!(compute::classic_tier1_runs_for_test() > 4);
}

/// A batch that fits one allocation still runs as a single dispatch.
#[test]
fn metal_classic_tier1_batch_within_the_buffer_limit_is_one_run() {
    if !should_run_metal_runtime() {
        return;
    }

    let pixels = j2k_test_support::patterned_rgb8(320, 200);
    compute::reset_classic_tier1_runs_for_test();
    assert_classic_lossless_rgb8_matches_cpu(320, 200, &pixels);
    assert_eq!(compute::classic_tier1_runs_for_test(), 0);
}

/// With a limit below any one code block's buffers, every block runs alone
/// (it still fits the real allocation cap) and the codestream must not change.
#[test]
fn metal_classic_code_blocks_past_the_buffer_limit_run_alone_and_match_cpu_codestream() {
    if !should_run_metal_runtime() {
        return;
    }

    let pixels = j2k_test_support::patterned_rgb8(96, 80);
    let _limit = BufferLimitOverride::set(1024);
    compute::reset_classic_tier1_runs_for_test();
    let code_blocks = assert_classic_lossless_rgb8_matches_cpu(96, 80, &pixels);
    assert!(code_blocks > 1);
    assert_eq!(compute::classic_tier1_runs_for_test(), code_blocks);
}
