// SPDX-License-Identifier: MIT OR Apache-2.0

//! Correctness-gated end-to-end CPU versus Metal HT Tier-1 encode benchmark.
//!
//! Budgeted lossy profiles exercise candidate sets. Reversible lossless
//! profiles are controls because that public path does not accept byte budgets.

#[cfg(target_os = "macos")]
#[path = "htj2k_candidates/case.rs"]
mod case;
#[cfg(target_os = "macos")]
#[path = "htj2k_candidates/runner.rs"]
mod runner;

#[cfg(not(target_os = "macos"))]
fn main() {
    assert!(
        std::env::var_os("J2K_REQUIRE_METAL_BENCH").is_none(),
        "J2K Metal HTJ2K candidate benchmark requires macOS"
    );
    eprintln!("J2K Metal HTJ2K candidate benchmark skipped outside macOS");
}

#[cfg(target_os = "macos")]
fn main() {
    runner::run();
}
