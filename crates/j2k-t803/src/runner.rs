// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fail-closed external-corpus and IUT runner support.

mod archive;
mod cache;
mod cases;
mod cli;
mod cpu;
#[cfg(feature = "cuda-runner")]
mod cuda;
mod encoder;
mod evidence;
mod execute;
#[cfg(feature = "metal-runner")]
mod metal;
mod oracle;

pub use archive::{extract_selected_archive, verify_corpus, ArchiveLimits, RunnerError};
pub use cli::run_cli;

#[cfg(any(
    feature = "cuda-runner",
    all(feature = "metal-runner", target_os = "macos")
))]
const fn adapter_claim(suite: crate::T803Suite) -> &'static str {
    match suite {
        crate::T803Suite::Part1 => "Profile-1 Cclass-1 adapter IUT; Profile-1 Cclass-1HF adapter IUT; Annex G JP2 reader adapter IUT (candidate evidence)",
        crate::T803Suite::Part15 => "HTJ2K DS1-HM Cclass-1h, MMAGB 15 adapter IUT, including DS1-HT, DS0-HM, and DS0-HT subset evidence; HTJ2K Cclass-1HFh, MMAGB 20 adapter IUT; Annex G JPH reader adapter IUT at MMAGB 15 (candidate evidence)",
        crate::T803Suite::All => "Profile-1 Cclass-1 adapter IUT; Profile-1 Cclass-1HF adapter IUT; Annex G JP2 reader adapter IUT; HTJ2K DS1-HM Cclass-1h, MMAGB 15 adapter IUT, including DS1-HT, DS0-HM, and DS0-HT subset evidence; HTJ2K Cclass-1HFh, MMAGB 20 adapter IUT; Annex G JPH reader adapter IUT at MMAGB 15 (candidate evidence)",
    }
}
