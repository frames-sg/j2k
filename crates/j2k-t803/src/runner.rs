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
