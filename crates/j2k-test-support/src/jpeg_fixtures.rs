// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fixture JPEGs for decode integration tests. Shared conformance inputs are
//! committed under `j2k-test-support/fixtures/conformance/` and embedded
//! via `include_bytes!` so tests remain hermetic.

mod builders;
mod reference_decode;
mod tables;

pub use builders::*;
pub use tables::*;
