// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fixture JPEGs for decode integration tests. Shared conformance inputs are
//! committed under `j2k-test-support/fixtures/conformance/` and embedded
//! via `include_bytes!` so tests remain hermetic.

mod builders;
mod reference_decode;
mod tables;

// These two modules are catalogs: adding a fixture should expose it through
// this test prelude without maintaining a second, hundred-item export list.
// `crate::tests::wildcard_reexports_are_confined_to_the_fixture_catalog` locks
// the exception to these catalog boundaries.
pub use builders::*;
pub use tables::*;
