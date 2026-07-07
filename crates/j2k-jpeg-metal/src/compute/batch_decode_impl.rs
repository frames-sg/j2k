// SPDX-License-Identifier: MIT OR Apache-2.0

// JPEG Metal batch decode implementation is split across:
// - batch_decode_full.rs
// - batch_decode_region.rs
// - batch_decode_entry.rs
// Keep compute.rs include order byte-compatible with the original monolith.
